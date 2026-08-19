"""Workflow: load ephemeris -> build grid -> sweep eclipses -> checkpoint + blend + save."""

from __future__ import annotations

import argparse
import logging
import shutil
import signal
import sys
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ..data.checkpoint import CheckpointStore
from ..data.raster import save_geotiff, save_numpy, to_raster, to_raster_rgba
from ..data.table import build_results_dataframe, save_csv
from ..models.eclipse_type import EclipseType
from ..models.grid import GridSpec, generate_grid
from ..plots.blend import blend_events
from ..plots.main_map import save_heatmap_png
from ..plots.registry import ANALYSIS_PLOTS
from ..science.eclipses import iter_solar_eclipses
from ..science.visibility import VisibilityPool
from ..utils.astro_date import AstroDate
from ..utils.logging import setup_logging
from .run_state import CheckpointMerger, RunState

logger = logging.getLogger("eclipse_heatmap.logic")

ANALYSIS_PLOT_INTERVAL_YEARS = 1000

_stop_requested = False


def _request_stop(signum, frame) -> None:
    global _stop_requested
    if _stop_requested:
        sys.exit(1)
    _stop_requested = True
    logger.warning("Stop requested (signal %d) -- finishing the current eclipse, then saving and exiting.", signum)


def load_ephemeris(data_dir: Path, filename: str):
    """Auto-downloads the JPL ephemeris kernel and leap-second tables into data_dir on first use."""
    from skyfield.api import Loader

    data_dir.mkdir(parents=True, exist_ok=True)
    loader = Loader(str(data_dir))
    logger.info("Loading ephemeris %s (downloading to %s if not already cached)...", filename, data_dir)
    return loader(filename), loader.timescale()


def save_data_outputs(grid: GridSpec, state: RunState, output_dir: Path) -> None:
    df = build_results_dataframe(
        grid, state.days_until, state.eclipse_dates, state.eclipse_type, state.eclipse_magnitude, state.eclipse_index
    )
    raster_days = to_raster(state.days_until, grid)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(raster_days, output_dir / "days_until_next_eclipse.npy")
    save_geotiff(raster_days, grid, output_dir / "days_until_next_eclipse.tif")
    save_csv(df, output_dir / "days_until_next_eclipse.csv")


def save_image_output(grid: GridSpec, store: CheckpointStore, output_dir: Path, magnitude_threshold: float) -> None:
    """Painting only shows where magnitude exceeds magnitude_threshold, same gate as days-until coverage tracking."""
    import matplotlib.pyplot as plt

    painted = (np.where(cp.magnitude > magnitude_threshold, cp.magnitude, 0.0) for cp in store)
    accum_rgba = blend_events(painted, len(store), grid.lat_flat.size, plt.get_cmap("viridis"))
    raster_rgba = to_raster_rgba(accum_rgba, grid)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap_png(raster_rgba, grid, output_dir / "days_until_next_eclipse.png", list(store.dates))


def save_outputs(grid: GridSpec, state: RunState, store: CheckpointStore, output_dir: Path, magnitude_threshold: float) -> None:
    save_data_outputs(grid, state, output_dir)
    save_image_output(grid, store, output_dir, magnitude_threshold)


def regenerate_analysis_plots(store: CheckpointStore, grid: GridSpec, output_dir: Path) -> None:
    for module in ANALYSIS_PLOTS:
        try:
            module.generate(store, grid, output_dir / module.OUTPUT_FILENAME)
        except SystemExit as e:
            logger.warning("Skipped %s: %s", module.__name__.rsplit(".", 1)[-1], e)


def run(args: argparse.Namespace) -> None:
    log = setup_logging(args.log_level)
    log.info("Solar eclipse heat map generator starting.")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    eph, ts = load_ephemeris(args.data_dir, args.ephemeris_filename)

    grid = generate_grid(args.resolution, lat_bounds=tuple(args.lat_bounds), lon_bounds=tuple(args.lon_bounds))
    n_points = grid.lat_flat.size
    log.info("Grid: %d x %d = %d points at %.3f° resolution.", grid.shape[0], grid.shape[1], n_points, args.resolution)

    today = args.start_date or AstroDate.today_utc()
    checkpoint_dir = args.output_dir / "checkpoint"

    if args.fresh and checkpoint_dir.is_dir():
        log.info("--fresh: clearing existing checkpoint at %s", checkpoint_dir)
        shutil.rmtree(checkpoint_dir)

    threshold = args.magnitude_threshold
    store = CheckpointStore(checkpoint_dir)
    checkpointed_dates = set(store.dates)
    state = RunState(n_points)
    merger = CheckpointMerger(store)

    # Order-independent union of all coverage, for progress/stop-condition only;
    # state.assigned fills in lazily by true date order via merger and can lag behind this.
    known_covered = np.zeros(n_points, dtype=bool)
    for cp in store:
        if cp.magnitude.size != n_points:
            raise ValueError(
                f"Checkpoint {cp.date} has {cp.magnitude.size} points but the current grid has {n_points} -- "
                "resolution/bounds must match to resume. Use --fresh to start over."
            )
        known_covered |= cp.magnitude > threshold

    if args.start_date is not None:
        search_start = args.start_date
    elif store:
        search_start = store.last_date.add_days(1)
    else:
        search_start = today

    merger.apply_up_to(state, today, threshold, search_start)

    if store:
        extending_backward = search_start <= store.first_date
        log.info(
            "Found %d existing checkpoint(s) spanning %s to %s (%d/%d points covered). Searching from %s%s.",
            len(store),
            store.first_date,
            store.last_date,
            int(known_covered.sum()),
            n_points,
            search_start,
            " -- extending backward, already-checkpointed dates will be merged in by true date order"
            if extending_backward
            else "",
        )

    if args.end_date is None:
        log.info(
            "No --end-date set: searching indefinitely into the future, checkpointing after "
            "every eclipse, until full coverage is reached or the run is stopped (Ctrl+C)."
        )

    events_iter = iter_solar_eclipses(eph, ts, search_start, args.end_date)
    if args.max_eclipses is not None:
        events_iter = islice(events_iter, args.max_eclipses)

    lat_all = grid.lat_flat
    lon_all = grid.lon_flat

    stop_reason = "eclipse search exhausted"
    n_new_events = 0
    n_skipped = 0

    log.info("Starting %d worker process(es) for the visibility sweep...", args.workers)
    pool = VisibilityPool(args.workers, str(args.data_dir), args.ephemeris_filename)

    progress = tqdm(
        total=100.0,
        desc="Coverage",
        unit="%",
        bar_format="{l_bar}{bar}| {n:.4f}/{total_fmt}% [{elapsed}<{remaining}]",
    )

    def refresh_progress(postfix: str) -> None:
        progress.set_postfix_str(postfix)
        progress.n = 100.0 * int(known_covered.sum()) / n_points
        progress.refresh()

    refresh_progress(f"{len(state.processed_dates)} eclipses")
    last_analysis_plot_year = search_start.year
    try:
        for event in events_iter:
            if _stop_requested:
                stop_reason = "stopped by user (signal)"
                break

            if known_covered.all() and not args.ignore_full_coverage:
                stop_reason = "full global coverage reached"
                break

            if event.date.year - last_analysis_plot_year >= ANALYSIS_PLOT_INTERVAL_YEARS:
                log.info("%d+ years since last analysis-plot run -- regenerating...", ANALYSIS_PLOT_INTERVAL_YEARS)
                regenerate_analysis_plots(store, grid, args.output_dir)
                last_analysis_plot_year = event.date.year

            if event.date in checkpointed_dates:
                n_skipped += 1
                log.debug("%s: already checkpointed, skipping.", event.date)
                refresh_progress(f"{len(state.processed_dates)} eclipses, {n_skipped} skipped")
                continue

            merger.apply_up_to(state, today, threshold, event.date)

            magnitude, eclipse_type = pool.compute(
                event.search_start.tt, event.search_end.tt, lat_all, lon_all, args.time_step_seconds
            )
            n_new_events += 1

            store.save(event.date, magnitude, eclipse_type)
            checkpointed_dates.add(event.date)
            n_newly_assigned = state.apply_event(today, event.date, magnitude, eclipse_type, threshold)
            prev_covered = int(known_covered.sum())
            known_covered |= magnitude > threshold
            n_newly_covered = int(known_covered.sum()) - prev_covered
            refresh_progress(f"{len(state.processed_dates)} eclipses")

            log.info(
                "%s (%s): +%d (%d/%d covered).",
                event.date,
                EclipseType.name(eclipse_type[magnitude > 0.0].max()) if n_newly_assigned else "none",
                n_newly_covered,
                int(known_covered.sum()),
                n_points,
            )

            if n_newly_assigned > 0:
                save_data_outputs(grid, state, args.output_dir)
                if n_newly_covered > 0:
                    log.info("Repainting outputs...")
                    save_image_output(grid, store, args.output_dir, threshold)
        else:
            stop_reason = f"reached search end date {args.end_date}"
    except ValueError as e:
        stop_reason = (
            f"reached the edge of the ephemeris's valid date range ({args.ephemeris_filename}: {e}) -- "
            "remaining uncovered points may never be reached by this eclipse type/threshold"
        )
    finally:
        progress.close()
        pool.close()

    merger.apply_rest(state, today, threshold)

    n_unassigned = int((~state.assigned).sum())
    log.info(
        "Stopped: %s. %d new event(s) processed this run (%d already-checkpointed skipped, %d total); "
        "%d/%d grid points covered (%d uncovered).",
        stop_reason,
        n_new_events,
        n_skipped,
        len(state.processed_dates),
        n_points - n_unassigned,
        n_points,
        n_unassigned,
    )

    save_outputs(grid, state, store, args.output_dir, threshold)
    log.info("Done. Outputs written to %s", args.output_dir.resolve())
