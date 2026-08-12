"""Workflow: load ephemeris -> build grid -> sweep eclipses -> save outputs after each one."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .models.eclipse_type import EclipseType
from .models.grid import GridSpec, generate_grid
from .rendering.heatmap import save_heatmap_png
from .rendering.raster import save_geotiff, save_numpy, to_raster
from .rendering.table import build_results_dataframe, save_csv
from .science.eclipses import iter_solar_eclipses
from .science.visibility import VisibilityPool
from .utils.logging import setup_logging

logger = logging.getLogger("eclipse_heatmap.pipeline")

# Set by the SIGINT/SIGTERM handler; checked between eclipse events so a
# stop lands after the current event finishes, not mid-computation.
_stop_requested = False


def _request_stop(signum, frame) -> None:
    global _stop_requested
    if _stop_requested:
        sys.exit(1)  # second signal: exit immediately
    _stop_requested = True
    logger.warning("Stop requested (signal %d) -- finishing the current eclipse, then saving and exiting.", signum)


def load_ephemeris(data_dir: Path, filename: str):
    """Auto-downloads the JPL ephemeris kernel and leap-second tables into data_dir on first use."""
    from skyfield.api import Loader

    data_dir.mkdir(parents=True, exist_ok=True)
    loader = Loader(str(data_dir))
    logger.info("Loading ephemeris %s (downloading to %s if not already cached)...", filename, data_dir)
    return loader(filename), loader.timescale()


def save_outputs(
    grid: GridSpec,
    days_until: np.ndarray,
    eclipse_dates: np.ndarray,
    eclipse_type: np.ndarray,
    eclipse_magnitude: np.ndarray,
    eclipse_index: np.ndarray,
    output_dir: Path,
    magnitude_threshold: float,
    show_eclipse_paths: bool = False,
    color_by: str = "eclipse_index",
) -> None:
    """Write all four output files reflecting the current (possibly partial) results."""
    df = build_results_dataframe(grid, days_until, eclipse_dates, eclipse_type, eclipse_magnitude, eclipse_index)
    raster_days = to_raster(days_until, grid)
    raster_type = to_raster(eclipse_type, grid)
    raster_index = to_raster(eclipse_index, grid)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(raster_days, output_dir / "days_until_next_eclipse.npy")
    save_geotiff(raster_days, grid, output_dir / "days_until_next_eclipse.tif")
    save_csv(df, output_dir / "days_until_next_eclipse.csv")
    save_heatmap_png(
        raster_days,
        grid,
        output_dir / "days_until_next_eclipse.png",
        magnitude_threshold=magnitude_threshold,
        eclipse_type_raster=raster_type,
        show_eclipse_paths=show_eclipse_paths,
        eclipse_index_raster=raster_index,
        color_by=color_by,
    )


def run(args: argparse.Namespace) -> None:
    log = setup_logging(args.log_level)
    log.info("Solar eclipse heat map generator starting.")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    eph, ts = load_ephemeris(args.data_dir, args.ephemeris_filename)

    grid = generate_grid(args.resolution, lat_bounds=tuple(args.lat_bounds), lon_bounds=tuple(args.lon_bounds))
    n_points = grid.lat_flat.size
    log.info("Grid: %d x %d = %d points at %.3f° resolution.", grid.shape[0], grid.shape[1], n_points, args.resolution)
    if args.end_date is None:
        log.info(
            "No --end-date set: searching indefinitely into the future, painting the map after "
            "every eclipse, until full coverage is reached or the run is stopped (Ctrl+C)."
        )

    today = args.start_date or datetime.now(timezone.utc).date()
    events_iter = iter_solar_eclipses(eph, ts, today, args.end_date)
    if args.max_eclipses is not None:
        events_iter = islice(events_iter, args.max_eclipses)

    assigned = np.zeros(n_points, dtype=bool)
    days_until = np.full(n_points, np.nan, dtype=np.float64)
    eclipse_dates = np.full(n_points, None, dtype=object)
    eclipse_type = np.zeros(n_points, dtype=np.int8)
    eclipse_magnitude = np.zeros(n_points, dtype=np.float64)
    # 1-based index, in processing order, of the event that first covered
    # each point -- what the PNG colors by default. NaN until assigned.
    eclipse_index = np.full(n_points, np.nan, dtype=np.float64)

    lat_all = grid.lat_flat
    lon_all = grid.lon_flat

    n_events_processed = 0
    stop_reason = "eclipse search exhausted"

    log.info("Starting %d worker process(es) for the visibility sweep...", args.workers)
    pool = VisibilityPool(args.workers, str(args.data_dir), args.ephemeris_filename)

    try:
        for event in tqdm(events_iter, desc="Sweeping eclipses", unit="eclipse"):
            if _stop_requested:
                stop_reason = "stopped by user (signal)"
                break

            remaining_idx = np.where(~assigned)[0]
            if remaining_idx.size == 0:
                stop_reason = "full global coverage reached"
                break

            max_mag, best_type = pool.compute(
                event.search_start.tt,
                event.search_end.tt,
                lat_all[remaining_idx],
                lon_all[remaining_idx],
                args.time_step_seconds,
            )
            n_events_processed += 1

            hit_mask = max_mag >= args.magnitude_threshold
            hit_idx = remaining_idx[hit_mask]
            if hit_idx.size:
                assigned[hit_idx] = True
                days_until[hit_idx] = (event.date - today).days
                eclipse_dates[hit_idx] = event.date.isoformat()
                eclipse_type[hit_idx] = best_type[hit_mask]
                eclipse_magnitude[hit_idx] = max_mag[hit_mask]
                eclipse_index[hit_idx] = n_events_processed

            log.info(
                "%s (%s): %d/%d remaining points newly assigned; %d/%d total covered. Repainting outputs...",
                event.date,
                EclipseType.name(best_type[hit_mask].max()) if hit_idx.size else "none",
                hit_idx.size,
                remaining_idx.size,
                int(assigned.sum()),
                n_points,
            )

            # Render after every eclipse, not just ones that assigned new
            # points, so outputs on disk always reflect the most recently
            # evaluated eclipse.
            save_outputs(
                grid,
                days_until,
                eclipse_dates,
                eclipse_type,
                eclipse_magnitude,
                eclipse_index,
                args.output_dir,
                args.magnitude_threshold,
                args.show_eclipse_paths,
                args.color_by,
            )

        else:
            # Loop exhausted without a break -- only possible when --end-date bounds the search.
            stop_reason = f"reached search end date {args.end_date}"
    finally:
        pool.close()

    n_unassigned = int((~assigned).sum())
    log.info(
        "Stopped: %s. Processed %d eclipse event(s); %d/%d grid points covered (%d uncovered).",
        stop_reason,
        n_events_processed,
        n_points - n_unassigned,
        n_points,
        n_unassigned,
    )

    save_outputs(
        grid,
        days_until,
        eclipse_dates,
        eclipse_type,
        eclipse_magnitude,
        eclipse_index,
        args.output_dir,
        args.magnitude_threshold,
        args.show_eclipse_paths,
        args.color_by,
    )
    log.info("Done. Outputs written to %s", args.output_dir.resolve())
