"""Workflow: load ephemeris -> build grid -> sweep eclipses -> checkpoint + blend + save."""

from __future__ import annotations

import argparse
import logging
import shutil
import signal
import sys
from datetime import datetime, timedelta, timezone
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .checkpoint import EventCheckpoint, load_checkpoints, save_event_checkpoint
from .models.eclipse_type import EclipseType
from .models.grid import GridSpec, generate_grid
from .rendering.blend import blend_events
from .rendering.heatmap import save_heatmap_png
from .rendering.raster import save_geotiff, save_numpy, to_raster, to_raster_rgba
from .rendering.table import build_results_dataframe, save_csv
from .science.eclipses import iter_solar_eclipses
from .science.visibility import VisibilityPool
from .utils.logging import setup_logging

logger = logging.getLogger("eclipse_heatmap.pipeline")

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


class RunState:
    """Cumulative per-point tracking, rebuilt by replaying checkpoints on resume."""

    def __init__(self, n_points: int):
        self.assigned = np.zeros(n_points, dtype=bool)
        self.days_until = np.full(n_points, np.nan, dtype=np.float64)
        self.eclipse_dates = np.full(n_points, None, dtype=object)
        self.eclipse_type = np.zeros(n_points, dtype=np.int8)
        self.eclipse_magnitude = np.zeros(n_points, dtype=np.float64)
        self.eclipse_index = np.full(n_points, np.nan, dtype=np.float64)
        self.magnitudes: list[np.ndarray] = []
        self.processed_dates: list = []

    def apply_event(self, today, event_date, magnitude: np.ndarray, eclipse_type: np.ndarray) -> int:
        """Updates cumulative state for one event; returns count of newly-assigned points."""
        self.magnitudes.append(magnitude)
        self.processed_dates.append(event_date)
        event_index = len(self.processed_dates)

        hit_mask = (~self.assigned) & (magnitude > 0.0)
        hit_idx = np.where(hit_mask)[0]
        if hit_idx.size:
            self.assigned[hit_idx] = True
            self.days_until[hit_idx] = (event_date - today).days
            self.eclipse_dates[hit_idx] = event_date.isoformat()
            self.eclipse_type[hit_idx] = eclipse_type[hit_idx]
            self.eclipse_magnitude[hit_idx] = magnitude[hit_idx]
            self.eclipse_index[hit_idx] = event_index
        return hit_idx.size


def _load_resume_state(checkpoint_dir: Path, grid: GridSpec, today) -> tuple[RunState, list]:
    """Replays every stored per-event checkpoint to rebuild cumulative state without re-running the visibility sweep."""
    checkpoints = load_checkpoints(checkpoint_dir)
    n_points = grid.lat_flat.size
    state = RunState(n_points)
    for cp in checkpoints:
        if cp.magnitude.size != n_points:
            raise ValueError(
                f"Checkpoint at {checkpoint_dir} has {cp.magnitude.size} points but the current grid "
                f"has {n_points} -- resolution/bounds must match to resume. Use --fresh to start over."
            )
        state.apply_event(today, cp.date, cp.magnitude, cp.eclipse_type)
    return state, checkpoints


def save_outputs(grid: GridSpec, state: RunState, output_dir: Path) -> None:
    """Recomputes the blend from all stored per-event magnitudes (color scale = actual event count, no hardcoded ceiling)."""
    df = build_results_dataframe(
        grid, state.days_until, state.eclipse_dates, state.eclipse_type, state.eclipse_magnitude, state.eclipse_index
    )
    raster_days = to_raster(state.days_until, grid)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(raster_days, output_dir / "days_until_next_eclipse.npy")
    save_geotiff(raster_days, grid, output_dir / "days_until_next_eclipse.tif")
    save_csv(df, output_dir / "days_until_next_eclipse.csv")

    import matplotlib.pyplot as plt

    accum_rgba = blend_events(state.magnitudes, plt.get_cmap("cividis"))
    raster_rgba = to_raster_rgba(accum_rgba, grid)
    save_heatmap_png(raster_rgba, grid, output_dir / "days_until_next_eclipse.png", state.processed_dates)


def run(args: argparse.Namespace) -> None:
    log = setup_logging(args.log_level)
    log.info("Solar eclipse heat map generator starting.")

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    eph, ts = load_ephemeris(args.data_dir, args.ephemeris_filename)

    grid = generate_grid(args.resolution, lat_bounds=tuple(args.lat_bounds), lon_bounds=tuple(args.lon_bounds))
    n_points = grid.lat_flat.size
    log.info("Grid: %d x %d = %d points at %.3f° resolution.", grid.shape[0], grid.shape[1], n_points, args.resolution)

    today = args.start_date or datetime.now(timezone.utc).date()
    checkpoint_dir = args.output_dir / "checkpoint"

    if args.fresh and checkpoint_dir.is_dir():
        log.info("--fresh: clearing existing checkpoint at %s", checkpoint_dir)
        shutil.rmtree(checkpoint_dir)

    state, checkpoints = _load_resume_state(checkpoint_dir, grid, today)
    search_start = today
    if checkpoints:
        search_start = checkpoints[-1].date + timedelta(days=1)
        log.info(
            "Resuming from checkpoint: %d event(s) already processed (through %s), %d/%d points covered. "
            "Continuing search from %s.",
            len(checkpoints),
            checkpoints[-1].date,
            int(state.assigned.sum()),
            n_points,
            search_start,
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

    log.info("Starting %d worker process(es) for the visibility sweep...", args.workers)
    pool = VisibilityPool(args.workers, str(args.data_dir), args.ephemeris_filename)

    try:
        for event in tqdm(events_iter, desc="Sweeping eclipses", unit="eclipse"):
            if _stop_requested:
                stop_reason = "stopped by user (signal)"
                break

            if state.assigned.all():
                stop_reason = "full global coverage reached"
                break

            magnitude, eclipse_type = pool.compute(
                event.search_start.tt, event.search_end.tt, lat_all, lon_all, args.time_step_seconds
            )
            n_new_events += 1

            save_event_checkpoint(checkpoint_dir, len(state.processed_dates) + 1, event.date, magnitude, eclipse_type)
            n_newly_assigned = state.apply_event(today, event.date, magnitude, eclipse_type)

            log.info(
                "%s (%s): %d newly assigned a first eclipse; %d/%d total covered.",
                event.date,
                EclipseType.name(eclipse_type[magnitude > 0.0].max()) if n_newly_assigned else "none",
                n_newly_assigned,
                int(state.assigned.sum()),
                n_points,
            )

            if n_newly_assigned > 0:
                log.info("Coverage changed -- repainting outputs...")
                save_outputs(grid, state, args.output_dir)
        else:
            stop_reason = f"reached search end date {args.end_date}"
    finally:
        pool.close()

    n_unassigned = int((~state.assigned).sum())
    log.info(
        "Stopped: %s. %d new event(s) processed this run (%d total); %d/%d grid points covered (%d uncovered).",
        stop_reason,
        n_new_events,
        len(state.processed_dates),
        n_points - n_unassigned,
        n_points,
        n_unassigned,
    )

    save_outputs(grid, state, args.output_dir)
    log.info("Done. Outputs written to %s", args.output_dir.resolve())
