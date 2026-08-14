"""Workflow: load ephemeris -> build grid -> sweep eclipses -> blend + save outputs after each one."""

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
from .rendering.raster import save_geotiff, save_numpy, to_raster, to_raster_rgba
from .rendering.table import build_results_dataframe, save_csv
from .science.eclipses import iter_solar_eclipses
from .science.visibility import VisibilityPool
from .utils.logging import setup_logging

logger = logging.getLogger("eclipse_heatmap.pipeline")

_stop_requested = False

FIXED_MAX_INDEX_SCALE = 100


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


def _blend_over(accum_rgba: np.ndarray, event_rgb: np.ndarray, layer_alpha: np.ndarray) -> None:
    """In-place Porter-Duff "over": event_rgb/layer_alpha composited on top of accum_rgba."""
    accum_alpha = accum_rgba[:, 3]
    new_alpha = layer_alpha + accum_alpha * (1.0 - layer_alpha)
    denom = np.where(new_alpha > 1e-12, new_alpha, 1.0)
    new_rgb = (
        event_rgb[None, :] * layer_alpha[:, None]
        + accum_rgba[:, :3] * accum_alpha[:, None] * (1.0 - layer_alpha[:, None])
    ) / denom[:, None]
    accum_rgba[:, :3] = new_rgb
    accum_rgba[:, 3] = new_alpha


def save_outputs(
    grid: GridSpec,
    days_until: np.ndarray,
    eclipse_dates: np.ndarray,
    eclipse_type: np.ndarray,
    eclipse_magnitude: np.ndarray,
    eclipse_index: np.ndarray,
    accum_rgba: np.ndarray,
    processed_dates: list,
    output_dir: Path,
) -> None:
    """Write all output files reflecting the current (possibly partial) results."""
    df = build_results_dataframe(grid, days_until, eclipse_dates, eclipse_type, eclipse_magnitude, eclipse_index)
    raster_days = to_raster(days_until, grid)
    raster_rgba = to_raster_rgba(accum_rgba, grid)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_numpy(raster_days, output_dir / "days_until_next_eclipse.npy")
    save_geotiff(raster_days, grid, output_dir / "days_until_next_eclipse.tif")
    save_csv(df, output_dir / "days_until_next_eclipse.csv")
    save_heatmap_png(
        raster_rgba, grid, output_dir / "days_until_next_eclipse.png", FIXED_MAX_INDEX_SCALE, processed_dates
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
    eclipse_index = np.full(n_points, np.nan, dtype=np.float64)
    accum_rgba = np.zeros((n_points, 4), dtype=np.float64)
    processed_dates: list = []

    lat_all = grid.lat_flat
    lon_all = grid.lon_flat

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("cividis")

    n_events_processed = 0
    stop_reason = "eclipse search exhausted"

    log.info("Starting %d worker process(es) for the visibility sweep...", args.workers)
    pool = VisibilityPool(args.workers, str(args.data_dir), args.ephemeris_filename)

    try:
        for event in tqdm(events_iter, desc="Sweeping eclipses", unit="eclipse"):
            if _stop_requested:
                stop_reason = "stopped by user (signal)"
                break

            if assigned.all():
                stop_reason = "full global coverage reached"
                break

            max_mag, best_type = pool.compute(
                event.search_start.tt, event.search_end.tt, lat_all, lon_all, args.time_step_seconds
            )
            n_events_processed += 1
            processed_dates.append(event.date)

            hit_mask = (~assigned) & (max_mag > 0.0)
            hit_idx = np.where(hit_mask)[0]
            if hit_idx.size:
                assigned[hit_idx] = True
                days_until[hit_idx] = (event.date - today).days
                eclipse_dates[hit_idx] = event.date.isoformat()
                eclipse_type[hit_idx] = best_type[hit_mask]
                eclipse_magnitude[hit_idx] = max_mag[hit_mask]
                eclipse_index[hit_idx] = n_events_processed

            t = min(n_events_processed / FIXED_MAX_INDEX_SCALE, 1.0)
            event_rgb = np.array(cmap(t)[:3])
            layer_alpha = np.clip(max_mag, 0.0, 1.0)
            _blend_over(accum_rgba, event_rgb, layer_alpha)

            log.info(
                "%s (%s): %d newly assigned a first eclipse; %d/%d total covered. Repainting outputs...",
                event.date,
                EclipseType.name(best_type[hit_mask].max()) if hit_idx.size else "none",
                hit_idx.size,
                int(assigned.sum()),
                n_points,
            )

            save_outputs(
                grid,
                days_until,
                eclipse_dates,
                eclipse_type,
                eclipse_magnitude,
                eclipse_index,
                accum_rgba,
                processed_dates,
                args.output_dir,
            )

        else:
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
        accum_rgba,
        processed_dates,
        args.output_dir,
    )
    log.info("Done. Outputs written to %s", args.output_dir.resolve())
