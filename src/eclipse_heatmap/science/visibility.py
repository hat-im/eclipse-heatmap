"""Per-grid-point eclipse visibility for one SolarEclipseEvent, swept across worker processes.

Numerical approach (vectorized geocentric + parallax, not per-point ephemeris):
1. Discretize the event's adaptive time window into time_step_seconds steps.
2. At each step, compute the Sun's GEOCENTRIC apparent position once (one
   ephemeris query -- light-time, aberration, deflection all included).
   Solar altitude at every grid point is then the exact spherical-geometry
   identity 90 - great_circle_distance(point, subsolar point); no per-point
   ephemeris call needed for the day/night gate.
3. Compute the Sun's and Moon's GEOCENTRIC apparent position vectors once
   per step (two ephemeris queries total, not one per grid point). Each
   point's TOPOCENTRIC vector to Sun/Moon is geocentric_vector minus that
   point's own geocentric position (wgs84.latlon(...).at(t).position,
   itself ephemeris-free -- pure geometry from Earth orientation data).
   This is the standard parallax correction (observer offset << target
   distance) and reproduces Skyfield's own per-point .observe() results
   to within ~0.01 arcsecond -- negligible next to the ~30 arcsecond-scale
   uncertainty already inherent in a 60-second time step. It replaces
   O(points) expensive light-time-iterated ephemeris evaluations per step
   with O(1), turning the dominant cost into cheap vectorized numpy.
4. Classify geometry from the topocentric separation/angular radii, fold
   into running per-point max magnitude and best type.

Coarse time_step_seconds can miss brief totality/annularity at the edge
of a path (lasts at most minutes at any point).

VisibilityPool splits each eclipse's grid sweep across worker processes
(points are independent within one sweep). Each worker loads its own
ephemeris copy in a pool initializer -- Skyfield objects aren't picklable
across the process boundary, so only plain floats/arrays cross it.
"""

from __future__ import annotations

from multiprocessing import get_context
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING

import numpy as np
from skyfield.api import wgs84

from ..models.eclipse_type import EclipseType
from ..utils.geo import haversine_deg
from .geometry import classify_eclipse, moon_angular_radius_deg, sun_angular_radius_deg

if TYPE_CHECKING:
    from skyfield.timelib import Time, Timescale


def subsolar_point(eph, ts: "Timescale", t: "Time") -> tuple[float, float]:
    """Geographic (lat, lon) directly beneath the Sun at time t: where hour angle = 0."""
    earth, sun = eph["earth"], eph["sun"]
    astro_sun = earth.at(t).observe(sun).apparent()
    ra, dec, _ = astro_sun.radec(epoch="date")
    gast_hours = t.gast
    lon = ((ra.hours - gast_hours) * 15.0 + 180.0) % 360.0 - 180.0
    return dec.degrees, lon


def compute_visibility_window(
    eph,
    ts: "Timescale",
    t_start_tt: float,
    t_end_tt: float,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    time_step_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """(max_magnitude, best_type) per point in (lat_grid, lon_grid) over [t_start_tt, t_end_tt] TT.

    Takes plain TT Julian-date floats rather than Skyfield Time objects
    so it runs identically in the main process or a worker process
    without pickling Skyfield objects across the process boundary.
    """
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    n_points = lat_grid.size

    max_magnitude = np.zeros(n_points, dtype=np.float64)
    best_type = np.zeros(n_points, dtype=np.int8)

    if n_points == 0:
        return max_magnitude, best_type

    step_days = time_step_seconds / 86400.0
    n_steps = max(1, int(round((t_end_tt - t_start_tt) / step_days)) + 1)
    step_tts = np.linspace(t_start_tt, t_end_tt, n_steps)

    for step_tt in step_tts:
        t = ts.tt_jd(step_tt)

        sub_lat, sub_lon = subsolar_point(eph, ts, t)

        # Exact geocentric solar altitude: 90 - angular distance from the
        # subsolar point. Solar parallax (~8.8") is negligible here.
        solar_altitude_deg = 90.0 - haversine_deg(lat_grid, lon_grid, sub_lat, sub_lon)
        day_idx = np.where(solar_altitude_deg > 0.0)[0]
        if day_idx.size == 0:
            continue

        # Two ephemeris queries total (not one per point): geocentric
        # apparent Sun/Moon vectors, already including light-time,
        # aberration, and deflection.
        astro_sun = earth.at(t).observe(sun).apparent()
        astro_moon = earth.at(t).observe(moon).apparent()
        sun_vec_km = astro_sun.position.km
        moon_vec_km = astro_moon.position.km

        # Per-point geocentric position: pure geometry (Earth orientation
        # only), no ephemeris call, fully vectorized.
        obs_vec_km = wgs84.latlon(lat_grid[day_idx], lon_grid[day_idx]).at(t).position.km

        # Parallax correction: topocentric = geocentric - observer offset.
        topo_sun_km = sun_vec_km[:, None] - obs_vec_km
        topo_moon_km = moon_vec_km[:, None] - obs_vec_km

        sun_dist_km = np.linalg.norm(topo_sun_km, axis=0)
        moon_dist_km = np.linalg.norm(topo_moon_km, axis=0)
        cos_separation = np.sum(topo_sun_km * topo_moon_km, axis=0) / (sun_dist_km * moon_dist_km)
        separation_deg = np.degrees(np.arccos(np.clip(cos_separation, -1.0, 1.0)))

        sun_r_deg = sun_angular_radius_deg(sun_dist_km)
        moon_r_deg = moon_angular_radius_deg(moon_dist_km)

        magnitude, type_code = classify_eclipse(separation_deg, sun_r_deg, moon_r_deg)

        target_idx = day_idx
        improved = magnitude > max_magnitude[target_idx]
        max_magnitude[target_idx[improved]] = magnitude[improved]
        better_type = type_code > best_type[target_idx]
        best_type[target_idx[better_type]] = type_code[better_type]

    return max_magnitude, best_type


# Set once per worker process by _init_worker.
_worker_eph = None
_worker_ts = None


def _init_worker(data_dir: str, ephemeris_filename: str) -> None:
    global _worker_eph, _worker_ts
    from skyfield.api import Loader

    loader = Loader(data_dir)
    _worker_eph = loader(ephemeris_filename)
    _worker_ts = loader.timescale()


def _worker_task(
    args: tuple[float, float, np.ndarray, np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray]:
    t_start_tt, t_end_tt, lat_chunk, lon_chunk, time_step_seconds = args
    return compute_visibility_window(
        _worker_eph, _worker_ts, t_start_tt, t_end_tt, lat_chunk, lon_chunk, time_step_seconds
    )


class VisibilityPool:
    """Persistent worker pool for parallel per-eclipse visibility sweeps.

    Create once per run, after loading the ephemeris in the main process
    (so worker init reads cached files instead of racing to download
    them). Call compute() once per eclipse event, close() when done.
    """

    def __init__(self, n_workers: int, data_dir: str, ephemeris_filename: str):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        ctx = get_context("spawn")  # avoids inheriting parent's open C-extension state
        self.pool: Pool = ctx.Pool(
            processes=n_workers, initializer=_init_worker, initargs=(data_dir, ephemeris_filename)
        )

    def compute(
        self,
        search_start_tt: float,
        search_end_tt: float,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        time_step_seconds: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sweep (lat_grid, lon_grid) over [search_start_tt, search_end_tt] across workers.

        Splits into up to n_workers contiguous chunks, dispatches one per
        worker, concatenates results in original point order.
        """
        n = lat_grid.size
        if n == 0:
            return np.zeros(0), np.zeros(0, dtype=np.int8)

        n_chunks = max(1, min(self.n_workers, n))
        lat_chunks = np.array_split(lat_grid, n_chunks)
        lon_chunks = np.array_split(lon_grid, n_chunks)
        tasks = [
            (search_start_tt, search_end_tt, lat_c, lon_c, time_step_seconds)
            for lat_c, lon_c in zip(lat_chunks, lon_chunks)
            if lat_c.size
        ]

        results = self.pool.map(_worker_task, tasks)
        max_magnitude = np.concatenate([r[0] for r in results])
        best_type = np.concatenate([r[1] for r in results])
        return max_magnitude, best_type

    def close(self) -> None:
        self.pool.close()
        self.pool.join()

    def __enter__(self) -> "VisibilityPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
