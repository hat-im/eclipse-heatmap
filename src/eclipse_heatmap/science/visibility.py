"""Vectorized per-grid-point eclipse visibility: geocentric Sun/Moon per step + parallax correction, no per-point ephemeris."""

from __future__ import annotations

from multiprocessing import get_context
from multiprocessing.pool import Pool
from typing import TYPE_CHECKING

import numpy as np
from skyfield.api import wgs84
from skyfield.framelib import itrs

from ..models.eclipse_type import EclipseType
from ..utils.geo import haversine_deg
from .geometry import classify_eclipse, moon_angular_radius_deg, sun_angular_radius_deg

if TYPE_CHECKING:
    from skyfield.timelib import Time, Timescale


def subsolar_point(astro_sun, t: "Time") -> tuple[float, float]:
    """Geographic (lat, lon) directly beneath the Sun at time t: where hour angle = 0."""
    ra, dec, _ = astro_sun.radec(epoch="date")
    lon = ((ra.hours - t.gast) * 15.0 + 180.0) % 360.0 - 180.0
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
    """(max_magnitude, best_type) per point over [t_start_tt, t_end_tt] TT; plain floats so it pickles across workers."""
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    n_points = lat_grid.size

    max_magnitude = np.zeros(n_points, dtype=np.float64)
    best_type = np.zeros(n_points, dtype=np.int8)

    if n_points == 0:
        return max_magnitude, best_type

    step_days = time_step_seconds / 86400.0
    n_steps = max(1, int(round((t_end_tt - t_start_tt) / step_days)) + 1)
    step_tts = np.linspace(t_start_tt, t_end_tt, n_steps)

    ecef_km = wgs84.latlon(lat_grid, lon_grid).itrs_xyz.km  # fixed per point; rotated into GCRS once per step

    for step_tt in step_tts:
        t = ts.tt_jd(step_tt)

        observer = earth.at(t)
        astro_sun = observer.observe(sun).apparent()
        sub_lat, sub_lon = subsolar_point(astro_sun, t)

        # Exact geocentric solar altitude: 90 - angular distance from the
        # subsolar point. Solar parallax (~8.8") is negligible here.
        solar_altitude_deg = 90.0 - haversine_deg(lat_grid, lon_grid, sub_lat, sub_lon)
        day_idx = np.where(solar_altitude_deg > 0.0)[0]
        if day_idx.size == 0:
            continue

        astro_moon = observer.observe(moon).apparent()
        sun_vec_km = astro_sun.position.km
        moon_vec_km = astro_moon.position.km

        obs_vec_km = itrs.rotation_at(t).T @ ecef_km[:, day_idx]

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
    """Persistent worker pool for parallel per-eclipse visibility sweeps; create after the ephemeris is cached."""

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
        """Sweeps the grid across workers in contiguous chunks, concatenating results in point order."""
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
