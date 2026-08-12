"""Finds solar eclipse events via Skyfield/DE440 ephemeris positions. No eclipse catalog.

Algorithm:
1. New moons: skyfield.almanac.moon_phases finds every new-moon instant
   (exact root-finding on Sun-Moon ecliptic longitude, not a fixed
   29.5-day stride).
2. Refine: scipy.optimize.minimize_scalar finds the true minimum
   geocentric Sun-Moon separation near each new moon (ecliptic-longitude
   conjunction and minimum separation differ slightly due to lunar
   latitude motion).
3. Possibility filter: geocentric_separation < moon_parallax + sun_radius
   + moon_radius is the classical "ecliptic limit" necessary condition
   for an eclipse to be visible anywhere on Earth (Meeus ch. 54;
   Explanatory Supplement to the Astronomical Almanac Sec. 11.4). Fast
   filter only -- visibility.py independently verifies per grid point.
4. Adaptive window: step outward from the conjunction instant until the
   same criterion fails, bounding how long the eclipse could be visible
   anywhere on Earth.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from scipy.optimize import minimize_scalar
from skyfield import almanac
from skyfield.timelib import Time, Timescale

from ..models.eclipse_event import SolarEclipseEvent
from .geometry import moon_angular_radius_deg, moon_horizontal_parallax_deg, sun_angular_radius_deg

logger = logging.getLogger("eclipse_heatmap.eclipses")


def _geocentric_separation_deg(eph, t: Time) -> float:
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    observer = earth.at(t)
    astro_sun = observer.observe(sun).apparent()
    astro_moon = observer.observe(moon).apparent()
    return astro_sun.separation_from(astro_moon).degrees


def _geocentric_geometry(eph, t: Time) -> tuple[float, float, float, float]:
    """(separation_deg, sun_radius_deg, moon_radius_deg, parallax_deg) at time t."""
    earth, sun, moon = eph["earth"], eph["sun"], eph["moon"]
    observer = earth.at(t)
    astro_sun = observer.observe(sun).apparent()
    astro_moon = observer.observe(moon).apparent()
    separation = astro_sun.separation_from(astro_moon).degrees
    sun_r = float(sun_angular_radius_deg(astro_sun.distance().km))
    moon_r = float(moon_angular_radius_deg(astro_moon.distance().km))
    parallax = float(moon_horizontal_parallax_deg(astro_moon.distance().km))
    return separation, sun_r, moon_r, parallax


def _find_new_moon_candidates(eph, ts: Timescale, t0: Time, t1: Time) -> list[Time]:
    f = almanac.moon_phases(eph)
    times, phases = almanac.find_discrete(t0, t1, f)
    return [t for t, p in zip(times, phases) if int(p) == 0]


def _refine_conjunction(eph, ts: Timescale, approx_time: Time, half_window_hours: float = 24.0) -> Time:
    """Minimize geocentric Sun-Moon separation over +/-half_window_hours via Brent's method."""
    center_tt = approx_time.tt
    window_days = half_window_hours / 24.0

    def objective(tt_jd: float) -> float:
        return _geocentric_separation_deg(eph, ts.tt_jd(tt_jd))

    result = minimize_scalar(
        objective,
        bounds=(center_tt - window_days, center_tt + window_days),
        method="bounded",
        options={"xatol": 1e-6},  # ~0.1s precision in TT
    )
    return ts.tt_jd(result.x)


def _adaptive_window(
    eph, ts: Timescale, t_max: Time, limit_deg: float, max_half_span_hours: float = 6.0, step_minutes: float = 5.0
) -> tuple[Time, Time]:
    """Step outward from t_max until the geocentric possibility criterion fails."""
    step_days = step_minutes / (24.0 * 60.0)
    max_steps = int((max_half_span_hours * 60.0) / step_minutes) + 1

    def _extend(direction: int) -> Time:
        t = t_max
        for i in range(1, max_steps + 1):
            candidate = ts.tt_jd(t_max.tt + direction * i * step_days)
            if _geocentric_separation_deg(eph, candidate) >= limit_deg:
                return candidate
            t = candidate
        return t

    return _extend(-1), _extend(1)


def _evaluate_new_moon(eph, ts: Timescale, approx_t: Time) -> SolarEclipseEvent | None:
    refined_t = _refine_conjunction(eph, ts, approx_t)
    separation, sun_r, moon_r, parallax = _geocentric_geometry(eph, refined_t)
    limit_deg = parallax + sun_r + moon_r

    if separation >= limit_deg:
        return None

    search_start, search_end = _adaptive_window(eph, ts, refined_t, limit_deg)
    return SolarEclipseEvent(
        max_time=refined_t,
        date=refined_t.utc_datetime().date(),
        separation_deg=separation,
        sun_radius_deg=sun_r,
        moon_radius_deg=moon_r,
        parallax_deg=parallax,
        search_start=search_start,
        search_end=search_end,
    )


def iter_solar_eclipses(
    eph,
    ts: Timescale,
    start_date: date,
    end_date: date | None = None,
    chunk_days: int = 366,
):
    """Yield confirmed solar eclipse events in chronological order, lazily.

    Searches in chunk_days windows, yielding each confirmed event as
    found -- callers can start processing before the whole range is
    searched. end_date=None searches indefinitely, one chunk at a time,
    until the caller stops iterating (used by main.py for open-ended runs).
    """
    chunk_start = start_date
    n_found = 0
    while True:
        if end_date is not None and chunk_start > end_date:
            return

        chunk_end = chunk_start + timedelta(days=chunk_days)
        if end_date is not None:
            chunk_end = min(chunk_end, end_date)

        t0 = ts.utc(chunk_start.year, chunk_start.month, chunk_start.day)
        t1 = ts.utc(chunk_end.year, chunk_end.month, chunk_end.day, 23, 59, 59)
        logger.info("Searching for new moons between %s and %s ...", chunk_start, chunk_end)
        new_moons = _find_new_moon_candidates(eph, ts, t0, t1)

        for approx_t in new_moons:
            event = _evaluate_new_moon(eph, ts, approx_t)
            if event is not None:
                n_found += 1
                yield event

        logger.info("Chunk %s..%s: %d confirmed eclipse(s) so far.", chunk_start, chunk_end, n_found)

        if end_date is not None and chunk_end >= end_date:
            return
        chunk_start = chunk_end + timedelta(days=1)
