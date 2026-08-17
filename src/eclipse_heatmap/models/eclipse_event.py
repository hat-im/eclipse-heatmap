"""Confirmed solar eclipse event data."""

from __future__ import annotations

from dataclasses import dataclass

from skyfield.timelib import Time

from ..utils.astro_date import AstroDate


@dataclass(frozen=True)
class SolarEclipseEvent:
    """A candidate solar eclipse, confirmed possible somewhere on Earth.

    max_time: Skyfield Time of minimum geocentric Sun-Moon separation.
    date: UTC calendar date of max_time; used globally as "the" eclipse
        date (see README limitations re: local date).
    separation_deg, sun_radius_deg, moon_radius_deg, parallax_deg: geocentric geometry at max_time.
    search_start, search_end: adaptive window bounds within which the eclipse could be visible somewhere on Earth.
    """

    max_time: Time
    date: AstroDate
    separation_deg: float
    sun_radius_deg: float
    moon_radius_deg: float
    parallax_deg: float
    search_start: Time
    search_end: Time
