"""Command-line argument parsing."""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

DEFAULT_GRID_RESOLUTION_DEG: float = 0.25
DEFAULT_TIME_STEP_SECONDS: float = 60.0
DEFAULT_EPHEMERIS_FILENAME: str = "de440.bsp"
DEFAULT_OUTPUT_DIR: Path = Path("output")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a global heat map of days-until-next-visible-solar-eclipse.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resolution", type=float, default=DEFAULT_GRID_RESOLUTION_DEG, help="Grid spacing, in degrees.")
    p.add_argument(
        "--start-date",
        type=_parse_date,
        default=None,
        help="Search start date, YYYY-MM-DD. Defaults to today (UTC).",
    )
    p.add_argument(
        "--end-date",
        type=_parse_date,
        default=None,
        help="Search end date, YYYY-MM-DD. Default: no fixed end -- keep searching further into "
        "the future, painting the map as each eclipse is confirmed, until either every grid "
        "point has been assigned an eclipse or the run is stopped (Ctrl+C / SIGTERM).",
    )
    p.add_argument(
        "--time-step-seconds",
        type=float,
        default=DEFAULT_TIME_STEP_SECONDS,
        help="Time discretization step used to scan each eclipse's visibility window. "
        "Smaller = more accurate (better chance of catching brief totality) but slower.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to cache the downloaded JPL ephemeris kernel in.",
    )
    p.add_argument(
        "--ephemeris-filename",
        type=str,
        default=DEFAULT_EPHEMERIS_FILENAME,
        help="Ephemeris kernel filename (auto-downloaded into --data-dir if missing). "
        "Use 'de440s.bsp' for a much smaller (~32MB vs ~114MB) kernel of equivalent "
        "accuracy over this project's date range, if disk/bandwidth is a concern.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write outputs into.")
    p.add_argument(
        "--max-eclipses",
        type=int,
        default=None,
        help="Debug/testing aid: stop after processing this many eclipse events.",
    )
    p.add_argument(
        "--lat-bounds",
        type=float,
        nargs=2,
        default=(-90.0, 90.0),
        metavar=("MIN", "MAX"),
        help="Restrict the grid to a latitude range (for quick regional test runs).",
    )
    p.add_argument(
        "--lon-bounds",
        type=float,
        nargs=2,
        default=(-180.0, 180.0),
        metavar=("MIN", "MAX"),
        help="Restrict the grid to a longitude range (for quick regional test runs).",
    )
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    p.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of worker processes for the per-eclipse visibility sweep (the dominant cost). "
        "Each worker loads its own copy of the ephemeris once at startup.",
    )
    return p


def main() -> None:
    from .pipeline import run

    run(build_arg_parser().parse_args())
