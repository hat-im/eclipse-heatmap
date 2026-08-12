"""Command-line argument parsing."""

from __future__ import annotations

import argparse
import os
from datetime import date, datetime
from pathlib import Path

DEFAULT_GRID_RESOLUTION_DEG: float = 0.25
DEFAULT_MAGNITUDE_THRESHOLD: float = 0.01
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
        "--magnitude-threshold",
        type=float,
        default=DEFAULT_MAGNITUDE_THRESHOLD,
        help="Minimum eclipse magnitude (fraction of solar diameter covered) to count as 'visible'.",
    )
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
    p.add_argument(
        "--show-eclipse-paths",
        action="store_true",
        help="Draw a contour line on the PNG marking the '100%% line' -- the boundary of every region "
        "where the first qualifying eclipse reaches magnitude 1.0 (total or annular). Derived from "
        "already-computed data, no extra cost.",
    )
    p.add_argument(
        "--color-by",
        type=str,
        choices=["eclipse_index", "days"],
        default="eclipse_index",
        help="What the PNG heat map's color axis encodes. 'eclipse_index' (default) colors "
        "LINEARLY by which chronological eclipse event first covered each point (1st, 2nd, "
        "3rd, ...), giving every eclipse equal visual weight regardless of the real-world gap "
        "in time before it. 'days' reproduces the original log-scaled, time-based coloring.",
    )
    return p


def main() -> None:
    from .pipeline import run

    run(build_arg_parser().parse_args())
