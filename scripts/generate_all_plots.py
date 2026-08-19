"""Runs the checkpoint-data analysis plots (see eclipse_heatmap.plots), streaming checkpoints from disk per plot.

Each plot streams the checkpoint files one at a time, so memory stays flat
no matter how many thousands there are. Safe to run alongside a live
main.py process still writing new checkpoints (read-only).

Usage: python scripts/generate_all_plots.py [checkpoint_dir] [output_dir] [--only NAME [NAME ...]]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eclipse_heatmap.data.checkpoint import CheckpointStore
from eclipse_heatmap.models.grid import generate_grid
from eclipse_heatmap.plots.registry import ANALYSIS_PLOTS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint_dir", type=Path, nargs="?", default=Path("output/checkpoint"))
    p.add_argument("output_dir", type=Path, nargs="?", default=Path("output"))
    p.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="Only run these plots, by module name (e.g. frequency path_length). Default: all.",
    )
    args = p.parse_args()

    plots = ANALYSIS_PLOTS
    if args.only:
        by_name = {m.__name__.rsplit(".", 1)[-1]: m for m in ANALYSIS_PLOTS}
        unknown = set(args.only) - by_name.keys()
        if unknown:
            raise SystemExit(f"Unknown plot(s): {sorted(unknown)}. Available: {sorted(by_name)}")
        plots = [by_name[name] for name in args.only]

    checkpoints = CheckpointStore(args.checkpoint_dir)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found at {args.checkpoint_dir}")
    print(f"Found {len(checkpoints)} checkpoints spanning {checkpoints.first_date} to {checkpoints.last_date}")

    n_points = checkpoints.load(checkpoints.first_date).magnitude.size
    grid = generate_grid(0.25)
    if grid.lat_flat.size != n_points:
        raise SystemExit(f"Checkpoint grid has {n_points} points but the default 0.25 deg grid has {grid.lat_flat.size}")

    failures = []
    for module in plots:
        name = module.__name__.rsplit(".", 1)[-1]
        print(f"\n=== {name} ===")
        start = time.monotonic()
        try:
            module.generate(checkpoints, grid, args.output_dir / module.OUTPUT_FILENAME)
        except SystemExit as e:
            print(f"  skipped: {e}")
            failures.append(name)
            continue
        print(f"  ({time.monotonic() - start:.1f}s)")

    n_ok = len(plots) - len(failures)
    print(f"\n{n_ok}/{len(plots)} plot(s) written to {args.output_dir.resolve()}")
    if failures:
        print(f"Skipped: {', '.join(failures)}")


if __name__ == "__main__":
    main()
