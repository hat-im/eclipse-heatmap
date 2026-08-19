"""Distribution of how many total solar eclipses have touched each grid point, from checkpoint data.

Companion to frequency.py, which maps this count geographically -- this
instead histograms the count values themselves, answering "how many spots
on Earth have seen exactly N total eclipses."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.checkpoint import CheckpointStore
from ..models.eclipse_type import EclipseType
from ..models.grid import GridSpec

OUTPUT_FILENAME = "eclipse_count_distribution.png"


def generate(checkpoints: CheckpointStore, grid: GridSpec, output_png: Path) -> None:
    n_points = grid.lat_flat.size
    count = np.zeros(n_points, dtype=np.int32)
    for cp in checkpoints:
        count += (cp.eclipse_type == EclipseType.TOTAL).astype(np.int32)

    print(f"Count per point: min={count.min()}, max={count.max()}, mean={count.mean():.3f}, median={np.median(count):.1f}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    max_count = int(count.max())
    bins = np.arange(-0.5, max_count + 1.5, 1)
    ax.hist(count, bins=bins, color="#c44e52", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Number of total solar eclipses at that spot")
    ax.set_ylabel("Number of grid points")
    ax.set_xticks(range(0, max_count + 1))
    ax.set_title(
        f"Distribution of Total Eclipse Count per Location ({checkpoints.first_date} to {checkpoints.last_date})",
        fontsize=14,
        fontweight="bold",
    )
    ax.axvline(count.mean(), color="black", linewidth=1, linestyle="--", label=f"mean = {count.mean():.2f}")
    ax.legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")
