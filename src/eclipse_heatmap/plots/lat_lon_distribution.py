"""Frequency distribution of the latitudes and longitudes swept by total eclipse paths, from checkpoint data.

For every event, every grid point with magnitude > threshold counts once
toward that point's lat/lon bin -- so the histograms reflect how much
total-eclipse path coverage falls at each latitude/longitude across all
processed events, not just a binary covered/uncovered count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.checkpoint import CheckpointStore
from ..models.grid import GridSpec

THRESHOLD = 1.0
OUTPUT_FILENAME = "eclipse_lat_lon_distribution.png"


def generate(checkpoints: CheckpointStore, grid: GridSpec, output_png: Path) -> None:
    lat_hits: list[np.ndarray] = []
    lon_hits: list[np.ndarray] = []
    for cp in checkpoints:
        hit = cp.magnitude > THRESHOLD
        if hit.any():
            lat_hits.append(grid.lat_flat[hit])
            lon_hits.append(grid.lon_flat[hit])

    if not lat_hits:
        raise SystemExit(f"No point ever exceeded magnitude {THRESHOLD} -- nothing to plot.")

    all_lats = np.concatenate(lat_hits)
    all_lons = np.concatenate(lon_hits)
    print(f"Total (event, point) hits at magnitude > {THRESHOLD}: {all_lats.size}")

    import matplotlib.pyplot as plt

    fig, (ax_lat, ax_lon) = plt.subplots(1, 2, figsize=(16, 6))

    ax_lat.hist(all_lats, bins=np.arange(-90, 91, 2), color="#4c72b0", edgecolor="none")
    ax_lat.set_xlabel("Latitude (deg)")
    ax_lat.set_ylabel("Grid-point hits (summed over all total eclipses)")
    ax_lat.set_title("Latitude Distribution of Total Eclipse Paths")
    ax_lat.set_xlim(-90, 90)
    ax_lat.axvline(0, color="gray", linewidth=0.6, linestyle=":")

    ax_lon.hist(all_lons, bins=np.arange(-180, 181, 2), color="#dd8452", edgecolor="none")
    ax_lon.set_xlabel("Longitude (deg)")
    ax_lon.set_ylabel("Grid-point hits (summed over all total eclipses)")
    ax_lon.set_title("Longitude Distribution of Total Eclipse Paths")
    ax_lon.set_xlim(-180, 180)
    ax_lon.axvline(0, color="gray", linewidth=0.6, linestyle=":")

    fig.suptitle(
        f"Total Solar Eclipse Path Coverage by Latitude/Longitude ({checkpoints.first_date} to {checkpoints.last_date})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")
