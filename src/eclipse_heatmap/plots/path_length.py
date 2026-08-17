"""Distribution of total-eclipse path lengths (great-circle extent of the totality footprint), from checkpoint data.

Path length is approximated as the great-circle "diameter" of each event's
totality footprint (grid points with magnitude > 1.0): the maximum
distance between any two points in that footprint. Computed via a
spherical convex hull first (scipy), since the true diameter endpoints
must lie on the hull -- checking all pairs among a few dozen hull points
is far cheaper than all pairs among the (potentially thousands of)
raw footprint points.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.checkpoint import EventCheckpoint
from ..models.grid import GridSpec
from ..utils.geo import great_circle_km, to_unit_vectors

THRESHOLD = 1.0
OUTPUT_FILENAME = "eclipse_path_length.png"


def path_length_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> float:
    if lat_deg.size < 2:
        return 0.0
    vecs = to_unit_vectors(lat_deg, lon_deg)
    if lat_deg.size <= 4:
        hull_vecs = vecs
    else:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(vecs)
        hull_vecs = vecs[hull.vertices]

    n = hull_vecs.shape[0]
    best = 0.0
    for i in range(n):
        dists = great_circle_km(hull_vecs[i], hull_vecs[i + 1 :])
        if dists.size:
            best = max(best, float(dists.max()))
    return best


def generate(checkpoints: list[EventCheckpoint], grid: GridSpec, output_png: Path) -> None:
    lengths = []
    dates = []
    for i, cp in enumerate(checkpoints):
        hit = cp.magnitude > THRESHOLD
        if not hit.any():
            continue
        length = path_length_km(grid.lat_flat[hit], grid.lon_flat[hit])
        lengths.append(length)
        dates.append(cp.date)
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(checkpoints)} checkpoints...", flush=True)

    lengths = np.array(lengths)
    print(f"\nTotal eclipses with a measurable path: {len(lengths)}")
    print(f"Path length (km): min={lengths.min():.0f}, max={lengths.max():.0f}, mean={lengths.mean():.0f}, median={np.median(lengths):.0f}")

    longest_idx = int(np.argmax(lengths))
    print(f"Longest path: {dates[longest_idx]} at {lengths[longest_idx]:.0f} km")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.hist(lengths, bins=50, color="#55a868", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Path length (km, great-circle extent of totality footprint)")
    ax.set_ylabel("Number of total eclipses")
    ax.set_title(
        f"Total Solar Eclipse Path Length Distribution ({checkpoints[0].date} to {checkpoints[-1].date})",
        fontsize=14,
        fontweight="bold",
    )
    ax.axvline(lengths.mean(), color="black", linewidth=1, linestyle="--", label=f"mean = {lengths.mean():.0f} km")
    ax.legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")
