"""Heat map of the average time between total solar eclipses at each grid point, from checkpoint data.

For each point, walks its total-eclipse hits in chronological order and
averages the gaps between consecutive hits. Undefined (masked grey) for
points hit 0 or 1 times -- a "time between" needs at least two events.

Streams through checkpoints once, keeping only a few per-point running
arrays (last-hit date, sum of gaps, gap count) rather than a dense
(events x points) matrix, so memory stays O(n_points) regardless of how
many events are processed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.checkpoint import CheckpointStore
from ..data.raster import to_raster
from ..models.grid import GridSpec

THRESHOLD = 1.0
DAYS_PER_YEAR = 365.25
OUTPUT_FILENAME = "eclipse_interval_heatmap.png"


def generate(checkpoints: CheckpointStore, grid: GridSpec, output_png: Path) -> None:
    n_points = grid.lat_flat.size
    last_hit_ordinal = np.full(n_points, -1, dtype=np.int64)
    sum_gap_days = np.zeros(n_points, dtype=np.float64)
    gap_count = np.zeros(n_points, dtype=np.int64)

    for cp in checkpoints:
        hit = np.where(cp.magnitude > THRESHOLD)[0]
        if hit.size == 0:
            continue
        ordinal = cp.date.to_jd()
        has_prev = last_hit_ordinal[hit] >= 0
        prev_idx = hit[has_prev]
        sum_gap_days[prev_idx] += ordinal - last_hit_ordinal[prev_idx]
        gap_count[prev_idx] += 1
        last_hit_ordinal[hit] = ordinal

    avg_gap_years = np.full(n_points, np.nan, dtype=np.float64)
    has_gap = gap_count > 0
    avg_gap_years[has_gap] = (sum_gap_days[has_gap] / gap_count[has_gap]) / DAYS_PER_YEAR

    finite = avg_gap_years[has_gap]
    print(f"Points with >=2 total eclipses: {has_gap.sum()}/{n_points} ({100 * has_gap.mean():.2f}%)")
    if finite.size == 0:
        raise SystemExit("No point has been touched by 2+ total eclipses yet -- nothing to plot.")
    print(f"Average interval (years): min={finite.min():.1f}, max={finite.max():.1f}, mean={finite.mean():.1f}, median={np.median(finite):.1f}")

    raster = to_raster(avg_gap_years, grid)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()
    ax.set_facecolor("#bfbfbf")

    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="#bfbfbf")
    masked = np.ma.masked_invalid(raster)

    mesh = ax.imshow(
        masked,
        transform=ccrs.PlateCarree(),
        extent=[-180.0, 180.0, -90.0, 90.0],
        origin="upper",
        interpolation="nearest",
        cmap=cmap,
        norm=Normalize(vmin=np.nanmin(avg_gap_years), vmax=np.nanpercentile(avg_gap_years, 98)),
    )

    ax.coastlines(resolution="50m", linewidth=0.6, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="dimgray")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4, linestyle=":")

    fig.canvas.draw()
    map_pos = ax.get_position()
    cax = fig.add_axes([map_pos.x0, map_pos.y0 - 0.08, map_pos.width, 0.03])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="max")
    cbar.set_label("Average years between total solar eclipses (grey = fewer than 2 ever recorded)")

    ax.set_title(
        f"Average Time Between Total Solar Eclipses ({checkpoints.first_date} to {checkpoints.last_date})",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")
