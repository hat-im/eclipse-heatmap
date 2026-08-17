"""Heat map of how many total solar eclipses have touched each grid point, from checkpoint data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.checkpoint import EventCheckpoint
from ..data.raster import to_raster
from ..models.eclipse_type import EclipseType
from ..models.grid import GridSpec

OUTPUT_FILENAME = "total_eclipse_frequency.png"


def generate(checkpoints: list[EventCheckpoint], grid: GridSpec, output_png: Path) -> None:
    n_points = checkpoints[0].magnitude.size
    count = np.zeros(n_points, dtype=np.int32)
    for cp in checkpoints:
        count += (cp.eclipse_type == EclipseType.TOTAL).astype(np.int32)

    print(f"Total-eclipse count per point: min={count.min()}, max={count.max()}, mean={count.mean():.3f}")
    print(f"Never touched by totality: {(count == 0).sum()}/{n_points} points ({100 * (count == 0).mean():.2f}%)")

    raster = to_raster(count.astype(np.float64), grid)

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()
    ax.set_facecolor("#1a1a2e")

    mesh = ax.imshow(
        raster,
        transform=ccrs.PlateCarree(),
        extent=[-180.0, 180.0, -90.0, 90.0],
        origin="upper",
        interpolation="nearest",
        cmap="inferno",
        norm=Normalize(vmin=0, vmax=max(1, count.max())),
    )

    ax.coastlines(resolution="50m", linewidth=0.6, color="white")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="lightgray")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.4, linestyle=":")

    fig.canvas.draw()
    map_pos = ax.get_position()
    cax = fig.add_axes([map_pos.x0, map_pos.y0 - 0.08, map_pos.width, 0.03])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label("Number of total solar eclipses")

    ax.set_title(
        f"Total Solar Eclipse Frequency by Location ({checkpoints[0].date} to {checkpoints[-1].date})",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png}")
