"""Main days-until-next-eclipse map: a pre-composited RGBA raster (see plots/blend.py)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..models.grid import GridSpec
from ..utils.astro_date import AstroDate

logger = logging.getLogger("eclipse_heatmap.plots")


def save_heatmap_png(
    rgba_raster: np.ndarray,
    grid: GridSpec,
    path: Path,
    date_range: tuple[AstroDate, AstroDate] | None,
    quantile_dates: list[AstroDate],
) -> None:
    """Equal Earth PNG from a pre-blended RGBA raster; quantile_dates label the equal-area colorbar."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.EqualEarth())
    ax.set_global()
    ax.set_facecolor("#bfbfbf")

    ax.imshow(
        rgba_raster,
        transform=ccrs.PlateCarree(),
        extent=[-180.0, 180.0, -90.0, 90.0],
        origin="upper",
        interpolation="bilinear",
    )

    ax.coastlines(resolution="50m", linewidth=0.6, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="dimgray")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5, linestyle=":")

    reference = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=plt.get_cmap("viridis"))

    fig.canvas.draw()
    map_pos = ax.get_position()
    cax = fig.add_axes([map_pos.x0, map_pos.y0 - 0.08, map_pos.width, 0.03])
    cbar = fig.colorbar(reference, cax=cax, orientation="horizontal")
    if quantile_dates:
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels([str(d) for d in quantile_dates])
    cbar.ax.tick_params(length=4)
    cbar.set_label("Date of last eclipse exposure (equal-area color scale)")

    if date_range is not None:
        title = f"Solar Eclipse Exposure ({date_range[0]} to {date_range[1]})"
    else:
        title = "Solar Eclipse Exposure"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote heat map PNG: %s", path)
