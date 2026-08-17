"""Main days-until-next-eclipse map: a pre-composited RGBA raster (see plots/blend.py)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..models.grid import GridSpec
from ..utils.astro_date import AstroDate

logger = logging.getLogger("eclipse_heatmap.plots")


def _date_ticks(processed_dates: list[AstroDate], n_ticks: int = 6) -> tuple[list[int], list[str]]:
    n = len(processed_dates)
    if n == 0:
        return [], []
    positions = sorted(set(np.linspace(1, n, num=min(n_ticks, n), dtype=int)))
    labels = [f"{processed_dates[i - 1].year:04d}-{processed_dates[i - 1].month:02d}" for i in positions]
    return positions, labels


def save_heatmap_png(
    rgba_raster: np.ndarray,
    grid: GridSpec,
    path: Path,
    processed_dates: list[AstroDate],
) -> None:
    """Equal Earth-projection PNG from an already-blended (H, W, 4) RGBA raster.

    Color is the Porter-Duff "over" composite of every eclipse touching each
    point so far, so overlap shows as a genuine color mix. The color scale
    spans exactly [1, len(processed_dates)] -- the true number of events
    processed, not a guessed ceiling.
    """
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

    n_events = max(len(processed_dates), 1)
    reference = ScalarMappable(norm=Normalize(vmin=1, vmax=n_events), cmap=plt.get_cmap("viridis"))

    fig.canvas.draw()
    map_pos = ax.get_position()
    cax = fig.add_axes([map_pos.x0, map_pos.y0 - 0.08, map_pos.width, 0.03])
    cbar = fig.colorbar(reference, cax=cax, orientation="horizontal")
    positions, labels = _date_ticks(processed_dates)
    if positions:
        cbar.set_ticks(positions)
        cbar.set_ticklabels(labels)
    cbar.ax.tick_params(length=4)
    cbar.set_label("Eclipse date")

    if processed_dates:
        title = f"Solar Eclipse Exposure ({processed_dates[0]} to {processed_dates[-1]})"
    else:
        title = "Solar Eclipse Exposure"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote heat map PNG: %s", path)
