"""PNG heat map rendering: a pre-composited RGBA raster (see pipeline.py's blending loop)."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np

from ..models.grid import GridSpec

logger = logging.getLogger("eclipse_heatmap.rendering")


def _date_ticks(processed_dates: list[date], max_index_scale: int, n_ticks: int = 6) -> tuple[list[int], list[str]]:
    n = min(len(processed_dates), max_index_scale)
    if n == 0:
        return [], []
    positions = sorted(set(np.linspace(1, n, num=min(n_ticks, n), dtype=int)))
    labels = [processed_dates[i - 1].strftime("%Y-%m") for i in positions]
    return positions, labels


def save_heatmap_png(
    rgba_raster: np.ndarray,
    grid: GridSpec,
    path: Path,
    max_index_scale: int,
    processed_dates: list[date],
) -> None:
    """Robinson-projection PNG from an already-blended (H, W, 4) RGBA raster.

    Color is the Porter-Duff "over" composite of every eclipse touching each
    point so far, so overlap shows as a genuine color mix. Rendered via
    imshow since the RGBA is pre-composited.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.Robinson())
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

    reference = ScalarMappable(norm=Normalize(vmin=1, vmax=max_index_scale), cmap=plt.get_cmap("cividis"))
    cbar = fig.colorbar(reference, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6)
    positions, labels = _date_ticks(processed_dates, max_index_scale)
    if positions:
        cbar.set_ticks(positions)
        cbar.set_ticklabels(labels, rotation=45, ha="right")
    cbar.ax.tick_params(length=4)
    cbar.set_label("Eclipse date")

    ax.set_title("Solar Eclipse Exposure", fontsize=15, fontweight="bold", pad=12)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote heat map PNG: %s", path)
