"""PNG heat map rendering."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..models.eclipse_type import EclipseType
from ..models.grid import GridSpec

logger = logging.getLogger("eclipse_heatmap.rendering")


def _smooth_for_display(array2d: np.ndarray, sigma: float = 1.2, log_scale: bool = False) -> np.ndarray:
    """Gaussian-blur for display only; never touches exported data.

    The raster is piecewise-constant (points sharing a "first eclipse"
    have identical values), so it renders as flat regions with hard
    edges regardless of shading mode -- no shading algorithm can
    interpolate within a constant region. This blurs values (log10-space
    if log_scale, else linear) to turn hard edges into soft transitions.
    NaN cells are filled with their nearest valid neighbor before
    blurring and restored to NaN after, so the no-data boundary stays
    sharp and only interior boundaries soften.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    nan_mask = np.isnan(array2d)
    if nan_mask.all():
        return array2d

    values = np.log10(np.clip(array2d, 0.5, None)) if log_scale else array2d.astype(np.float64)
    if nan_mask.any():
        nearest_idx = distance_transform_edt(nan_mask, return_distances=False, return_indices=True)
        values = values[tuple(nearest_idx)]

    smoothed = gaussian_filter(values, sigma=sigma, mode="nearest")
    if log_scale:
        smoothed = np.power(10.0, smoothed)
    if nan_mask.any():
        smoothed = np.where(nan_mask, np.nan, smoothed)
    return smoothed


def save_heatmap_png(
    days_raster: np.ndarray,
    grid: GridSpec,
    path: Path,
    magnitude_threshold: float,
    generated_on: datetime | None = None,
    eclipse_type_raster: np.ndarray | None = None,
    show_eclipse_paths: bool = False,
    eclipse_index_raster: np.ndarray | None = None,
    color_by: str = "eclipse_index",
) -> None:
    """Robinson-projection PNG heat map: coastlines, borders, colorbar, title, timestamp.

    color_by="eclipse_index" (default): linear scale by chronological
    event rank (1st, 2nd, ...), bounded to [1, N confirmed events].
    color_by="days": log scale by elapsed days.

    show_eclipse_paths + eclipse_type_raster: draws the "100% line"
    (path of totality/annularity) where eclipse_type is TOTAL or
    ANNULAR. Derived from already-computed data, no extra ephemeris work.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import LogLocator, MaxNLocator, ScalarFormatter

    generated_on = generated_on or datetime.now(timezone.utc)

    # Rasters are north-up (flipped); lat_centers must match that row order.
    lat_centers_northup = grid.lat_centers[::-1]

    if color_by == "eclipse_index":
        if eclipse_index_raster is None:
            raise ValueError("color_by='eclipse_index' requires eclipse_index_raster")
        source = eclipse_index_raster
        log_scale = False
        colorbar_label = (
            "First visible eclipse, by chronological order (1 = first event found; "
            "linear scale, magnitude ≥ %.3g)" % magnitude_threshold
        )
        title = "First Visible Solar Eclipse, by Location (Chronological Order)"
    elif color_by == "days":
        source = days_raster
        log_scale = True
        colorbar_label = "Days until next visible solar eclipse (log scale, magnitude ≥ %.3g)" % magnitude_threshold
        title = "Days Until Next Visible Solar Eclipse, by Location"
    else:
        raise ValueError(f"color_by must be 'eclipse_index' or 'days', got {color_by!r}")

    display = _smooth_for_display(source, log_scale=log_scale)
    masked = np.ma.masked_invalid(display)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#bfbfbf")  # not yet covered

    # vmin/vmax from unsmoothed data: colorbar reflects true range even
    # though blurring pulls smoothed values inward from their extremes.
    finite_vals = np.ma.masked_invalid(source).compressed()
    if log_scale:
        # LogNorm needs positive values; "today" (0 days) clips to 1-day color.
        if finite_vals.size:
            vmin = max(1.0, float(finite_vals.min()))
            vmax = max(vmin + 1.0, float(finite_vals.max()))
        else:
            vmin, vmax = 1.0, 2.0
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)
    else:
        if finite_vals.size:
            vmin = float(finite_vals.min())
            vmax = max(vmin + 1.0, float(finite_vals.max()))
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    fig = plt.figure(figsize=(18, 10))
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    # Grey background: gouraud shading leaves masked-corner quads unfilled.
    ax.set_facecolor("#bfbfbf")

    mesh = ax.pcolormesh(
        grid.lon_centers,
        lat_centers_northup,
        masked,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="gouraud",
    )

    ax.coastlines(resolution="50m", linewidth=0.6, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="dimgray")
    ax.gridlines(draw_labels=False, linewidth=0.3, color="gray", alpha=0.5, linestyle=":")

    if show_eclipse_paths and eclipse_type_raster is not None:
        is_central = (eclipse_type_raster >= EclipseType.ANNULAR).astype(float)
        if is_central.max() > 0:
            ax.contour(
                grid.lon_centers,
                grid.lat_centers[::-1],
                is_central,
                levels=[0.5],
                colors="red",
                linewidths=1.1,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
            from matplotlib.lines import Line2D

            path_line_handle = Line2D(
                [0], [0], color="red", linewidth=1.1, label="Path of totality/annularity (100% line)"
            )
            ax.legend(handles=[path_line_handle], loc="lower left", fontsize=9, framealpha=0.8)

    if log_scale:
        cbar = fig.colorbar(
            mesh, ax=ax, orientation="horizontal", pad=0.05, shrink=0.6, extend="neither", ticks=LogLocator(base=10.0)
        )
        cbar.ax.xaxis.set_major_formatter(ScalarFormatter())
    else:
        cbar = fig.colorbar(
            mesh,
            ax=ax,
            orientation="horizontal",
            pad=0.05,
            shrink=0.6,
            extend="neither",
            ticks=MaxNLocator(integer=True, nbins=10),
        )
    cbar.ax.minorticks_off()
    cbar.set_label(colorbar_label)

    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    fig.text(
        0.5,
        0.02,
        f"Generated {generated_on.strftime('%Y-%m-%d %H:%M UTC')} — "
        f"JPL DE440 ephemeris via Skyfield — grid resolution {grid.resolution_deg}° — "
        "grey = not yet covered by any eclipse found so far",
        ha="center",
        fontsize=9,
        color="dimgray",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote heat map PNG: %s", path)
