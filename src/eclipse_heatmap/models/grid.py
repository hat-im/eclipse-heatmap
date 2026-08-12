"""Global lat/lon grid of cell centers.

Cell-center convention: for resolution r, latitude runs -90+r/2 .. 90-r/2
and longitude -180+r/2 .. 180-r/2, step r. Avoids duplicate antimeridian
samples and matches rasterio/GDAL's pixel-center convention for GeoTIFF
export.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """resolution_deg: cell spacing. lat/lon_centers: 1-D, ascending. lat/lon_flat: row-major flattened meshgrid. shape: (n_lat, n_lon)."""

    resolution_deg: float
    lat_centers: np.ndarray
    lon_centers: np.ndarray
    lat_flat: np.ndarray
    lon_flat: np.ndarray
    shape: tuple[int, int]


def generate_grid(
    resolution_deg: float,
    lat_bounds: tuple[float, float] = (-90.0, 90.0),
    lon_bounds: tuple[float, float] = (-180.0, 180.0),
) -> GridSpec:
    if resolution_deg <= 0:
        raise ValueError(f"resolution_deg must be positive, got {resolution_deg}")

    lat_min, lat_max = lat_bounds
    lon_min, lon_max = lon_bounds

    half = resolution_deg / 2.0
    lat_centers = np.arange(lat_min + half, lat_max, resolution_deg)
    lon_centers = np.arange(lon_min + half, lon_max, resolution_deg)

    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)  # shape (n_lat, n_lon)

    return GridSpec(
        resolution_deg=resolution_deg,
        lat_centers=lat_centers,
        lon_centers=lon_centers,
        lat_flat=lat_grid.ravel(order="C"),
        lon_flat=lon_grid.ravel(order="C"),
        shape=(lat_centers.size, lon_centers.size),
    )
