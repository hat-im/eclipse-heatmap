"""Raster shaping and NumPy/GeoTIFF export."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..models.grid import GridSpec

logger = logging.getLogger("eclipse_heatmap.rendering")


def to_raster(values_flat: np.ndarray, grid: GridSpec) -> np.ndarray:
    """Flat per-point array -> north-up 2-D raster. lat_flat/lon_flat are south-to-north; image convention wants north-up."""
    raster = values_flat.reshape(grid.shape)
    return np.flipud(raster)


def save_numpy(array2d: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array2d)
    logger.info("Wrote NumPy array: %s", path)


def save_geotiff(array2d: np.ndarray, grid: GridSpec, path: Path) -> None:
    """Skips with a warning if rasterio is not installed (optional dependency)."""
    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        logger.warning("rasterio is not installed; skipping GeoTIFF export.")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    res = grid.resolution_deg
    west = grid.lon_centers.min() - res / 2.0
    north = grid.lat_centers.max() + res / 2.0
    transform = from_origin(west, north, res, res)

    data = np.where(np.isnan(array2d), -9999.0, array2d).astype(np.float32)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
    logger.info("Wrote GeoTIFF: %s", path)
