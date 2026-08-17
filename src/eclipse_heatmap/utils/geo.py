"""Generic (non-domain) geometric math."""

from __future__ import annotations

import numpy as np

from ..science.constants import EARTH_EQUATORIAL_RADIUS_KM


def haversine_deg(
    lat1_deg: np.ndarray, lon1_deg: np.ndarray, lat2_deg: np.ndarray, lon2_deg: np.ndarray
) -> np.ndarray:
    """Great-circle distance in degrees. Spherical law of cosines is fine here: used only for the day/night pre-filter."""
    lat1 = np.radians(lat1_deg)
    lat2 = np.radians(lat2_deg)
    dlon = np.radians(lon1_deg - lon2_deg)
    cos_d = np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)
    cos_d = np.clip(cos_d, -1.0, 1.0)
    return np.degrees(np.arccos(cos_d))


def to_unit_vectors(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """(lat, lon) in degrees -> (N, 3) unit vectors on the sphere."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def great_circle_km(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Great-circle distance in km between unit vectors (broadcastable)."""
    cos_angle = np.clip(np.sum(v1 * v2, axis=-1), -1.0, 1.0)
    return EARTH_EQUATORIAL_RADIUS_KM * np.arccos(cos_angle)
