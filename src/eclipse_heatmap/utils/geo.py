"""Generic (non-domain) geometric math."""

from __future__ import annotations

import numpy as np


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
