"""Eclipse geometry: angular radii, separation classification, magnitude. No ephemeris calls.

Classification, given sun_radius, moon_radius, separation (topocentric, degrees):
  separation >= sun_radius + moon_radius        -> no eclipse
  separation <= |moon_radius - sun_radius|      -> central: total (moon larger) or annular (moon smaller)
  otherwise                                     -> partial

Magnitude (NASA GSFC / Meeus ch. 54 definition), fraction of solar diameter covered:
  magnitude = (sun_radius + moon_radius - separation) / (2 * sun_radius)
"""

from __future__ import annotations

import numpy as np

from ..models.eclipse_type import EclipseType
from .constants import EARTH_EQUATORIAL_RADIUS_KM, MOON_RADIUS_KM, SUN_RADIUS_KM

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is an optional dependency
    _NUMBA_AVAILABLE = False


def angular_radius_deg(physical_radius_km: np.ndarray | float, distance_km: np.ndarray) -> np.ndarray:
    """Apparent angular radius, degrees: sin(r) = physical_radius / distance. Exact, no small-angle approximation."""
    return np.degrees(np.arcsin(np.clip(physical_radius_km / distance_km, -1.0, 1.0)))


def sun_angular_radius_deg(distance_km: np.ndarray) -> np.ndarray:
    return angular_radius_deg(SUN_RADIUS_KM, distance_km)


def moon_angular_radius_deg(distance_km: np.ndarray) -> np.ndarray:
    return angular_radius_deg(MOON_RADIUS_KM, distance_km)


def moon_horizontal_parallax_deg(distance_km: np.ndarray) -> np.ndarray:
    """Angle subtended by Earth's equatorial radius as seen from the Moon.

    Bounds the maximum shift between geocentric and topocentric Moon
    position. Used in eclipses.py as a fast necessary-condition filter:
    if geocentric separation exceeds sun_radius + moon_radius + parallax,
    no eclipse is possible anywhere on Earth.
    """
    return angular_radius_deg(EARTH_EQUATORIAL_RADIUS_KM, distance_km)


def _classify_numpy(
    separation_deg: np.ndarray, sun_radius_deg: np.ndarray, moon_radius_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    total_radius = sun_radius_deg + moon_radius_deg
    diff_radius = np.abs(moon_radius_deg - sun_radius_deg)

    magnitude = (total_radius - separation_deg) / (2.0 * sun_radius_deg)
    magnitude = np.clip(magnitude, 0.0, None)

    type_code = np.zeros(np.broadcast(separation_deg, sun_radius_deg, moon_radius_deg).shape, dtype=np.int8)

    partial_mask = separation_deg < total_radius
    type_code = np.where(partial_mask, EclipseType.PARTIAL, type_code)

    central_mask = separation_deg <= diff_radius
    total_mask = central_mask & (moon_radius_deg >= sun_radius_deg)
    annular_mask = central_mask & (moon_radius_deg < sun_radius_deg)
    type_code = np.where(total_mask, EclipseType.TOTAL, type_code)
    type_code = np.where(annular_mask, EclipseType.ANNULAR, type_code)

    return magnitude, type_code.astype(np.int8)


if _NUMBA_AVAILABLE:  # pragma: no branch - trivial branch, exercised by import outcome
    from numba import prange

    @njit(cache=True, parallel=True)
    def _classify_numba(separation_deg, sun_radius_deg, moon_radius_deg):
        n = separation_deg.shape[0]
        magnitude = np.zeros(n, dtype=np.float64)
        type_code = np.zeros(n, dtype=np.int8)
        for i in prange(n):  # noqa: B007 - prange is numba's parallel range
            sep = separation_deg[i]
            rs = sun_radius_deg[i]
            rm = moon_radius_deg[i]
            total_radius = rs + rm
            diff_radius = abs(rm - rs)
            mag = (total_radius - sep) / (2.0 * rs)
            if mag < 0.0:
                mag = 0.0
            magnitude[i] = mag
            if sep < total_radius:
                if sep <= diff_radius:
                    type_code[i] = 3 if rm >= rs else 2
                else:
                    type_code[i] = 1
            else:
                type_code[i] = 0
        return magnitude, type_code


def classify_eclipse(
    separation_deg: np.ndarray, sun_radius_deg: np.ndarray, moon_radius_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(magnitude, EclipseType code) for each input. Scalars or broadcastable arrays.

    Numba path used for 1-D arrays with >=10,000 elements; NumPy otherwise.
    """
    separation_deg = np.asarray(separation_deg, dtype=np.float64)
    sun_radius_deg = np.broadcast_to(np.asarray(sun_radius_deg, dtype=np.float64), separation_deg.shape)
    moon_radius_deg = np.broadcast_to(np.asarray(moon_radius_deg, dtype=np.float64), separation_deg.shape)

    if _NUMBA_AVAILABLE and separation_deg.ndim == 1 and separation_deg.size >= 10_000:
        return _classify_numba(
            np.ascontiguousarray(separation_deg),
            np.ascontiguousarray(sun_radius_deg),
            np.ascontiguousarray(moon_radius_deg),
        )
    return _classify_numpy(separation_deg, sun_radius_deg, moon_radius_deg)
