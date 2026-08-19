"""Streamed Porter-Duff "over" compositing: color = caller-supplied ramp position, alpha = magnitude."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _composite_over(accum_rgba: np.ndarray, layer_rgb: np.ndarray, layer_alpha: np.ndarray) -> None:
    accum_alpha = accum_rgba[:, 3]
    new_alpha = layer_alpha + accum_alpha * (1.0 - layer_alpha)
    denom = np.where(new_alpha > 1e-12, new_alpha, 1.0)
    new_rgb = (
        layer_rgb[None, :] * layer_alpha[:, None] + accum_rgba[:, :3] * accum_alpha[:, None] * (1.0 - layer_alpha[:, None])
    ) / denom[:, None]
    accum_rgba[:, :3] = new_rgb
    accum_rgba[:, 3] = new_alpha


def blend_events(layers: Iterable[tuple[float, np.ndarray]], n_points: int, cmap) -> np.ndarray:
    """Composites (ramp position, magnitude) layers, in chronological order, into one (n_points, 4) RGBA array."""
    accum_rgba = np.zeros((n_points, 4), dtype=np.float64)
    for t, magnitude in layers:
        layer_rgb = np.array(cmap(t)[:3])
        layer_alpha = np.clip(magnitude, 0.0, 1.0)
        _composite_over(accum_rgba, layer_rgb, layer_alpha)
    return np.clip(accum_rgba, 0.0, 1.0)
