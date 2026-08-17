"""Single source of truth for the checkpoint-data analysis plots (not the live pipeline's main map)."""

from __future__ import annotations

from . import count_distribution, frequency, interval_heatmap, lat_lon_distribution, path_length

ANALYSIS_PLOTS = [
    frequency,
    lat_lon_distribution,
    path_length,
    count_distribution,
    interval_heatmap,
]
