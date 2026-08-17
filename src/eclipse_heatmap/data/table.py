"""CSV table export."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..models.eclipse_type import EclipseType
from ..models.grid import GridSpec

logger = logging.getLogger("eclipse_heatmap.data")


def build_results_dataframe(
    grid: GridSpec,
    days_until: np.ndarray,
    eclipse_dates: np.ndarray,
    eclipse_type: np.ndarray,
    magnitude: np.ndarray,
    eclipse_index: np.ndarray | None = None,
) -> pd.DataFrame:
    """eclipse_index: 1-based position, in processing order, of the event that first covered each point."""
    type_names = np.array([EclipseType.name(c) for c in eclipse_type], dtype=object)
    data = {
        "latitude": grid.lat_flat,
        "longitude": grid.lon_flat,
        "days_until_next_eclipse": days_until,
        "eclipse_date": eclipse_dates,
        "eclipse_type": type_names,
        "eclipse_magnitude": magnitude,
    }
    if eclipse_index is not None:
        data["eclipse_index"] = eclipse_index
    return pd.DataFrame(data)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.5f")
    logger.info("Wrote CSV table (%d rows): %s", len(df), path)
