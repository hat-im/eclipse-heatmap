"""Per-event checkpoint files: enables exact resume and non-hardcoded blending.

Each processed eclipse stores its own small compressed file (date, per-point
magnitude, per-point type). Writing one is O(1) in the number of events
already processed, so checkpointing an N-event run costs O(N) total, not
O(N^2) -- unlike rewriting one ever-growing state file every event.

Resuming replays the stored per-event data to reconstruct both the
"first eclipse per point" tracking arrays and the color blend, so nothing
about the expensive Skyfield visibility sweep needs to be redone for
already-processed events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EventCheckpoint:
    date: date
    magnitude: np.ndarray  # per-point, float32
    eclipse_type: np.ndarray  # per-point, int8


def _event_path(checkpoint_dir: Path, event_number: int) -> Path:
    return checkpoint_dir / f"event_{event_number:05d}.npz"


def save_event_checkpoint(
    checkpoint_dir: Path,
    event_number: int,
    event_date: date,
    magnitude: np.ndarray,
    eclipse_type: np.ndarray,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _event_path(checkpoint_dir, event_number),
        date=event_date.isoformat(),
        magnitude=magnitude.astype(np.float32),
        eclipse_type=eclipse_type.astype(np.int8),
    )


def load_checkpoints(checkpoint_dir: Path) -> list[EventCheckpoint]:
    """Loads every event checkpoint found, in chronological (filename) order. Empty list if none exist."""
    if not checkpoint_dir.is_dir():
        return []
    checkpoints = []
    for path in sorted(checkpoint_dir.glob("event_*.npz")):
        with np.load(path) as data:
            checkpoints.append(
                EventCheckpoint(
                    date=datetime.strptime(str(data["date"]), "%Y-%m-%d").date(),
                    magnitude=data["magnitude"],
                    eclipse_type=data["eclipse_type"],
                )
            )
    return checkpoints
