"""Per-event checkpoint files: enables exact resume (forward or backward in time) and non-hardcoded blending.

Each processed eclipse stores its own small compressed file, named by the
eclipse's own date (not insertion order). Loading sorts by filename, so
checkpoints always come back in true chronological order no matter what
order they were computed in -- extending a run backward in time (an
earlier --start-date than any existing checkpoint) merges correctly
without renumbering anything already on disk. Writing one is O(1) in the
number of events already processed, so checkpointing an N-event run
costs O(N) total, not O(N^2).

Resuming replays the stored per-event data to reconstruct both the
"first eclipse per point" tracking arrays and the color blend, so nothing
about the expensive Skyfield visibility sweep needs to be redone for
already-processed events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.astro_date import AstroDate


@dataclass(frozen=True)
class EventCheckpoint:
    date: AstroDate
    magnitude: np.ndarray  # per-point, float32
    eclipse_type: np.ndarray  # per-point, int8


def _event_path(checkpoint_dir: Path, event_date: AstroDate) -> Path:
    return checkpoint_dir / f"{event_date.isoformat()}.npz"


def save_event_checkpoint(
    checkpoint_dir: Path,
    event_date: AstroDate,
    magnitude: np.ndarray,
    eclipse_type: np.ndarray,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _event_path(checkpoint_dir, event_date),
        date=event_date.isoformat(),
        magnitude=magnitude.astype(np.float32),
        eclipse_type=eclipse_type.astype(np.int8),
    )


def load_checkpoints(checkpoint_dir: Path) -> list[EventCheckpoint]:
    """Loads every event checkpoint found, sorted by each event's own stored date (the source of truth, not the filename)."""
    if not checkpoint_dir.is_dir():
        return []
    checkpoints = []
    for path in checkpoint_dir.glob("*.npz"):
        with np.load(path) as data:
            checkpoints.append(
                EventCheckpoint(
                    date=AstroDate.parse(str(data["date"])),
                    magnitude=data["magnitude"],
                    eclipse_type=data["eclipse_type"],
                )
            )
    checkpoints.sort(key=lambda cp: cp.date)
    return checkpoints
