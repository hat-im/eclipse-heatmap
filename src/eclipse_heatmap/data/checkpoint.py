"""Per-event checkpoint files, one per eclipse, named by date; the full set exceeds RAM so access is always streamed."""

from __future__ import annotations

from bisect import insort
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..utils.astro_date import AstroDate


@dataclass(frozen=True)
class EventCheckpoint:
    date: AstroDate
    magnitude: np.ndarray  # per-point, float32
    eclipse_type: np.ndarray  # per-point, int8


def _event_path(checkpoint_dir: Path, event_date: AstroDate) -> Path:
    return checkpoint_dir / f"{event_date.isoformat()}.npz"


class CheckpointStore:
    """Date-ordered lazy view over a checkpoint dir: holds only dates in RAM, loads event data per-file on demand."""

    def __init__(self, checkpoint_dir: Path):
        self._dir = checkpoint_dir
        self._dates: list[AstroDate] = []
        if checkpoint_dir.is_dir():
            for path in checkpoint_dir.glob("*.npz"):
                with np.load(path) as data:
                    self._dates.append(AstroDate.parse(str(data["date"])))
            self._dates.sort()

    def __len__(self) -> int:
        return len(self._dates)

    def __iter__(self) -> Iterator[EventCheckpoint]:
        for date in list(self._dates):
            yield self.load(date)

    def load(self, date: AstroDate) -> EventCheckpoint:
        with np.load(_event_path(self._dir, date)) as data:
            return EventCheckpoint(date=date, magnitude=data["magnitude"], eclipse_type=data["eclipse_type"])

    def save(self, date: AstroDate, magnitude: np.ndarray, eclipse_type: np.ndarray) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _event_path(self._dir, date),
            date=date.isoformat(),
            magnitude=magnitude.astype(np.float32),
            eclipse_type=eclipse_type.astype(np.int8),
        )
        insort(self._dates, date)

    @property
    def dates(self) -> list[AstroDate]:
        return self._dates

    @property
    def first_date(self) -> AstroDate:
        return self._dates[0]

    @property
    def last_date(self) -> AstroDate:
        return self._dates[-1]
