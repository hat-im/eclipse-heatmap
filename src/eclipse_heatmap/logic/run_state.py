"""Cumulative per-point run state, and merging stored checkpoints into it in true chronological order."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..data.checkpoint import EventCheckpoint


class RunState:
    """Cumulative per-point tracking, rebuilt by replaying checkpoints on resume."""

    def __init__(self, n_points: int):
        self.assigned = np.zeros(n_points, dtype=bool)
        self.days_until = np.full(n_points, np.nan, dtype=np.float64)
        self.eclipse_dates = np.full(n_points, None, dtype=object)
        self.eclipse_type = np.zeros(n_points, dtype=np.int8)
        self.eclipse_magnitude = np.zeros(n_points, dtype=np.float64)
        self.eclipse_index = np.full(n_points, np.nan, dtype=np.float64)
        self.processed_dates: list = []

    def apply_event(
        self, today, event_date, magnitude: np.ndarray, eclipse_type: np.ndarray, magnitude_threshold: float = 0.0
    ) -> int:
        """Updates cumulative state for one event; returns count of newly-assigned points.

        magnitude_threshold only gates what counts as "covered" for
        days-until tracking -- the color/opacity blend always uses the
        raw magnitude, unaffected by this threshold.
        """
        self.processed_dates.append(event_date)
        event_index = len(self.processed_dates)

        hit_mask = (~self.assigned) & (magnitude > magnitude_threshold)
        hit_idx = np.where(hit_mask)[0]
        if hit_idx.size:
            self.assigned[hit_idx] = True
            self.days_until[hit_idx] = event_date - today
            self.eclipse_dates[hit_idx] = event_date.isoformat()
            self.eclipse_type[hit_idx] = eclipse_type[hit_idx]
            self.eclipse_magnitude[hit_idx] = magnitude[hit_idx]
            self.eclipse_index[hit_idx] = event_index
        return hit_idx.size


class CheckpointMerger:
    """Applies stored checkpoints to RunState lazily, interleaved by true date order with newly-found events.

    Extending a run backward in time (an earlier --start-date than any
    existing checkpoint) means newly-discovered events can be
    chronologically earlier than already-stored ones. Applying all old
    checkpoints up front (as a plain replay would) then appending new
    ones as they're found gets eclipse_index and the color blend order
    wrong -- both need true global date order, not processing order.
    Re-sorting and replaying everything from scratch after every event
    would fix it but costs O(N^2) over a long run. This merges instead:
    each stored checkpoint is applied exactly once, at the point its date
    is first known to precede whatever's next, so the whole run costs
    O(N) apply_event calls in total, same as before.
    """

    def __init__(self, checkpoints: Iterable[EventCheckpoint]):
        self._iter = iter(checkpoints)
        self._next = next(self._iter, None)

    def apply_up_to(self, state: RunState, today, magnitude_threshold: float, cutoff_date) -> None:
        """Applies every not-yet-applied checkpoint dated strictly before cutoff_date."""
        while self._next is not None and self._next.date < cutoff_date:
            state.apply_event(today, self._next.date, self._next.magnitude, self._next.eclipse_type, magnitude_threshold)
            self._next = next(self._iter, None)

    def apply_rest(self, state: RunState, today, magnitude_threshold: float) -> None:
        while self._next is not None:
            state.apply_event(today, self._next.date, self._next.magnitude, self._next.eclipse_type, magnitude_threshold)
            self._next = next(self._iter, None)
