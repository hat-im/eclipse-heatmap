"""Eclipse type classification."""

from __future__ import annotations


class EclipseType:
    """Eclipse type codes, ordered by significance (TOTAL > ANNULAR > PARTIAL > NONE)."""

    NONE = 0
    PARTIAL = 1
    ANNULAR = 2
    TOTAL = 3

    NAMES = {NONE: "none", PARTIAL: "partial", ANNULAR: "annular", TOTAL: "total"}

    @classmethod
    def name(cls, code: int) -> str:
        return cls.NAMES.get(int(code), "unknown")
