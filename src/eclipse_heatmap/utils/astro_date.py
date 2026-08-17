"""Calendar date with no year-range limit, unlike datetime.date (years 1-9999)."""

from __future__ import annotations

from dataclasses import dataclass

from skyfield.timelib import compute_calendar_date, julian_day


@dataclass(frozen=True, order=True)
class AstroDate:
    year: int
    month: int
    day: int

    @classmethod
    def from_jd(cls, jd_integer: int) -> "AstroDate":
        year, month, day = compute_calendar_date(int(jd_integer))
        return cls(int(year), int(month), int(day))

    @classmethod
    def today_utc(cls) -> "AstroDate":
        from datetime import datetime, timezone

        d = datetime.now(timezone.utc).date()
        return cls(d.year, d.month, d.day)

    @classmethod
    def parse(cls, s: str) -> "AstroDate":
        neg = s.startswith("-")
        body = s[1:] if neg else s
        try:
            year_str, month_str, day_str = body.split("-")
            year, month, day = int(year_str), int(month_str), int(day_str)
        except ValueError:
            raise ValueError(f"Invalid date {s!r}, expected YYYY-MM-DD") from None
        if neg:
            year = -year
        if compute_calendar_date(julian_day(year, month, day)) != (year, month, day):
            raise ValueError(f"Invalid calendar date {s!r}")
        return cls(year, month, day)

    def to_jd(self) -> int:
        return julian_day(self.year, self.month, self.day)

    def add_days(self, days: int) -> "AstroDate":
        return AstroDate.from_jd(self.to_jd() + days)

    def __sub__(self, other: "AstroDate") -> int:
        return self.to_jd() - other.to_jd()

    def isoformat(self) -> str:
        sign = "-" if self.year < 0 else ""
        return f"{sign}{abs(self.year):04d}-{self.month:02d}-{self.day:02d}"

    def __str__(self) -> str:
        return self.isoformat()
