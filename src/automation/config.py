"""Season windows and division helpers for autonomous pipelines."""

from __future__ import annotations

from datetime import date

from src.automation import ACTIVE_DIVISIONS

# Month ranges when each division has meaningful games (inclusive)
_IN_SEASON = {
    "mens": {10, 11, 12, 1, 2, 3, 4},   # Nov–Apr peak; Oct pre-season
    "cfb":  {8, 9, 10, 11, 12, 1},      # late Aug – bowls
}


def division_in_season(division: str, for_date: date | None = None) -> bool:
    d = for_date or date.today()
    months = _IN_SEASON.get(division)
    if not months:
        return False
    return d.month in months


def divisions_for_collection(for_date: date | None = None) -> list[str]:
    """Divisions that should receive daily game fetches."""
    d = for_date or date.today()
    return [div for div in ACTIVE_DIVISIONS if division_in_season(div, d)]


# Approximate season start dates for countdown messaging
_SEASON_START = {
    "mens": (11, 4),   # Nov 4
    "cfb":  (8, 23),   # late Aug
}


def days_until_season_start(division: str, for_date: date | None = None) -> int | None:
    """Days until next season kickoff. None if already in season."""
    d = for_date or date.today()
    if division_in_season(division, d):
        return None
    month, day = _SEASON_START.get(division, (1, 1))
    yr = d.year if (d.month, d.day) < (month, day) else d.year + 1
    start = date(yr, month, day)
    return (start - d).days


def is_pre_season_prep(for_date: date | None = None) -> bool:
    """July–August: aggressive CFB backfill before kickoff."""
    d = for_date or date.today()
    return d.month in (7, 8)
