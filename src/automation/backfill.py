"""
Proactive data backfill — never start a season cold again.

Runs on every daily collect (and extra often in pre-season) to:
  1. Backfill full prior seasons for calibration
  2. Detect and fill date gaps in the cache
  3. Report data health
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from src.utils.data import fetch_day, fetch_season

_ROOT = Path(__file__).resolve().parents[2]

# Full season windows for backfill (start, end inclusive)
SEASON_WINDOWS: dict[str, list[tuple[date, date]]] = {
    "mens": [
        (date(2024, 11, 4), date(2025, 4, 7)),
        (date(2025, 11, 4), date(2026, 4, 7)),
    ],
    "cfb": [
        (date(2024, 8, 24), date(2025, 1, 20)),
        (date(2025, 8, 23), date(2026, 1, 19)),
    ],
}


def _cache_dir(division: str) -> Path:
    p = _ROOT / "data" / "raw" / division
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cached_dates(division: str) -> set[date]:
    cache = _cache_dir(division)
    dates: set[date] = set()
    for f in cache.glob("*.json"):
        stem = f.stem
        if len(stem) == 8 and stem.isdigit():
            try:
                dates.add(date(int(stem[:4]), int(stem[4:6]), int(stem[6:8])))
            except ValueError:
                pass
    return dates


def _season_dates(start: date, end: date) -> list[date]:
    """All calendar dates in range (we fetch even if no games — empty file is OK)."""
    out: list[date] = []
    d = start
    while d <= end:
        out.append(d)
        d += timedelta(days=1)
    return out


def backfill_season_windows(division: str, *, verbose: bool = True) -> int:
    """Fetch any missing dates from configured season windows. Returns days filled."""
    cache = _cache_dir(division)
    cached = _cached_dates(division)
    filled = 0

    for start, end in SEASON_WINDOWS.get(division, []):
        end = min(end, date.today())
        if end < start:
            continue
        missing = [d for d in _season_dates(start, end) if d not in cached]
        if not missing and verbose:
            print(f"  [backfill/{division}] {start}→{end}: complete")
            continue
        if verbose:
            print(f"  [backfill/{division}] {len(missing)} missing days in {start}→{end}")
        for d in missing:
            games = fetch_day(d, division=division)
            (cache / f"{d.strftime('%Y%m%d')}.json").write_text(json.dumps(games))
            cached.add(d)
            filled += 1
            if verbose and filled % 25 == 0:
                print(f"    ... {filled} days filled")
    return filled


def fill_recent_gaps(division: str, lookback_days: int = 14) -> int:
    """Re-fetch last N days to catch late-finalized scores."""
    cache = _cache_dir(division)
    filled = 0
    today = date.today()
    for i in range(lookback_days):
        d = today - timedelta(days=i)
        games = fetch_day(d, division=division)
        path = cache / f"{d.strftime('%Y%m%d')}.json"
        path.write_text(json.dumps(games))
        filled += 1
    return filled


def data_health(division: str) -> dict:
    """Return coverage stats for a division."""
    cached = _cached_dates(division)
    windows = SEASON_WINDOWS.get(division, [])
    if not windows:
        return {"division": division, "pct": 0, "missing": 0, "expected": 0}

    expected = 0
    missing = 0
    for start, end in windows:
        end = min(end, date.today())
        if end < start:
            continue
        days = _season_dates(start, end)
        expected += len(days)
        missing += sum(1 for d in days if d not in cached)

    pct = round(100 * (expected - missing) / expected, 1) if expected else 0
    return {
        "division": division,
        "expected_days": expected,
        "cached_days": len(cached),
        "missing_days": missing,
        "coverage_pct": pct,
    }


def run_proactive_backfill(divisions: list[str] | None = None) -> dict[str, dict]:
    """
    Full proactive backfill for all divisions.
    CFB runs year-round (pre-season prep starts July).
    """
    divisions = divisions or ["mens", "cfb"]
    results: dict[str, dict] = {}

    for division in divisions:
        print(f"\n  [backfill] {division}...")
        n = backfill_season_windows(division)
        fill_recent_gaps(division, lookback_days=7)
        health = data_health(division)
        health["days_filled_this_run"] = n
        results[division] = health
        print(
            f"  [backfill/{division}] coverage {health['coverage_pct']}% "
            f"({health['missing_days']} days missing)"
        )

    return results
