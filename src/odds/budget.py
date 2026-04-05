"""
Odds API budget manager — tracks quota usage and determines which sports to poll.

20k requests/month plan. This module tracks usage across multiple sports.

Usage:
    from src.odds.budget import QuotaBudget, active_sports_today

    budget = QuotaBudget()
    sports = active_sports_today(["mens", "nba", "mlb", "epl", "mls", "ufc"])
    if budget.can_spend():
        # make API call ...
        budget.record_usage(remaining=19500, used=500)  # from response headers
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

_DEFAULT_QUOTA_PATH = Path(__file__).resolve().parents[2] / "data" / "odds" / "quota.json"

# Monthly request limit (20k plan)
FREE_TIER_LIMIT = 20_000
SAFETY_MARGIN = 100  # always keep this many in reserve


def active_sports_today(
    divisions: list[str] | None = None,
    for_date: date | None = None,
) -> list[str]:
    """
    Return divisions that have at least one game today.
    Uses ESPN scoreboard (free, no Odds API cost).
    """
    from src.live.feed import fetch_other_games

    divisions = divisions or ["mens", "womens", "nba", "mlb", "epl", "mls", "ufc"]
    target = for_date or date.today()
    active: list[str] = []

    for div in divisions:
        try:
            games = fetch_other_games(division=div, for_date=target)
            if games:
                active.append(div)
        except Exception as e:
            print(f"  [budget] could not check {div} schedule: {e}")

    return active


class QuotaBudget:
    """
    Tracks Odds API quota usage via the x-requests-remaining / x-requests-used
    headers returned by every API response.

    Persists state to disk so daemons share a single source of truth.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else _DEFAULT_QUOTA_PATH
        self._state = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"used": 0, "remaining": FREE_TIER_LIMIT, "last_updated": ""}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2) + "\n")

    @property
    def remaining(self) -> int:
        return self._state.get("remaining", FREE_TIER_LIMIT)

    @property
    def used(self) -> int:
        return self._state.get("used", 0)

    def can_spend(self, n: int = 1) -> bool:
        """Return True if we have enough budget for n requests."""
        return self.remaining >= (n + SAFETY_MARGIN)

    def record_usage(self, remaining: int, used: int) -> None:
        """
        Update quota from Odds API response headers.
        The API's x-requests-used is the authoritative month-to-date count.
        """
        self._state["remaining"] = remaining
        self._state["used"] = used
        self._state["last_updated"] = date.today().isoformat()
        self._save()

    def record_from_headers(self, headers: dict) -> None:
        """Extract quota info from response headers and record it."""
        remaining = headers.get("x-requests-remaining")
        used = headers.get("x-requests-used")
        if remaining is not None and used is not None:
            try:
                self.record_usage(int(remaining), int(used))
            except (ValueError, TypeError):
                pass

    def summary(self) -> str:
        return (
            f"Odds API quota: {self.used} used / {self.remaining} remaining "
            f"(updated {self._state.get('last_updated', 'never')})"
        )
