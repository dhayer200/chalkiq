"""
Fetch live NCAA basketball games from ESPN's unofficial scoreboard API.

Returns in-progress games with score, clock, and period so the live win
probability model can compute updated win estimates.

Usage:
    from src.live.feed import fetch_live_games
    games = fetch_live_games(division="mens")
    # games = [{"game_id": ..., "home_id": ..., "home_score": 54,
    #            "away_score": 49, "minutes_remaining": 12.3, ...}, ...]
"""

from __future__ import annotations

from datetime import date

import requests

_ESPN_TMPL = (
    "https://site.api.espn.com/apis/site/v2/sports"
    "/basketball/{sport}/scoreboard"
)

_SPORTS = {
    "mens":   "mens-college-basketball",
    "womens": "womens-college-basketball",
}

# Statuses that count as a "live" game we want to show
_LIVE_STATUSES = {"STATUS_IN_PROGRESS", "STATUS_HALFTIME"}

# Statuses for completed and upcoming games
_FINAL_STATUSES     = {"STATUS_FINAL", "STATUS_FINAL_OT", "STATUS_FINAL_OVERTIME"}
_SCHEDULED_STATUSES = {"STATUS_SCHEDULED", "STATUS_PREGAME", "STATUS_DELAYED"}

# College basketball: 2 halves of 20 min, OT periods of 5 min
_HALF_MINUTES = 20.0


def _parse_clock(display_clock: str) -> float:
    """
    Parse ESPN displayClock string (e.g. '12:34' or '0:00') into minutes.
    Returns 0.0 on any parse error.
    """
    try:
        parts = display_clock.strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1]) / 60.0
    except (ValueError, AttributeError):
        pass
    return 0.0


def _minutes_remaining(period: int, clock_mins: float, status_name: str) -> float:
    """
    Total minutes remaining in the game (regulation + future halves).

    period:      1 = 1st half, 2 = 2nd half, 3+ = OT
    clock_mins:  minutes left in the *current* period
    status_name: ESPN status type name
    """
    if status_name == "STATUS_HALFTIME":
        return _HALF_MINUTES   # full second half still to be played

    if period == 1:
        # time in current half  +  full second half
        return clock_mins + _HALF_MINUTES
    elif period == 2:
        return clock_mins
    else:
        # OT — just use time remaining in current OT period
        return clock_mins


def fetch_live_games(
    division: str = "mens",
    for_date: date | None = None,
) -> list[dict]:
    """
    Fetch currently live games.  Returns a list of dicts:

    {
        "game_id":           str,
        "home_id":           str,   # ESPN team ID
        "home_name":         str,
        "away_id":           str,
        "away_name":         str,
        "home_score":        int,
        "away_score":        int,
        "period":            int,   # 1=1st half, 2=2nd half, 3+=OT
        "clock_mins":        float, # minutes left in current period
        "minutes_remaining": float, # total minutes left (incl. future halves)
        "status":            str,   # "in_progress" | "halftime"
        "status_detail":     str,   # e.g. "12:14 - 2nd Half"
        "neutral":           bool,
    }

    Returns an empty list when no games are live or on network error.
    """
    sport = _SPORTS.get(division, _SPORTS["mens"])
    url   = _ESPN_TMPL.format(sport=sport)
    params: dict = {"limit": 300}
    if for_date:
        params["dates"] = for_date.strftime("%Y%m%d")

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    games: list[dict] = []
    for event in resp.json().get("events", []):
        status_obj  = event.get("status", {})
        status_type = status_obj.get("type", {})
        status_name = status_type.get("name", "")

        if status_name not in _LIVE_STATUSES:
            continue

        comp        = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])

        try:
            home = next(c for c in competitors if c["homeAway"] == "home")
            away = next(c for c in competitors if c["homeAway"] == "away")
        except StopIteration:
            continue

        try:
            home_score = int(home.get("score", 0) or 0)
            away_score = int(away.get("score", 0) or 0)
        except (ValueError, TypeError):
            home_score = away_score = 0

        period       = int(status_obj.get("period", 1) or 1)
        display_clk  = status_type.get("shortDetail", "") or status_obj.get("displayClock", "20:00")
        # shortDetail often has format "12:14 - 2nd" — grab just the time part
        clk_part     = display_clk.split(" - ")[0].strip() if " - " in display_clk else display_clk
        clock_mins   = _parse_clock(clk_part)

        mins_left = _minutes_remaining(period, clock_mins, status_name)

        games.append({
            "game_id":           event.get("id", ""),
            "home_id":           home["team"]["id"],
            "home_name":         home["team"].get("displayName", ""),
            "away_id":           away["team"]["id"],
            "away_name":         away["team"].get("displayName", ""),
            "home_score":        home_score,
            "away_score":        away_score,
            "period":            period,
            "clock_mins":        clock_mins,
            "minutes_remaining": mins_left,
            "status":            "halftime" if status_name == "STATUS_HALFTIME" else "in_progress",
            "status_detail":     status_type.get("detail", display_clk),
            "neutral":           comp.get("neutralSite", False),
        })

    return games


def fetch_other_games(
    division: str = "mens",
    for_date: date | None = None,
    status_filter: set[str] | None = None,
) -> list[dict]:
    """
    Fetch completed or scheduled (non-live) games for a given date.

    status_filter: ESPN status type names to include.
        Defaults to all completed + all scheduled statuses.

    Returns a list of dicts:
    {
        "game_id":       str,
        "home_id":       str,
        "home_name":     str,
        "away_id":       str,
        "away_name":     str,
        "home_score":    int,   # 0 for scheduled games
        "away_score":    int,   # 0 for scheduled games
        "status":        str,   # "final" | "scheduled"
        "status_detail": str,   # e.g. "Final" or "7:00 PM ET"
        "neutral":       bool,
        "game_date":     str,   # ISO date prefix e.g. "2026-03-20"
    }
    """
    if status_filter is None:
        status_filter = _FINAL_STATUSES | _SCHEDULED_STATUSES

    sport  = _SPORTS.get(division, _SPORTS["mens"])
    url    = _ESPN_TMPL.format(sport=sport)
    params: dict = {"limit": 300}
    if for_date:
        params["dates"] = for_date.strftime("%Y%m%d")

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    games: list[dict] = []
    for event in resp.json().get("events", []):
        status_obj  = event.get("status", {})
        status_type = status_obj.get("type", {})
        status_name = status_type.get("name", "")

        if status_name not in status_filter:
            continue

        comp        = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])

        try:
            home = next(c for c in competitors if c["homeAway"] == "home")
            away = next(c for c in competitors if c["homeAway"] == "away")
        except StopIteration:
            continue

        try:
            home_score = int(home.get("score", 0) or 0)
            away_score = int(away.get("score", 0) or 0)
        except (ValueError, TypeError):
            home_score = away_score = 0

        is_final = status_name in _FINAL_STATUSES

        games.append({
            "game_id":       event.get("id", ""),
            "home_id":       home["team"]["id"],
            "home_name":     home["team"].get("displayName", ""),
            "away_id":       away["team"]["id"],
            "away_name":     away["team"].get("displayName", ""),
            "home_score":    home_score,
            "away_score":    away_score,
            "status":        "final" if is_final else "scheduled",
            "status_detail": status_type.get("detail", ""),
            "neutral":       comp.get("neutralSite", False),
            "game_date":     event.get("date", "")[:10],
        })

    return games
