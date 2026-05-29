"""
ESPN tennis data fetcher for Grand Slam tournaments.

Fetches completed match results and current draw for Wimbledon and US Open.
Results are returned in a format compatible with TennisEloEngine.process_matches().

Usage:
    from src.live.tennis_feed import fetch_tournament_matches, TOURNAMENTS

    matches = fetch_tournament_matches("wimbledon-atp", year=2026)
    # [{"winner": "...", "loser": "...", "surface": "grass", "round": "QF", "grand_slam": True}, ...]
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path
import json

import requests

_ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/tennis"

# tournament_key -> (espn_league, surface)
TOURNAMENTS: dict[str, tuple[str, str]] = {
    "wimbledon-atp": ("atp-wimbledon",  "grass"),
    "wimbledon-wta": ("wta-wimbledon",  "grass"),
    "us-open-atp":   ("atp-us-open",    "hard"),
    "us-open-wta":   ("wta-us-open",    "hard"),
    "french-atp":    ("atp-french-open","clay"),
    "french-wta":    ("wta-french-open","clay"),
    "ao-atp":        ("atp-aus-open",   "hard"),
    "ao-wta":        ("wta-aus-open",   "hard"),
}

_FINAL_STATUSES = {"STATUS_FINAL", "STATUS_FINAL_OT"}

_ROUND_NAMES = {
    "1": "R128", "2": "R64", "3": "R32", "4": "R16",
    "5": "QF", "6": "SF", "7": "F",
}


def _scoreboard_url(league: str) -> str:
    return f"{_ESPN_BASE}/{league}/scoreboard"


def fetch_tournament_matches(
    tournament: str,
    year: int | None = None,
    cache_dir: str | Path | None = None,
) -> list[dict]:
    """
    Fetch all completed matches for a Grand Slam tournament.

    tournament: key from TOURNAMENTS dict (e.g. "wimbledon-atp")
    year: season year; defaults to current year
    cache_dir: if provided, cache raw JSON per date

    Returns list of:
        {winner, loser, surface, round, tournament, grand_slam: True}
    """
    if tournament not in TOURNAMENTS:
        raise ValueError(f"Unknown tournament '{tournament}'. Options: {list(TOURNAMENTS)}")

    league, surface = TOURNAMENTS[tournament]
    target_year = year or date.today().year
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    url = _scoreboard_url(league)

    all_matches: list[dict] = []
    page = 1

    while True:
        params: dict = {"limit": 100, "page": page}
        cache_file = (cache / f"{tournament}_{target_year}_p{page}.json") if cache else None

        if cache_file and cache_file.exists():
            data = json.loads(cache_file.read_text())
        else:
            try:
                r = requests.get(url, params=params, timeout=15)
                r.raise_for_status()
                data = r.json()
                if cache_file:
                    cache_file.write_text(json.dumps(data))
                time.sleep(0.1)
            except requests.RequestException:
                break

        events = data.get("events", [])
        if not events:
            break

        for event in events:
            status = event.get("status", {}).get("type", {}).get("name", "")
            if status not in _FINAL_STATUSES:
                continue

            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            try:
                winner = next(c for c in competitors if c.get("winner"))
                loser  = next(c for c in competitors if not c.get("winner"))
            except StopIteration:
                continue

            round_num = str(comp.get("bracketRound", ""))
            round_label = _ROUND_NAMES.get(round_num, round_num or "?")

            all_matches.append({
                "winner":      winner["athlete"]["displayName"],
                "loser":       loser["athlete"]["displayName"],
                "surface":     surface,
                "round":       round_label,
                "tournament":  tournament,
                "grand_slam":  True,
            })

        # stop paging if we got fewer than a full page
        if len(events) < 100:
            break
        page += 1

    return all_matches


def fetch_current_draw(tournament: str) -> list[dict]:
    """
    Fetch the current draw for a tournament in progress.

    Returns list of players with their seed and section:
        [{name, seed, section}]
    """
    if tournament not in TOURNAMENTS:
        raise ValueError(f"Unknown tournament '{tournament}'")

    league, _ = TOURNAMENTS[tournament]
    url = f"{_ESPN_BASE}/{league}/bracket"

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException:
        return []

    players = []
    for bracket in data.get("bracket", []):
        for entry in bracket.get("competitors", []):
            athlete = entry.get("athlete", {})
            players.append({
                "name":    athlete.get("displayName", "Unknown"),
                "seed":    entry.get("seed"),
                "section": bracket.get("id"),
            })

    return players
