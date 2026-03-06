"""
The Odds API client — fetches live moneylines for upcoming games.

Free tier: 500 requests/month. Each call to fetch_odds() costs 1 request.
At a 10-minute poll interval that's ~4,320 requests/month — use sparingly.
Recommended: poll every 30 min during the day, hourly overnight.

Setup:
    export ODDS_API_KEY=your_key_here
    # or put it in a .env file and load with python-dotenv

Get a key at: https://the-odds-api.com/
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import requests

_BASE = "https://api.the-odds-api.com/v4"

SPORT_NCAAB = "basketball_ncaab"
SPORT_NBA   = "basketball_nba"
SPORT_MLB   = "baseball_mlb"
SPORT_EPL   = "soccer_epl"
SPORT_MLS   = "soccer_usa_mls"


def _key() -> str:
    k = os.environ.get("ODDS_API_KEY", "")
    if not k:
        raise EnvironmentError(
            "ODDS_API_KEY not set. "
            "Export it or add it to your .env file."
        )
    return k


def american_to_prob(ml: int | float) -> float:
    """American moneyline → raw implied probability (includes vig)."""
    ml = float(ml)
    if ml > 0:
        return 100.0 / (ml + 100.0)
    return abs(ml) / (abs(ml) + 100.0)


def remove_vig(p_home: float, p_away: float) -> tuple[float, float]:
    """Strip bookmaker vig from two raw implied probabilities."""
    total = p_home + p_away
    return p_home / total, p_away / total


def fetch_odds(
    sport: str = SPORT_NCAAB,
    markets: str = "h2h",
    bookmakers: str = "draftkings,fanduel,betmgm,caesars",
) -> list[dict]:
    """
    Fetch current moneyline (h2h) or spread odds for all upcoming games.

    Returns a list of dicts, one per game per bookmaker:
    {
        "game_id":       str,   # Odds API event ID
        "commence_time": str,   # ISO UTC gametime
        "home_team":     str,   # Odds API team name
        "away_team":     str,
        "bookmaker":     str,
        "home_ml":       int,   # American moneyline
        "away_ml":       int,
        "home_prob":     float, # vig-removed implied probability
        "away_prob":     float,
        "fetched_at":    str,   # ISO UTC timestamp of this fetch
    }
    """
    try:
        resp = requests.get(
            f"{_BASE}/sports/{sport}/odds",
            params={
                "apiKey":      _key(),
                "regions":     "us",
                "markets":     markets,
                "oddsFormat":  "american",
                "bookmakers":  bookmakers,
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [odds-api] request failed: {e}")
        return []

    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    print(f"  [odds-api] requests used={used}  remaining={remaining}")

    fetched_at = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []

    for event in resp.json():
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != markets:
                    continue

                outcomes = market.get("outcomes", [])
                h = next((o for o in outcomes if o["name"] == home_team), None)
                a = next((o for o in outcomes if o["name"] == away_team), None)
                if not h or not a:
                    continue

                home_ml = int(h["price"])
                away_ml = int(a["price"])
                raw_h   = american_to_prob(home_ml)
                raw_a   = american_to_prob(away_ml)
                fair_h, fair_a = remove_vig(raw_h, raw_a)

                results.append({
                    "game_id":       event["id"],
                    "commence_time": event.get("commence_time", ""),
                    "home_team":     home_team,
                    "away_team":     away_team,
                    "bookmaker":     bookmaker["key"],
                    "home_ml":       home_ml,
                    "away_ml":       away_ml,
                    "home_prob":     round(fair_h, 4),
                    "away_prob":     round(fair_a, 4),
                    "fetched_at":    fetched_at,
                })

    return results


def fetch_scores(sport: str = SPORT_NCAAB, days_from: int = 1) -> list[dict]:
    """
    Fetch recent completed scores. Used to match closing lines to outcomes.

    Returns list of {game_id, home_team, away_team, home_score, away_score, completed}.
    """
    try:
        resp = requests.get(
            f"{_BASE}/sports/{sport}/scores",
            params={
                "apiKey":      _key(),
                "daysFrom":    days_from,
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [odds-api] scores request failed: {e}")
        return []

    results = []
    for event in resp.json():
        scores = event.get("scores") or []
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")

        h_score = next((int(s["score"]) for s in scores if s["name"] == home_team), None)
        a_score = next((int(s["score"]) for s in scores if s["name"] == away_team), None)

        results.append({
            "game_id":    event["id"],
            "home_team":  home_team,
            "away_team":  away_team,
            "home_score": h_score,
            "away_score": a_score,
            "completed":  event.get("completed", False),
        })

    return results
