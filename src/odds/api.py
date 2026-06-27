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
from typing import TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from src.odds.budget import QuotaBudget

_BASE = "https://api.the-odds-api.com/v4"


def _budget_get(
    url: str,
    params: dict,
    timeout: int = 15,
    budget: QuotaBudget | None = None,
    cost: int = 1,
    label: str = "odds-api",
) -> requests.Response | None:
    """
    GET with optional budget checking.
    Returns the response, or None if budget is exhausted or request fails.
    """
    if budget and not budget.can_spend(cost):
        print(f"  [{label}] skipping — budget exhausted ({budget.remaining} remaining)")
        return None

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [{label}] request failed: {e}")
        return None

    if budget:
        budget.record_from_headers(dict(resp.headers))

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    print(f"  [{label}] requests used={used}  remaining={remaining}")

    return resp

SPORT_NCAAB          = "basketball_ncaab"
SPORT_NCAAF          = "americanfootball_ncaaf"
SPORT_NBA            = "basketball_nba"
SPORT_MLB            = "baseball_mlb"
SPORT_EPL            = "soccer_epl"
SPORT_MLS            = "soccer_usa_mls"
SPORT_UFC            = "mma_mixed_martial_arts"
SPORT_NHL            = "icehockey_nhl"
SPORT_NCAAW_VB       = "volleyball_ncaaw"
SPORT_ATP_WIMBLEDON  = "tennis_atp_wimbledon"
SPORT_WTA_WIMBLEDON  = "tennis_wta_wimbledon"
SPORT_ATP_US_OPEN    = "tennis_atp_us_open"
SPORT_WTA_US_OPEN    = "tennis_wta_us_open"


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
    budget: QuotaBudget | None = None,
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
    resp = _budget_get(
        f"{_BASE}/sports/{sport}/odds",
        params={
            "apiKey":      _key(),
            "regions":     "us",
            "markets":     markets,
            "oddsFormat":  "american",
            "bookmakers":  bookmakers,
        },
        budget=budget,
        label=f"odds-api/{sport}",
    )
    if resp is None:
        return []

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


def fetch_historical_odds(
    sport: str = SPORT_NCAAB,
    date_iso: str = "",
    bookmakers: str = "draftkings,fanduel,betmgm",
) -> list[dict]:
    """
    Fetch historical odds at a specific UTC timestamp (paid plan required).

    date_iso: ISO 8601 string, e.g. "2024-01-15T22:00:00Z"

    Returns same format as fetch_odds() — one dict per game per bookmaker.
    The historical endpoint wraps events in a "data" key.
    """
    try:
        resp = requests.get(
            f"{_BASE}/historical/sports/{sport}/odds",
            params={
                "apiKey":      _key(),
                "date":        date_iso,
                "regions":     "us",
                "markets":     "h2h",
                "oddsFormat":  "american",
                "bookmakers":  bookmakers,
            },
            timeout=20,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [hist-api] request failed: {e}")
        return []

    remaining = resp.headers.get("x-requests-remaining", "?")
    used      = resp.headers.get("x-requests-used", "?")
    print(f"  [hist-api] {date_iso}  used={used}  remaining={remaining}")

    results: list[dict] = []
    for event in resp.json().get("data", []):
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                if market["key"] != "h2h":
                    continue
                outcomes = market.get("outcomes", [])
                h = next((o for o in outcomes if o["name"] == home_team), None)
                a = next((o for o in outcomes if o["name"] == away_team), None)
                if not h or not a:
                    continue
                home_ml = int(h["price"])
                away_ml = int(a["price"])
                raw_h, raw_a = american_to_prob(home_ml), american_to_prob(away_ml)
                fair_h, fair_a = remove_vig(raw_h, raw_a)
                results.append({
                    "game_id":   event["id"],
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": bookmaker["key"],
                    "home_ml":   home_ml,
                    "away_ml":   away_ml,
                    "home_prob": round(fair_h, 4),
                    "away_prob": round(fair_a, 4),
                })
    return results


def fetch_events(
    sport: str = SPORT_NCAAB,
    budget: QuotaBudget | None = None,
) -> list[dict]:
    """
    Fetch upcoming events with their Odds API event IDs.
    Costs 1 API request. Use this to get event IDs for fetch_player_props().

    Returns list of {event_id, home_team, away_team, commence_time}.
    """
    resp = _budget_get(
        f"{_BASE}/sports/{sport}/events",
        params={"apiKey": _key()},
        budget=budget,
        label=f"events/{sport}",
    )
    if resp is None:
        return []

    return [
        {
            "event_id":      e["id"],
            "home_team":     e.get("home_team", ""),
            "away_team":     e.get("away_team", ""),
            "commence_time": e.get("commence_time", ""),
        }
        for e in resp.json()
    ]


_PROP_STAT_MAP = {
    "player_points":   "pts",
    "player_rebounds": "reb",
    "player_assists":  "ast",
    "player_steals":   "stl",
    "player_blocks":   "blk",
}


def fetch_player_props(
    sport: str,
    event_id: str,
    markets: str = "player_points,player_rebounds,player_assists,player_steals,player_blocks",
    bookmakers: str = "draftkings,fanduel,betmgm",
    budget: QuotaBudget | None = None,
) -> list[dict]:
    """
    Fetch player prop lines for a single event. Costs 1 API request.

    Returns list of dicts, one per player per stat per bookmaker:
    {
        "event_id":    str,
        "player_name": str,
        "stat":        str,   # 'pts', 'reb', 'ast', 'stl', 'blk'
        "line":        float,
        "over_price":  int,   # American moneyline
        "under_price": int,
        "over_prob":   float, # vig-removed implied prob
        "under_prob":  float,
        "bookmaker":   str,
        "fetched_at":  str,
    }
    """
    resp = _budget_get(
        f"{_BASE}/sports/{sport}/events/{event_id}/odds",
        params={
            "apiKey":     _key(),
            "regions":    "us",
            "markets":    markets,
            "oddsFormat": "american",
            "bookmakers": bookmakers,
        },
        budget=budget,
        label=f"props/{event_id[:8]}",
    )
    if resp is None:
        return []

    fetched_at = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []
    data = resp.json()

    for bookmaker in data.get("bookmakers", []):
        book_key = bookmaker["key"]
        for market in bookmaker.get("markets", []):
            market_key = market["key"]
            stat = _PROP_STAT_MAP.get(market_key)
            if not stat:
                continue

            # Group outcomes by player name
            outcomes = market.get("outcomes", [])
            players: dict[str, dict] = {}
            for o in outcomes:
                player_name = o.get("description", "")
                side        = o.get("name", "")  # "Over" or "Under"
                price       = o.get("price", 0)
                point       = o.get("point", 0.0)
                if not player_name:
                    continue
                if player_name not in players:
                    players[player_name] = {"line": point}
                if side == "Over":
                    players[player_name]["over_price"] = int(price)
                elif side == "Under":
                    players[player_name]["under_price"] = int(price)

            for player_name, p in players.items():
                over_price  = p.get("over_price")
                under_price = p.get("under_price")
                if over_price is None or under_price is None:
                    continue
                raw_o = american_to_prob(over_price)
                raw_u = american_to_prob(under_price)
                fair_o, fair_u = remove_vig(raw_o, raw_u)
                results.append({
                    "event_id":    event_id,
                    "player_name": player_name,
                    "stat":        stat,
                    "line":        p["line"],
                    "over_price":  over_price,
                    "under_price": under_price,
                    "over_prob":   round(fair_o, 4),
                    "under_prob":  round(fair_u, 4),
                    "bookmaker":   book_key,
                    "fetched_at":  fetched_at,
                })

    return results


def fetch_scores(
    sport: str = SPORT_NCAAB,
    days_from: int = 1,
    budget: QuotaBudget | None = None,
) -> list[dict]:
    """
    Fetch recent completed scores. Used to match closing lines to outcomes.

    Returns list of {game_id, home_team, away_team, home_score, away_score, completed}.
    """
    resp = _budget_get(
        f"{_BASE}/sports/{sport}/scores",
        params={
            "apiKey":      _key(),
            "daysFrom":    days_from,
        },
        budget=budget,
        label=f"scores/{sport}",
    )
    if resp is None:
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
