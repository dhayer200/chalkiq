"""
Fetch NCAA basketball game results from ESPN's unofficial API.
Supports both men's and women's divisions.

Each game returned as:
    {
        "date":       "YYYY-MM-DD",
        "home_id":    "254",
        "home_name":  "Utah Utes",
        "away_id":    "248",
        "away_name":  "Houston Cougars",
        "home_score": 52,
        "away_score": 66,
        "neutral":    False,
    }

Usage:
    from src.utils.data import fetch_season
    from datetime import date
    games = fetch_season(date(2025, 11, 4), date(2026, 2, 23),
                         cache_dir="data/raw/mens", division="mens")
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests

_ESPN_TMPL = (
    "https://site.api.espn.com/apis/site/v2/sports"
    "/basketball/{sport}/scoreboard"
)

_SPORTS = {
    "mens":   "mens-college-basketball",
    "womens": "womens-college-basketball",
}


def fetch_day(dt: date, division: str = "mens") -> list[dict]:
    """Return all completed games for a single calendar date."""
    sport = _SPORTS.get(division, _SPORTS["mens"])
    url = _ESPN_TMPL.format(sport=sport)
    params = {"dates": dt.strftime("%Y%m%d"), "limit": 300}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()

    games = []
    for event in r.json().get("events", []):
        status = event.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        comp = event["competitions"][0]
        competitors = comp["competitors"]

        try:
            home = next(c for c in competitors if c["homeAway"] == "home")
            away = next(c for c in competitors if c["homeAway"] == "away")
        except StopIteration:
            continue  # malformed event

        # score is a string in the API
        try:
            home_score = int(home["score"])
            away_score = int(away["score"])
        except (ValueError, KeyError):
            continue

        games.append(
            {
                "date": dt.isoformat(),
                "home_id": home["team"]["id"],
                "home_name": home["team"]["displayName"],
                "away_id": away["team"]["id"],
                "away_name": away["team"]["displayName"],
                "home_score": home_score,
                "away_score": away_score,
                "neutral": comp.get("neutralSite", False),
            }
        )
    return games


def fetch_season(
    start: date,
    end: date,
    cache_dir: str | Path | None = None,
    division: str = "mens",
    verbose: bool = True,
) -> list[dict]:
    """
    Fetch all completed games between start and end (inclusive).

    Results are cached per-day as JSON files in cache_dir (if provided),
    so re-runs don't hit the API again.
    """
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    all_games: list[dict] = []
    d = start
    total_days = (end - start).days + 1
    fetched = 0

    while d <= end:
        cache_file = (cache / f"{d.strftime('%Y%m%d')}.json") if cache else None

        if cache_file and cache_file.exists():
            games = json.loads(cache_file.read_text())
        else:
            games = fetch_day(d, division=division)
            if cache_file:
                cache_file.write_text(json.dumps(games))
            time.sleep(0.08)  # be polite to the API

        all_games.extend(games)
        fetched += 1

        if verbose and (fetched % 10 == 0 or d == end):
            pct = fetched / total_days * 100
            print(f"  fetched {fetched}/{total_days} days ({pct:.0f}%)  "
                  f"games so far: {len(all_games)}", flush=True)

        d += timedelta(days=1)

    return all_games
