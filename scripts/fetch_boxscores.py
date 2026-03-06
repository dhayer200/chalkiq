#!/usr/bin/env python3
"""
Fetch ESPN box scores for all cached games and save to data/raw/{division}/boxscores/.

Run once to backfill, then run again incrementally — already-fetched games are skipped.

Usage:
    python scripts/fetch_boxscores.py                    # current season, mens
    python scripts/fetch_boxscores.py --seasons 2024 2025 2026
    python scripts/fetch_boxscores.py --division womens
    python scripts/fetch_boxscores.py --limit 50         # test with 50 games

Output:
    data/raw/mens/boxscores/{YYYYMMDD}_{home_id}_{away_id}.json
    Each file contains game metadata + list of player stat dicts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.players.gamescores import parse_player
from src.utils.data import fetch_season

_BASE   = "https://site.api.espn.com/apis/site/v2/sports/basketball"
_DELAY  = 0.4   # seconds between ESPN API calls

SEASONS: dict[int, tuple[date, date]] = {
    2023: (date(2022, 11, 7),  date(2023, 4,  3)),
    2024: (date(2023, 11, 6),  date(2024, 4,  8)),
    2025: (date(2024, 11, 4),  date(2025, 4,  7)),
    2026: (date(2025, 11, 4),  date.today()),
}

CACHE_DIRS: dict[str, dict[int, str]] = {
    "mens": {
        2023: "data/raw/mens/2023",
        2024: "data/raw/mens/2024",
        2025: "data/raw/mens/2025",
        2026: "data/raw/mens",
    },
    "womens": {
        2023: "data/raw/womens/2023",
        2024: "data/raw/womens/2024",
        2025: "data/raw/womens/2025",
        2026: "data/raw/womens",
    },
}

ESPN_SPORT = {
    "mens":   "mens-college-basketball",
    "womens": "womens-college-basketball",
}


def _scoreboard(division: str, date_str: str) -> list[dict]:
    """
    Fetch ESPN scoreboard for a date. Returns list of
    {event_id, home_espn_id, away_espn_id, home_name, away_name}.
    """
    url = f"{_BASE}/{ESPN_SPORT[division]}/scoreboard"
    try:
        resp = requests.get(
            url,
            params={"dates": date_str.replace("-", ""), "limit": 200},
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [scoreboard] {date_str} failed: {e}")
        return []

    results = []
    for event in resp.json().get("events", []):
        comps = event.get("competitions", [{}])
        comp  = comps[0] if comps else {}
        competitors = comp.get("competitors", [])

        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        results.append({
            "event_id":     event["id"],
            "home_espn_id": home.get("team", {}).get("id", ""),
            "away_espn_id": away.get("team", {}).get("id", ""),
            "home_name":    home.get("team", {}).get("displayName", ""),
            "away_name":    away.get("team", {}).get("displayName", ""),
        })
    return results


def _summary(division: str, event_id: str) -> list[dict] | None:
    """
    Fetch ESPN game summary and parse player stats.
    Returns list of player dicts or None on failure.
    """
    url = f"{_BASE}/{ESPN_SPORT[division]}/summary"
    try:
        resp = requests.get(url, params={"event": event_id}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"    [summary] event={event_id} failed: {e}")
        return None

    bs = resp.json().get("boxscore", {})
    player_blocks = bs.get("players", [])
    if not player_blocks:
        return None

    players: list[dict] = []
    for team_block in player_blocks:
        team_id = team_block.get("team", {}).get("id", "")
        for stat_group in team_block.get("statistics", []):
            labels   = stat_group.get("labels", [])
            athletes = stat_group.get("athletes", [])
            for ath in athletes:
                parsed = parse_player(ath, team_id, labels)
                if parsed:
                    players.append(parsed)

    return players if players else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ESPN box scores for cached games",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seasons",  nargs="+", type=int, default=[2026],
                        help="Season years to fetch")
    parser.add_argument("--division", default="mens", choices=["mens", "womens"])
    parser.add_argument("--limit",    type=int, default=0,
                        help="Max new games to fetch (0 = unlimited, useful for testing)")
    parser.add_argument("--delay",    type=float, default=_DELAY,
                        help="Seconds between ESPN API calls")
    args = parser.parse_args()

    out_dir = Path(f"data/raw/{args.division}/boxscores")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all game records for requested seasons
    print("Loading cached game records...")
    all_games: list[dict] = []
    for season in sorted(args.seasons):
        if season not in SEASONS:
            print(f"  Unknown season {season}, skipping")
            continue
        start, end = SEASONS[season]
        cache = CACHE_DIRS.get(args.division, CACHE_DIRS["mens"]).get(
            season, f"data/raw/{args.division}/{season}"
        )
        games = fetch_season(start, end, cache_dir=cache, division=args.division, verbose=False)
        print(f"  {season}: {len(games):,} games")
        all_games.extend(games)

    if not all_games:
        print("No games found.")
        sys.exit(1)

    # Group by date
    from collections import defaultdict
    by_date: dict[str, list[dict]] = defaultdict(list)
    for g in all_games:
        by_date[g["date"]].append(g)

    # Count already-cached
    existing = {f.stem for f in out_dir.glob("*.json")}
    print(f"\nGames total    : {len(all_games):,}")
    print(f"Already cached : {len(existing):,}")

    total_written = 0
    total_failed  = 0

    for game_date in sorted(by_date.keys()):
        day_games = by_date[game_date]
        date_str  = game_date  # YYYY-MM-DD

        # Check which games on this date still need fetching
        needed = [
            g for g in day_games
            if f"{date_str}_{g['home_id']}_{g['away_id']}" not in existing
        ]
        if not needed:
            continue

        print(f"\n{date_str}  ({len(needed)} new games)")

        # Fetch scoreboard once per date to get event IDs
        time.sleep(args.delay)
        scoreboard = _scoreboard(args.division, date_str)
        if not scoreboard:
            print(f"  no scoreboard data, skipping")
            continue

        # Build lookup: (home_espn_id, away_espn_id) → event_id
        sb_lookup = {
            (e["home_espn_id"], e["away_espn_id"]): e["event_id"]
            for e in scoreboard
        }

        for g in needed:
            if args.limit and total_written >= args.limit:
                print(f"\nReached --limit {args.limit}, stopping.")
                return

            event_id = sb_lookup.get((g["home_id"], g["away_id"]))
            if not event_id:
                # Try reverse (scoreboard home/away can differ)
                event_id = sb_lookup.get((g["away_id"], g["home_id"]))
            if not event_id:
                print(f"  no event ID: {g['away_name']} @ {g['home_name']}")
                total_failed += 1
                continue

            time.sleep(args.delay)
            players = _summary(args.division, event_id)
            if players is None:
                print(f"  no box score: {g['away_name']} @ {g['home_name']} (event={event_id})")
                total_failed += 1
                continue

            record = {
                "game_date":  date_str,
                "event_id":   event_id,
                "home_id":    g["home_id"],
                "home_name":  g["home_name"],
                "away_id":    g["away_id"],
                "away_name":  g["away_name"],
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
                "players":    players,
            }

            out_file = out_dir / f"{date_str}_{g['home_id']}_{g['away_id']}.json"
            with open(out_file, "w") as f:
                json.dump(record, f)

            total_written += 1
            n_players = len(players)
            print(f"  {g['away_name']} @ {g['home_name']}  ({n_players} players)  [{total_written}]")

    print(f"\n{'='*50}")
    print(f"  Written : {total_written}")
    print(f"  Failed  : {total_failed}")
    print(f"  Output  : {out_dir}")


if __name__ == "__main__":
    main()
