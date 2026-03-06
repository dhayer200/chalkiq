#!/usr/bin/env python3
"""
Poll The Odds API for player props and compute edge vs our PlayerEloEngine.

Each event costs 1 API request for props. To avoid burning quota, this script
polls only events starting within the next N hours (default: 12).

Usage:
    python scripts/poll_props.py --once           # Fetch once and exit
    python scripts/poll_props.py --interval 60    # Poll every 60 minutes
    python scripts/poll_props.py --hours 6        # Only games starting in 6h
    python scripts/poll_props.py --edge 0.05      # Only show edges >= 5%
    python scripts/poll_props.py --sport basketball_ncaab

Output:
    data/odds/props.jsonl  -- every prop snapshot (line + our edge)
    Console alerts for edges above threshold

Environment:
    ODDS_API_KEY=your_key  (or .env)
"""

from __future__ import annotations

import argparse
import difflib
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.odds.api   import fetch_events, fetch_player_props, SPORT_NCAAB
from src.odds.store import save_prop_snapshot, get_prop_history
from src.players.engine import PlayerEloEngine, load_boxscores

_BOXSCORE_DIR = "data/raw/mens/boxscores"
_EDGE_THRESHOLD = 0.05   # default minimum edge to flag


def build_player_engine() -> PlayerEloEngine:
    print("Loading box scores...")
    games = load_boxscores(_BOXSCORE_DIR)
    engine = PlayerEloEngine()
    engine.process_games(games)
    counts = engine.player_game_counts()
    n_qualified = sum(1 for c in counts.values() if c >= 5)
    print(f"  {len(games)} games  {len(engine.ratings)} players  "
          f"{n_qualified} with 5+ games")
    return engine


def _normalize(name: str) -> str:
    return name.lower().strip()


def _match_player(odds_name: str, engine: PlayerEloEngine) -> str | None:
    """
    Fuzzy-match an Odds API player name to a player_id in our engine.
    Returns player_id or None.
    """
    norm = _normalize(odds_name)

    # Try exact match on display name
    for pid, display in engine.names.items():
        if _normalize(display) == norm:
            return pid

    # Try fuzzy match (difflib)
    all_names = {pid: _normalize(n) for pid, n in engine.names.items()}
    matches = difflib.get_close_matches(norm, all_names.values(), n=1, cutoff=0.85)
    if matches:
        for pid, n in all_names.items():
            if n == matches[0]:
                return pid

    return None


def poll_once(
    player_engine: PlayerEloEngine,
    sport: str,
    hours_ahead: float,
    edge_threshold: float,
    min_games: int,
) -> None:
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)
    print(f"\n[{now.isoformat()}] fetching events...")

    events = fetch_events(sport)
    if not events:
        print("  no events returned")
        return

    # Filter to games starting soon
    upcoming = []
    for e in events:
        try:
            ct = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            if now <= ct <= cutoff:
                upcoming.append(e)
        except Exception:
            continue

    print(f"  {len(upcoming)} games in next {hours_ahead:.0f}h (of {len(events)} total)")

    game_counts = player_engine.player_game_counts()
    edges_found = 0

    for event in upcoming:
        event_id   = event["event_id"]
        home_team  = event["home_team"]
        away_team  = event["away_team"]
        print(f"\n  {away_team} @ {home_team}")

        props = fetch_player_props(sport, event_id)
        if not props:
            print("    no props available")
            continue

        # Deduplicate: best book per (player, stat) — prefer DK > FD > MGM
        book_priority = {"draftkings": 0, "fanduel": 1, "betmgm": 2}
        best: dict[tuple[str, str], dict] = {}
        for p in props:
            key = (p["player_name"], p["stat"])
            existing = best.get(key)
            if existing is None or (
                book_priority.get(p["bookmaker"], 99)
                < book_priority.get(existing["bookmaker"], 99)
            ):
                best[key] = p

        for (player_name, stat), prop in best.items():
            pid = _match_player(player_name, player_engine)

            # Enrich with engine data
            model_edge = None
            model_mean = None
            p_over     = None
            n_games    = 0

            if pid and game_counts.get(pid, 0) >= min_games:
                edge_result = player_engine.prop_edge(pid, stat, prop["line"])
                if "error" not in edge_result:
                    model_edge = edge_result["edge"]
                    model_mean = edge_result["mean"]
                    p_over     = edge_result["p_over"]
                    n_games    = edge_result["n_games"]

            record = {
                **prop,
                "home_team":   home_team,
                "away_team":   away_team,
                "player_id":   pid,
                "model_mean":  model_mean,
                "p_over":      p_over,
                "model_edge":  model_edge,
                "n_games":     n_games,
            }
            save_prop_snapshot(record)

            # Detect line movement
            history = get_prop_history(event_id, player_name, stat, prop["bookmaker"])
            if len(history) >= 2:
                prev_line = history[-2]["line"]
                curr_line = prop["line"]
                if abs(curr_line - prev_line) >= 0.5:
                    direction = "UP" if curr_line > prev_line else "DOWN"
                    print(
                        f"    [LMA] {player_name} {stat} "
                        f"{prev_line} -> {curr_line} ({direction}) "
                        f"[{prop['bookmaker']}]"
                    )

            # Flag edge
            if model_edge is not None and abs(model_edge) >= edge_threshold:
                side = "OVER" if model_edge > 0 else "UNDER"
                print(
                    f"    [EDGE] {player_name} {stat.upper()} {prop['line']} "
                    f"mean={model_mean:.1f}  p_over={p_over:.1%}  "
                    f"edge={model_edge:+.1%}  -> {side}  "
                    f"[{prop['bookmaker']}  {n_games}g]"
                )
                edges_found += 1

    print(f"\n  edges found: {edges_found} (threshold={edge_threshold:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll player props and compute edge vs PlayerEloEngine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sport",     default=SPORT_NCAAB,
                        help="Odds API sport key")
    parser.add_argument("--hours",     type=float, default=12.0,
                        help="Only fetch props for games starting within this many hours")
    parser.add_argument("--interval",  type=int, default=60,
                        help="Poll interval in minutes (continuous mode)")
    parser.add_argument("--edge",      type=float, default=_EDGE_THRESHOLD,
                        help="Minimum |edge| to flag (0.05 = 5%%)")
    parser.add_argument("--min-games", type=int, default=5,
                        help="Min games for a player to have a model projection")
    parser.add_argument("--once",      action="store_true",
                        help="Fetch once and exit")
    args = parser.parse_args()

    player_engine = build_player_engine()

    if args.once:
        poll_once(player_engine, args.sport, args.hours, args.edge, args.min_games)
        return

    print(f"Polling every {args.interval} minutes. Ctrl+C to stop.")
    poll_count = 0
    while True:
        try:
            poll_once(player_engine, args.sport, args.hours, args.edge, args.min_games)
        except Exception as e:
            print(f"  [error] {e}")

        poll_count += 1
        # Rebuild engine every 4 polls to pick up new box scores
        if poll_count % 4 == 0:
            print("  rebuilding player engine...")
            player_engine = build_player_engine()

        print(f"  sleeping {args.interval} min...")
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
