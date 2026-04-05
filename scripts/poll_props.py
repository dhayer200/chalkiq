#!/usr/bin/env python3
"""
Poll The Odds API for player props and compute edge vs our PlayerEloEngine.

Supports multi-sport polling with budget-aware request management.
Each event costs 1 API request for props — use --max-games to limit spend.

Usage:
    python scripts/poll_props.py --divisions mens,nba --once
    python scripts/poll_props.py --divisions mens,nba --interval 60
    python scripts/poll_props.py --divisions mens --hours 6 --max-games 3

Options:
    --divisions     Comma-separated divisions (default: mens)
    --hours N       Only games starting in next N hours (default: 12)
    --interval N    Poll every N minutes (default: 60)
    --max-games N   Max games to fetch props for per division per poll (default: 3)
    --edge N        Minimum |edge| to flag (default: 0.05 = 5%)
    --once          Fetch once and exit

Environment:
    ODDS_API_KEY=your_key  (or .env)
"""

from __future__ import annotations

import argparse
import difflib
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Fix TLS cert resolution when venv is copied across machines.
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.odds.api    import fetch_events, fetch_player_props
from src.odds.budget import QuotaBudget, active_sports_today
from src.odds.store  import save_prop_snapshot, get_prop_history
from src.players.engine import PlayerEloEngine, load_boxscores
from src.slate.generate import ODDS_SPORT

_EDGE_THRESHOLD = 0.05


def build_player_engine(division: str) -> PlayerEloEngine:
    boxscore_dir = f"data/raw/{division}/boxscores"
    boxscore_path = Path(__file__).resolve().parents[1] / boxscore_dir
    if not boxscore_path.exists():
        print(f"  [{division}] no boxscores at {boxscore_dir} — skipping player engine")
        return None

    print(f"  Loading box scores for {division}...")
    games = load_boxscores(str(boxscore_path))
    engine = PlayerEloEngine()
    engine.process_games(games)
    counts = engine.player_game_counts()
    n_qualified = sum(1 for c in counts.values() if c >= 5)
    print(f"  [{division}] {len(games)} games  {len(engine.ratings)} players  "
          f"{n_qualified} with 5+ games")
    return engine


def _normalize(name: str) -> str:
    return name.lower().strip()


def _match_player(odds_name: str, engine: PlayerEloEngine) -> str | None:
    norm = _normalize(odds_name)
    for pid, display in engine.names.items():
        if _normalize(display) == norm:
            return pid
    all_names = {pid: _normalize(n) for pid, n in engine.names.items()}
    matches = difflib.get_close_matches(norm, all_names.values(), n=1, cutoff=0.85)
    if matches:
        for pid, n in all_names.items():
            if n == matches[0]:
                return pid
    return None


def poll_once(
    player_engine: PlayerEloEngine | None,
    sport: str,
    division: str,
    hours_ahead: float,
    edge_threshold: float,
    min_games: int,
    max_games: int,
    budget: QuotaBudget | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=hours_ahead)
    print(f"\n[{now.isoformat()}] fetching events for {division}...")

    events = fetch_events(sport, budget=budget)
    if not events:
        print(f"  [{division}] no events returned")
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

    print(f"  [{division}] {len(upcoming)} games in next {hours_ahead:.0f}h "
          f"(of {len(events)} total)")

    # Limit to max_games (budget optimization)
    if len(upcoming) > max_games:
        print(f"  [{division}] limiting to {max_games} games (budget)")
        upcoming = upcoming[:max_games]

    if player_engine is None:
        print(f"  [{division}] no player engine — fetching props without model edges")

    game_counts = player_engine.player_game_counts() if player_engine else {}
    edges_found = 0

    for event in upcoming:
        if budget and not budget.can_spend():
            print(f"  [{division}] budget exhausted — stopping props")
            break

        event_id = event["event_id"]
        home_team = event["home_team"]
        away_team = event["away_team"]
        print(f"\n  {away_team} @ {home_team}")

        props = fetch_player_props(sport, event_id, budget=budget)
        if not props:
            print("    no props available")
            continue

        # Deduplicate: best book per (player, stat)
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
            pid = _match_player(player_name, player_engine) if player_engine else None

            model_edge = None
            model_mean = None
            p_over = None
            n_games = 0

            if pid and game_counts.get(pid, 0) >= min_games:
                edge_result = player_engine.prop_edge(pid, stat, prop["line"])
                if "error" not in edge_result:
                    model_edge = edge_result["edge"]
                    model_mean = edge_result["mean"]
                    p_over = edge_result["p_over"]
                    n_games = edge_result["n_games"]

            record = {
                **prop,
                "division":   division,
                "home_team":  home_team,
                "away_team":  away_team,
                "player_id":  pid,
                "model_mean": model_mean,
                "p_over":     p_over,
                "model_edge": model_edge,
                "n_games":    n_games,
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

    print(f"\n  [{division}] edges found: {edges_found} (threshold={edge_threshold:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll player props (multi-sport, budget-aware)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--divisions", default="mens",
                        help="Comma-separated divisions: mens,womens,nba,mlb")
    parser.add_argument("--hours",     type=float, default=12.0,
                        help="Only fetch props for games starting within this many hours")
    parser.add_argument("--interval",  type=int, default=60,
                        help="Poll interval in minutes (continuous mode)")
    parser.add_argument("--edge",      type=float, default=_EDGE_THRESHOLD,
                        help="Minimum |edge| to flag (0.05 = 5%%)")
    parser.add_argument("--min-games", type=int, default=5,
                        help="Min games for a player to have a model projection")
    parser.add_argument("--max-games", type=int, default=3,
                        help="Max games to fetch props for per division per poll")
    parser.add_argument("--once",      action="store_true",
                        help="Fetch once and exit")
    args = parser.parse_args()

    requested_divisions = [d.strip() for d in args.divisions.split(",")]
    budget = QuotaBudget()

    # Cache player engines per division
    player_engines: dict[str, PlayerEloEngine | None] = {}

    def get_player_engine(division: str) -> PlayerEloEngine | None:
        if division not in player_engines:
            player_engines[division] = build_player_engine(division)
        return player_engines[division]

    def poll_all():
        active = active_sports_today(requested_divisions)
        if not active:
            print("  No games today for any requested division")
            return

        print(f"  Active sports: {', '.join(active)}")
        print(f"  {budget.summary()}")

        for division in active:
            sport = ODDS_SPORT.get(division)
            if not sport:
                continue
            engine = get_player_engine(division)
            poll_once(
                engine, sport, division,
                hours_ahead=args.hours,
                edge_threshold=args.edge,
                min_games=args.min_games,
                max_games=args.max_games,
                budget=budget,
            )

    if args.once:
        poll_all()
        return

    print(f"Polling {args.divisions} every {args.interval} minutes. Ctrl+C to stop.")
    poll_count = 0
    while True:
        try:
            poll_all()
        except Exception as e:
            print(f"  [error] {e}")

        poll_count += 1
        # Rebuild engines every 4 polls
        if poll_count % 4 == 0:
            print("  Rebuilding player engines...")
            player_engines.clear()

        print(f"  sleeping {args.interval} min...")
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
