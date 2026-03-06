#!/usr/bin/env python3
"""
Poll The Odds API for live moneylines and track CLV + line movement.

Run continuously:
    python scripts/poll_odds.py

Or once:
    python scripts/poll_odds.py --once

Environment:
    ODDS_API_KEY=your_key  (or set in .env)

Options:
    --interval N    Poll every N minutes (default: 30)
    --sport         Sport key (default: basketball_ncaab)
    --once          Fetch once and exit
    --division      ESPN division for engine (default: mens)
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.odds.api      import fetch_odds, fetch_scores, SPORT_NCAAB
from src.odds.clv      import compute_clv, detect_line_moves, clv_summary
from src.odds.match    import build_index, match_game
from src.odds.store    import (
    save_snapshot, save_clv, save_alert,
    get_opening_line, get_line_history, load_clv_records,
)
from src.ratings.elo   import EloEngine
from src.utils.data    import fetch_season
from datetime import date


def build_engine(division: str = "mens") -> EloEngine:
    from src.utils.data import fetch_season
    from datetime import date

    configs = {
        "mens":   ("data/raw/mens",   date(2025, 11, 4)),
        "womens": ("data/raw/womens", date(2025, 11, 4)),
    }
    cache_dir, season_start = configs.get(division, configs["mens"])
    games  = fetch_season(season_start, date.today(), cache_dir=cache_dir, division=division, verbose=False)
    engine = EloEngine(k=24.0, home_advantage=100.0)
    engine.process_games(games)
    print(f"  [engine] loaded {len(engine.history):,} games  {len(engine.ratings)} teams")
    return engine


def poll_once(engine: EloEngine, sport: str, division: str) -> None:
    index = build_index(engine.names)
    now   = datetime.now(timezone.utc).isoformat()

    print(f"\n[{now}] fetching odds...")
    odds = fetch_odds(sport=sport)
    if not odds:
        print("  no odds returned")
        return

    seen_games: set[str] = set()
    for record in odds:
        game_id   = record["game_id"]
        bookmaker = record["bookmaker"]

        # Match to ESPN team IDs
        espn_home, espn_away = match_game(
            record["home_team"], record["away_team"], index
        )
        if not espn_home or not espn_away:
            continue  # couldn't match to our engine

        # Get our model's probability
        model_prob_home = engine.win_prob(espn_home, espn_away, neutral=True)

        # Enrich record with ESPN IDs and model prob
        enriched = {
            **record,
            "espn_home_id":    espn_home,
            "espn_away_id":    espn_away,
            "model_prob_home": round(model_prob_home, 4),
        }
        save_snapshot(enriched)

        # Detect line moves
        history = get_line_history(game_id, bookmaker)
        if len(history) >= 2:
            moves = detect_line_moves(history)
            for move in moves:
                print(f"  [LMA] {move.summary}")
                save_alert({
                    "type":       "line_move",
                    "game_id":    game_id,
                    "home_team":  record["home_team"],
                    "away_team":  record["away_team"],
                    "bookmaker":  bookmaker,
                    "from_ml":    move.from_ml,
                    "to_ml":      move.to_ml,
                    "from_prob":  move.from_prob,
                    "to_prob":    move.to_prob,
                    "move_size":  move.move_size,
                    "sharp":      move.sharp,
                    "detected_at": now,
                })

        # Flag when model disagrees significantly with line
        gap = model_prob_home - record["home_prob"]
        if abs(gap) >= 0.05:
            direction = "HOME" if gap > 0 else "AWAY"
            print(
                f"  [EDGE] {record['home_team']} vs {record['away_team']} "
                f"[{bookmaker}]  model={model_prob_home:.1%}  "
                f"line={record['home_prob']:.1%}  gap={gap:+.1%}  -> {direction}"
            )

        if game_id not in seen_games:
            seen_games.add(game_id)

    print(f"  processed {len(odds)} odds records for {len(seen_games)} games")

    # Check for CLV opportunities (recently completed games)
    compute_pending_clv(engine, sport, index)


def compute_pending_clv(engine: EloEngine, sport: str, index: dict) -> None:
    """Check for completed games and compute CLV for them."""
    scores = fetch_scores(sport=sport, days_from=2)
    completed = [s for s in scores if s["completed"]]

    existing_clv_ids = {r["game_id"] for r in load_clv_records()}

    for s in completed:
        game_id = s["game_id"]
        if game_id in existing_clv_ids:
            continue

        espn_home, espn_away = match_game(s["home_team"], s["away_team"], index)
        if not espn_home or not espn_away:
            continue

        # Get opening and closing lines from our snapshots
        # Use DraftKings as reference book
        for bookmaker in ("draftkings", "fanduel", "betmgm"):
            history = get_line_history(game_id, bookmaker)
            if len(history) < 2:
                continue

            opening = history[0]
            closing = history[-1]
            model_prob = closing.get("model_prob_home", engine.win_prob(espn_home, espn_away))

            home_score = s.get("home_score")
            away_score = s.get("away_score")
            home_won = (home_score > away_score) if (home_score is not None and away_score is not None) else None

            clv = compute_clv(
                model_prob_home=model_prob,
                opening=opening,
                closing=closing,
                game_id=game_id,
                home_team=s["home_team"],
                away_team=s["away_team"],
                bookmaker=bookmaker,
                home_won=home_won,
            )
            print(f"  [CLV] {clv.summary}")
            save_clv({
                "game_id":           game_id,
                "home_team":         s["home_team"],
                "away_team":         s["away_team"],
                "bookmaker":         bookmaker,
                "model_prob_home":   clv.model_prob_home,
                "opening_home_ml":   opening["home_ml"],
                "closing_home_ml":   closing["home_ml"],
                "opening_home_prob": clv.opening_prob_home,
                "closing_home_prob": clv.closing_prob_home,
                "clv_vs_opening":    clv.clv_vs_opening,
                "clv_vs_closing":    clv.clv_vs_closing,
                "home_won":          home_won,
                "recorded_at":       datetime.now(timezone.utc).isoformat(),
            })
            break  # one CLV record per game is enough


def print_clv_report() -> None:
    from src.odds.clv import CLVResult, clv_summary
    from src.odds.store import load_clv_records

    records = load_clv_records()
    if not records:
        print("No CLV records yet.")
        return

    results = [
        CLVResult(
            game_id=r["game_id"],
            home_team=r["home_team"],
            away_team=r["away_team"],
            bookmaker=r["bookmaker"],
            model_prob_home=r["model_prob_home"],
            opening_prob_home=r["opening_home_prob"],
            closing_prob_home=r["closing_home_prob"],
            clv_vs_opening=r["clv_vs_opening"],
            clv_vs_closing=r["clv_vs_closing"],
            home_won=r.get("home_won"),
        )
        for r in records
    ]

    summary = clv_summary(results)
    print(f"\n{'='*50}")
    print(f"  CLV REPORT  ({summary['n_games']} games)")
    print(f"{'='*50}")
    print(f"  Beat closing line : {summary['beat_closing_pct']:.1%}")
    print(f"  Avg CLV           : {summary['avg_clv']:+.2%}")
    print(f"  Avg vs opening    : {summary['avg_clv_vs_open']:+.2%}")
    if summary.get("win_rate") is not None:
        print(f"  Win rate          : {summary['win_rate']:.1%}")
    print(f"{'='*50}\n")

    for r in results[-10:]:
        print(f"  {r.summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll odds and track CLV/LMA")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in minutes")
    parser.add_argument("--sport",    default=SPORT_NCAAB)
    parser.add_argument("--division", default="mens")
    parser.add_argument("--once",     action="store_true", help="Fetch once and exit")
    parser.add_argument("--report",   action="store_true", help="Print CLV report and exit")
    args = parser.parse_args()

    if args.report:
        print_clv_report()
        return

    print("Building Elo engine...")
    engine = build_engine(args.division)

    if args.once:
        poll_once(engine, args.sport, args.division)
        return

    print(f"Polling every {args.interval} minutes. Ctrl+C to stop.")
    while True:
        try:
            poll_once(engine, args.sport, args.division)
        except Exception as e:
            print(f"  [error] {e}")

        # Rebuild engine periodically to pick up new game results
        next_time = datetime.now(timezone.utc)
        print(f"  sleeping {args.interval} min...")
        time.sleep(args.interval * 60)

        # Rebuild engine every hour
        if (datetime.now(timezone.utc) - next_time).seconds >= 3600:
            print("  rebuilding engine...")
            engine = build_engine(args.division)


if __name__ == "__main__":
    main()
