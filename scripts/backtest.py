#!/usr/bin/env python3
"""
Backtest the Elo model against historical game data.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --division womens
    python scripts/backtest.py --min-edge 0.05 --warmup 100
    python scripts/backtest.py --seasons 2023 2024 2025

What it does:
  1. Loads game data for one or more seasons.
  2. Runs the Elo model game-by-game (ratings are NOT pre-loaded — they build
     from scratch each season, exactly as they would have in real time).
  3. Bets $100 flat on every game where model edge > min_edge at -110 juice.
  4. Reports: win rate, P&L, ROI, log loss, Brier score, monthly breakdown,
     and edge bucket analysis.

The key question: does the model consistently beat 52.4% (the -110 breakeven)?
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import Backtester
from src.utils.data      import fetch_season


# Season date ranges — extend as needed
SEASONS: dict[int, tuple[date, date]] = {
    2023: (date(2022, 11, 7), date(2023, 4, 3)),
    2024: (date(2023, 11, 6), date(2024, 4, 8)),
    2025: (date(2024, 11, 4), date(2025, 4, 7)),
    2026: (date(2025, 11, 4), date.today()),
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


def load_season(season: int, division: str) -> list[dict]:
    if season not in SEASONS:
        print(f"Unknown season {season}. Available: {list(SEASONS.keys())}")
        sys.exit(1)

    start, end = SEASONS[season]
    cache_dirs  = CACHE_DIRS.get(division, CACHE_DIRS["mens"])
    cache_dir   = cache_dirs.get(season, f"data/raw/{division}/{season}")

    print(f"  Loading {division} {season}: {start} to {end}...")
    games = fetch_season(start, end, cache_dir=cache_dir, division=division, verbose=False)
    print(f"  {len(games):,} games loaded")
    return games


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Elo model")
    parser.add_argument("--division",   default="mens",   choices=["mens", "womens"])
    parser.add_argument("--seasons",    nargs="+", type=int, default=[2026], help="Season year(s)")
    parser.add_argument("--min-edge",   type=float, default=0.03, help="Min edge to bet (default 0.03 = 3%%)")
    parser.add_argument("--warmup",     type=int, default=50, help="Games to skip before betting")
    parser.add_argument("--stake",      type=float, default=100.0, help="Flat stake per bet")
    parser.add_argument("--decay",      type=float, default=0.0,
                        help="Recency decay half-life in days (default 0 = off). "
                             "Games N days old get K scaled by 0.5^(N/decay). "
                             "Try 60-90 to weight recent form more heavily.")
    parser.add_argument("--combined",   action="store_true", help="Combine all seasons into one run")
    args = parser.parse_args()

    if args.decay > 0:
        print(f"  Recency decay: half-life = {args.decay:.0f} days "
              f"(games {args.decay:.0f}d ago have 50% K weight, {args.decay*2:.0f}d ago have 25%)")

    bt = Backtester(k=24.0, home_advantage=100.0, decay_half_life=args.decay)

    if args.combined and len(args.seasons) > 1:
        # Load and combine all seasons
        all_games: list[dict] = []
        for season in sorted(args.seasons):
            all_games.extend(load_season(season, args.division))

        print(f"\nRunning combined backtest ({len(all_games):,} total games)...")
        results = bt.run(
            all_games,
            min_edge=args.min_edge,
            stake=args.stake,
            warmup_games=args.warmup,
        )
        bt.print_summary(results, stake=args.stake)

        buckets = bt.edge_buckets(results)
        if buckets:
            print("  Edge bucket analysis (does higher edge = higher win rate?):")
            print(f"  {'Bucket':<12} {'N':>5} {'Win%':>7} {'Avg Edge':>10} {'P&L':>10}")
            print(f"  {'-'*50}")
            for bucket, stats in buckets.items():
                print(
                    f"  {bucket:<12} {stats['n']:>5} "
                    f"{stats['win_rate']:>7.1%} "
                    f"{stats['avg_edge']:>10.1%} "
                    f"${stats['pnl']:>9,.0f}"
                )

    else:
        # Run each season independently
        for season in args.seasons:
            games = load_season(season, args.division)
            print(f"\nRunning backtest: {args.division} {season}...")
            results = bt.run(
                games,
                min_edge=args.min_edge,
                stake=args.stake,
                warmup_games=args.warmup,
            )
            print(f"  Season {season}:")
            bt.print_summary(results, stake=args.stake)

            # Edge bucket breakdown
            buckets = bt.edge_buckets(results)
            if buckets:
                print("  Edge bucket analysis:")
                print(f"  {'Bucket':<12} {'N':>5} {'Win%':>7} {'Avg Edge':>10} {'P&L':>10}")
                print(f"  {'-'*50}")
                for bucket, stats in buckets.items():
                    print(
                        f"  {bucket:<12} {stats['n']:>5} "
                        f"{stats['win_rate']:>7.1%} "
                        f"{stats['avg_edge']:>10.1%} "
                        f"${stats['pnl']:>9,.0f}"
                    )


if __name__ == "__main__":
    main()
