#!/usr/bin/env python3
"""
bracket.py — Tournament bracket with Elo-powered predictions.

Usage:
    python scripts/bracket.py --sport cbb --gender men
    python scripts/bracket.py --sport cbb --gender women
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.ratings.elo import EloEngine, SPORT_CONFIGS
from src.utils.data  import fetch_season

BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


def load_engine(division: str) -> EloEngine:
    ROOT = Path(__file__).resolve().parents[1]
    today = date.today()

    if division == "mlb":
        yr = today.year if today.month >= 3 else today.year - 1
        season_start = date(yr - 1, 3, 20)
    elif division == "nba":
        yr = today.year if today.month >= 10 else today.year - 1
        season_start = date(yr, 10, 15)
    else:
        season_start = date(2025, 11, 4)

    cache_dir = ROOT / "data" / "raw" / division
    games = fetch_season(season_start, today,
                         cache_dir=str(cache_dir), division=division, verbose=False)

    cfg = SPORT_CONFIGS.get(division, SPORT_CONFIGS["mens"])
    eng = EloEngine(
        k=cfg["k"],
        home_advantage=cfg["home_advantage"],
        scale=cfg.get("scale", 300.0),
        season_regress=cfg["season_regress"],
        tempo_adjust=cfg["tempo_adjust"],
        season_boundary=cfg.get("season_boundary", "ncaa"),
    )
    eng.process_games(games)
    return eng


def run(sport: str, gender: str) -> None:
    if sport == "cbb":
        division = "mens" if gender == "men" else "womens"
    else:
        division = sport

    print()
    print(f"{CYAN}{BOLD}  ChalkIQ Bracket — {sport.upper()} {gender.upper()}{RESET}")
    print(f"  {'━' * 60}")
    print()

    engine = load_engine(division)
    top = engine.rankings()[:68 if sport == "cbb" else 30]

    print(f"  {BOLD}{'Seed':>4}  {'Team':<40} {'Elo':>6}{RESET}")
    print(f"  {'─' * 54}")

    for i, (tid, name, elo) in enumerate(top, 1):
        elo_color = GREEN if elo >= 1700 else YELLOW if elo >= 1600 else RESET
        print(f"  {i:>4}  {name:<40} {elo_color}{elo:>6.0f}{RESET}")

        # Bracket round separators
        if sport == "cbb" and i in (16, 32, 48, 64):
            print(f"  {'─' * 54}")

    # Simulate bracket if we have 64+ teams
    if sport == "cbb" and len(top) >= 64:
        print()
        print(f"  {BOLD}Elo-Simulated Final Four:{RESET}")
        print()
        # Simple: top 4 seeds by Elo
        for i, (tid, name, elo) in enumerate(top[:4], 1):
            print(f"    {GREEN}#{i}{RESET}  {BOLD}{name}{RESET}  ({elo:.0f})")

        # Championship prediction
        t1_id, t1_name, t1_elo = top[0]
        t2_id, t2_name, t2_elo = top[1]
        p = engine.win_prob(t1_id, t2_id, neutral=True)
        winner = t1_name if p >= 0.5 else t2_name
        wp = p if p >= 0.5 else 1 - p
        print()
        print(f"  {BOLD}Championship:{RESET} {t1_name} vs {t2_name}")
        print(f"  {BOLD}Predicted winner:{RESET} {GREEN}{winner}{RESET} ({wp:.0%})")

    print()
    print(f"  {'━' * 60}")
    print(f"  {DIM}Rankings as of {date.today().isoformat()}{RESET}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tournament bracket with Elo predictions")
    parser.add_argument("--sport", default="cbb", choices=["cbb", "mlb", "nba"])
    parser.add_argument("--gender", default="men", choices=["men", "women"])
    args = parser.parse_args()
    run(sport=args.sport, gender=args.gender)
