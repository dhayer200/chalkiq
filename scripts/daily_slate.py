#!/usr/bin/env python3
"""
daily_slate.py — Bettor's daily overview with edge analysis.

Supports NCAA basketball, NBA, and MLB. Uses shared generate_slate() for data
and formats output with ANSI colors for the terminal.

Usage:
    python scripts/daily_slate.py                  # mens D1
    python scripts/daily_slate.py --division nba
    python scripts/daily_slate.py --division mlb
    python scripts/daily_slate.py --min-edge 3     # only show games with ≥3% edge
    python scripts/daily_slate.py --top 10         # show only top 10 by edge
    python scripts/daily_slate.py --no-fetch       # use cached snapshots only (saves API calls)
    python scripts/daily_slate.py --yesterday      # yesterday's results
    python scripts/daily_slate.py --week           # next 7 days
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.signals.injuries import _OUT_STATUSES
from src.slate.generate import generate_slate, load_engine, DIVISION_LABELS

ET = ZoneInfo("America/New_York")

# ── Formatting helpers ────────────────────────────────────────────────────── #

def fmt_ml(ml: int) -> str:
    return f"+{ml}" if ml > 0 else str(ml)

def fmt_prob(p: float) -> str:
    return f"{p:.1%}"

def fmt_edge(e: float) -> str:
    sign = "+" if e >= 0 else ""
    return f"{sign}{e:.1%}"

def edge_flag(e: float) -> str:
    if   e >= 0.08: return "🔥 STRONG"
    elif e >= 0.05: return "✅ EDGE"
    elif e >= 0.03: return "⚡ LEAN"
    else:           return ""

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
YELLOW= "\033[93m"
DIM   = "\033[2m"
RESET = "\033[0m"

def color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


# ── Main display ─────────────────────────────────────────────────────────── #

def run(division: str, min_edge: float, top_n: int | None, no_fetch: bool,
        target_dates: list[date] | None = None) -> None:
    today = date.today()
    now_et = datetime.now(ET)
    dates = target_dates or [today]
    is_single_day = len(dates) == 1

    print()
    print(color("━" * 72, CYAN))
    if is_single_day:
        d = dates[0]
        label = "Today" if d == today else ("Tomorrow" if d == today + timedelta(days=1)
                else "Yesterday" if d == today - timedelta(days=1) else d.strftime("%A, %B %d %Y"))
        print(color(f"  ChalkIQ Slate  —  {label}  ({division.upper()})", BOLD))
    else:
        print(color(f"  ChalkIQ Slate  —  {dates[0].strftime('%b %d')} → {dates[-1].strftime('%b %d %Y')}  ({division.upper()})", BOLD))
    print(color("━" * 72, CYAN))
    print()

    # Build engine once for all dates
    print(color("  Loading Elo ratings...", DIM), end="\r")
    engine = load_engine(division)
    print(f"  Elo engine: {len(engine.ratings)} teams rated" + " " * 30)

    for d in dates:
        if len(dates) > 1:
            print(color(f"\n  ── {d.strftime('%A, %B %d')} ──", BOLD))

        result = generate_slate(
            division,
            engine=engine,
            min_edge=min_edge,
            top_n=top_n,
            no_fetch=no_fetch,
            target_date=d,
        )

        print(f"  {result.n_upcoming} upcoming  |  {result.n_live} live  |  {result.n_final} final")

        # Print completed game results
        if result.final_games:
            print()
            print(color(f"  RESULTS  ({result.n_final} games)", BOLD))
            print()
            for g in result.final_games:
                h_score = g["home_score"]
                a_score = g["away_score"]
                h_win = h_score > a_score
                correct = g.get("model_correct", False)
                model_h = g.get("model_home", 0.5)
                predicted_home = model_h >= 0.5

                if h_win:
                    h_str = color(f"{g['home_name']} {h_score}", GREEN + BOLD)
                    a_str = f"{g['away_name']} {a_score}"
                else:
                    h_str = f"{g['home_name']} {h_score}"
                    a_str = color(f"{g['away_name']} {a_score}", GREEN + BOLD)

                check = color("V", GREEN) if correct else color("X", RED)
                prob_str = f"{model_h:.0%}" if predicted_home else f"{1-model_h:.0%}"
                fav_name = g['home_name'] if predicted_home else g['away_name']
                print(f"    {check}  {a_str}  @  {h_str}"
                      f"    {color(f'(model: {fav_name} {prob_str})', DIM)}")

            if result.model_accuracy is not None:
                pct = result.model_accuracy * 100
                acc_color = GREEN if pct >= 60 else YELLOW if pct >= 50 else RED
                n_correct = sum(1 for g in result.final_games if g.get("model_correct", False))
                print(f"\n  Model accuracy: {color(f'{n_correct}/{result.n_final} ({pct:.0f}%)', acc_color)}")
            print()

        # Print upcoming game cards
        game_cards = result.game_cards
        bet_count = 0

        if game_cards:
            print(color(f"  UPCOMING  ({len(game_cards)} games)", BOLD))
            print()

        for card in game_cards:
            edge_h = card["edge_home"]
            edge_a = card["edge_away"]
            bet = card["bet_side"]
            status = card["status"]

            status_tag = color(" [LIVE] ", RED + BOLD) if status == "in" else ""
            matchup = f"{card['away_name']}  @  {card['home_name']}"
            print(color("─" * 72, DIM))
            print(f"  {color(matchup, BOLD)}{status_tag}   {color(card['tip_detail'], DIM)}")

            home_r = f"#{card['home_rank']} {card['home_name']} ({card['home_elo']:.0f})"
            away_r = f"#{card['away_rank']} {card['away_name']} ({card['away_elo']:.0f})"
            print(f"  {color('Elo:', DIM)} {away_r}  vs  {home_r}")

            print(f"  {color('Model:', DIM)}  {card['away_name']}: {fmt_prob(card['model_away'])}   "
                  f"{card['home_name']}: {fmt_prob(card['model_home'])}")

            if card["current_lines"]:
                print(f"  {color('Lines:', DIM)}", end="")
                books_shown = list(card["current_lines"].items())[:4]
                parts = []
                for book, ln in books_shown:
                    parts.append(f"  {book[:2].upper()}: {fmt_ml(ln['away_ml'])} / {fmt_ml(ln['home_ml'])}")
                print("".join(parts))

                if card["open_home_ml"] is not None:
                    open_tag = f"open {fmt_ml(card['open_home_ml'])} → now {fmt_ml(card['best_home_ml'])}" \
                               if card["best_home_ml"] is not None else f"open {fmt_ml(card['open_home_ml'])}"
                    move_tag = f"  {color(card['line_move'], YELLOW)}" if card["line_move"] else ""
                    print(f"  {color('Open→Close:', DIM)} {open_tag}{move_tag}")
                elif card["best_home_ml"] is not None:
                    print(f"  {color('Best line:', DIM)}  "
                          f"{card['away_name']}: {fmt_ml(card['best_away_ml'])}   "
                          f"{card['home_name']}: {fmt_ml(card['best_home_ml'])}")
            else:
                print(f"  {color('Lines:', DIM)}  no odds data (run poll_odds.py or omit --no-fetch)")

            if edge_h is not None:
                eh_str = color(fmt_edge(edge_h), GREEN if edge_h >= min_edge else RESET)
                ea_str = color(fmt_edge(edge_a), GREEN if edge_a >= min_edge else RESET)
                h_flag = edge_flag(edge_h) if edge_h >= min_edge else ""
                a_flag = edge_flag(edge_a) if edge_a >= min_edge else ""
                print(f"  {color('Edge:', DIM)}    "
                      f"{card['away_name']}: {ea_str} {a_flag}   "
                      f"{card['home_name']}: {eh_str} {h_flag}")

            for side_name, inj_list in [
                (card["away_name"], card["away_injuries"]),
                (card["home_name"], card["home_injuries"]),
            ]:
                if inj_list:
                    for p in inj_list:
                        status_c = RED if p["status"] in _OUT_STATUSES else YELLOW
                        detail = f" — {p['detail']}" if p["detail"] else ""
                        print(f"  {color('Injury:', DIM)}   {side_name}  "
                              f"{p['name']} ({p['pos']})  "
                              f"{color(p['status'], status_c)}{color(detail, DIM)}")

            if bet:
                bet_count += 1
                bet_name = card["home_name"] if bet == "HOME" else card["away_name"]
                bet_ml = card["best_home_ml"] if bet == "HOME" else card["best_away_ml"]
                bet_edge = edge_h if bet == "HOME" else edge_a
                flag = edge_flag(bet_edge)
                ml_str = f" @ {fmt_ml(bet_ml)}" if bet_ml is not None else ""
                print(f"  {color('► BET', GREEN + BOLD)}  {color(bet_name + ml_str, BOLD)}  "
                      f"edge {color(fmt_edge(bet_edge), GREEN)}  {flag}")
            else:
                print(f"  {color('  PASS', DIM)}  (no side clears {min_edge:.0%} edge threshold)")
            print()

        # Summary
        print(color("━" * 72, CYAN))
        print(f"  {color(str(len(game_cards)), BOLD)} games shown  |  "
              f"{color(str(bet_count), GREEN + BOLD)} with edge ≥ {min_edge:.0%}  |  "
              f"generated {now_et.strftime('%I:%M %p ET')}")
        if bet_count:
            bets = [c for c in game_cards if c["bet_side"]]
            print()
            print(color("  BET SHEET", BOLD))
            for c in bets:
                bet_name = c["home_name"] if c["bet_side"] == "HOME" else c["away_name"]
                bet_ml = c["best_home_ml"] if c["bet_side"] == "HOME" else c["best_away_ml"]
                bet_edge = c["edge_home"] if c["bet_side"] == "HOME" else c["edge_away"]
                ml_str = fmt_ml(bet_ml) if bet_ml is not None else "?"
                print(f"    {edge_flag(bet_edge):10s}  {bet_name:40s}  {ml_str:>6}  edge {fmt_edge(bet_edge)}")

        # CLV stats
        if result.clv_stats:
            stats = result.clv_stats
            avg_clv = stats.get("avg_clv", 0)
            beat_pct = stats.get("beat_closing_pct", 0)
            n_tracked = stats.get("n_games", 0)
            print()
            print(f"  {color('Season CLV:', DIM)}  "
                  f"avg {color(f'{avg_clv:+.2%}', GREEN)}  |  "
                  f"beat close {beat_pct:.1%}  |  "
                  f"{n_tracked} games tracked")

        print(color("━" * 72, CYAN))
        print()


# ── CLI ───────────────────────────────────────────────────────────────────── #

def _parse_dates(args: argparse.Namespace) -> list[date]:
    today = date.today()
    if args.date:
        return [date.fromisoformat(args.date)]
    if args.yesterday:
        return [today - timedelta(days=1)]
    if args.tomorrow:
        return [today + timedelta(days=1)]
    if args.week:
        return [today + timedelta(days=i) for i in range(7)]
    if args.days:
        n = args.days
        if n > 0:
            return [today + timedelta(days=i) for i in range(n)]
        else:
            return [today + timedelta(days=i) for i in range(n, 1)]
    return [today]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bettor's daily overview — edge analysis (multi-sport)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/daily_slate.py                # today (mens)
  python scripts/daily_slate.py --division nba # NBA
  python scripts/daily_slate.py --division mlb # MLB
  python scripts/daily_slate.py --yesterday    # yesterday's results
  python scripts/daily_slate.py --tomorrow     # tomorrow's schedule
  python scripts/daily_slate.py --week         # next 7 days
  python scripts/daily_slate.py --days -3      # last 3 days + today
  python scripts/daily_slate.py --date 2026-03-15
""",
    )
    parser.add_argument("--division",  default="mens",
                        choices=["mens", "womens", "mlb", "nba", "epl", "mls", "ufc"],
                        help="mens, womens, mlb, nba, epl, mls, or ufc (default: mens)")
    parser.add_argument("--min-edge",  type=float, default=0.03,
                        help="Minimum model edge %% to flag as a bet (default: 3%%)")
    parser.add_argument("--top",       type=int, default=None,
                        help="Show only top N games by edge (default: all)")
    parser.add_argument("--no-fetch",  action="store_true",
                        help="Skip live odds API call, use cached snapshots only")
    parser.add_argument("--date",      type=str, default=None,
                        help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--yesterday", action="store_true")
    parser.add_argument("--tomorrow",  action="store_true")
    parser.add_argument("--week",      action="store_true")
    parser.add_argument("--days",      type=int, default=None,
                        help="Number of days (positive=forward, negative=backward)")
    args = parser.parse_args()
    target_dates = _parse_dates(args)
    run(
        division=args.division,
        min_edge=args.min_edge / 100 if args.min_edge > 1 else args.min_edge,
        top_n=args.top,
        no_fetch=args.no_fetch,
        target_dates=target_dates,
    )
