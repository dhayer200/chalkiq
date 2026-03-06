#!/usr/bin/env python3
"""
daily_slate.py — Bettor's daily overview for today's CBB slate.

Usage:
    python scripts/daily_slate.py                  # mens D1
    python scripts/daily_slate.py --division womens
    python scripts/daily_slate.py --min-edge 3     # only show games with ≥3% edge
    python scripts/daily_slate.py --top 10         # show only top 10 by edge
    python scripts/daily_slate.py --no-fetch       # use cached snapshots only (saves API calls)

Output sections per game:
  • Tip-off time (ET) and matchup
  • Model win probabilities vs opening line vs current/best line
  • Edge on each side (model minus market)
  • Best available line across books
  • Elo ratings and rank
  • Injury report (OUT / Doubtful / Questionable players for both teams)
  • Line movement summary (open → now, direction)
  • Recommendation: BET HOME / BET AWAY / PASS (based on edge threshold)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.live.feed        import fetch_other_games
from src.odds.api         import fetch_odds, SPORT_NCAAB
from src.odds.store       import load_snapshots
from src.odds.match       import build_index, match_team
from src.ratings.elo      import EloEngine
from src.signals.injuries import fetch_team_injuries, _CONCERN_STATUSES, _OUT_STATUSES
from src.utils.data       import fetch_season

ET = ZoneInfo("America/New_York")

BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]

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

# ── Elo engine ────────────────────────────────────────────────────────────── #

def load_engine(division: str) -> EloEngine:
    from datetime import date as _date
    ROOT = Path(__file__).resolve().parents[1]
    season_start = _date(2025, 11, 4)
    season_end   = _date.today()
    cache_dir    = ROOT / "data" / "raw" / division
    print(color("  Loading Elo ratings...", DIM), end="\r")
    games = fetch_season(season_start, season_end,
                         cache_dir=str(cache_dir), division=division, verbose=False)
    eng = EloEngine(k=24.0, home_advantage=100.0)
    eng.process_games(games)
    print(f"  Elo engine: {len(eng.ratings)} teams rated from {len(games)} games")
    return eng

# ── Injury report ─────────────────────────────────────────────────────────── #

def get_injury_report(team_id: str, team_name: str, division: str) -> list[dict]:
    """Fetch live injury report for one team. Returns concern-level players."""
    raw = fetch_team_injuries(team_id, division=division)
    out = []
    for p in raw:
        status = p.get("status", "")
        if status in _CONCERN_STATUSES:
            out.append({
                "name":   p.get("athlete", {}).get("displayName", p.get("name", "?")),
                "pos":    p.get("athlete", {}).get("position", {}).get("abbreviation", "?"),
                "status": status,
                "detail": p.get("shortComment") or p.get("longComment") or "",
            })
    return out

# ── Opening line from snapshot history ───────────────────────────────────── #

def get_opening_lines(snapshots: list[dict]) -> dict[tuple, dict]:
    """
    Returns {(game_id, bookmaker): first_snapshot} across all historical snapshots.
    """
    seen: dict[tuple, dict] = {}
    for s in snapshots:
        key = (s.get("game_id", ""), s.get("bookmaker", ""))
        if key not in seen:
            seen[key] = s
    return seen

# ── Main ──────────────────────────────────────────────────────────────────── #

def run(division: str, min_edge: float, top_n: int | None, no_fetch: bool) -> None:
    today = date.today()
    now_et = datetime.now(ET)

    print()
    print(color("━" * 72, CYAN))
    print(color(f"  ChalkIQ Daily Slate  —  {today.strftime('%A, %B %d %Y')}  ({division.upper()})", BOLD))
    print(color("━" * 72, CYAN))
    print()

    # ── 1. Today's games ─────────────────────────────────────────────────── #
    print(color("  Fetching today's schedule from ESPN...", DIM), end="\r")
    games_today = fetch_other_games(division=division, for_date=today)
    scheduled   = [g for g in games_today if g["status"] == "scheduled"]
    live_now    = [g for g in games_today if g["status"] not in ("scheduled", "final")]
    final_today = [g for g in games_today if g["status"] == "final"]

    print(f"  Schedule: {len(scheduled)} upcoming  |  {len(live_now)} live  |  {len(final_today)} final today")
    print()

    if not scheduled and not live_now:
        print(color("  No games scheduled today. Enjoy the day off.", YELLOW))
        return

    all_upcoming = live_now + scheduled

    # ── 2. Elo engine ─────────────────────────────────────────────────────── #
    engine = load_engine(division)
    rankings = {tid: rank for rank, (tid, _, _r) in enumerate(engine.rankings(), 1)}
    index    = build_index(engine.names)

    # ── 3. Live odds ─────────────────────────────────────────────────────── #
    live_odds: list[dict] = []
    if not no_fetch:
        print(color("  Fetching live odds from The Odds API...", DIM), end="\r")
        try:
            live_odds = fetch_odds(sport=SPORT_NCAAB, bookmakers=",".join(BOOKS))
            print(f"  Live odds: {len(live_odds)} records across {len({o['game_id'] for o in live_odds})} games")
        except EnvironmentError as e:
            print(color(f"  [warn] {e}", YELLOW))
    else:
        print(color("  --no-fetch: skipping live odds API call", DIM))

    # ── 4. Historical snapshots (opening lines) ───────────────────────────── #
    snap_dir = Path(__file__).resolve().parents[1] / "data" / "odds"
    all_snaps = load_snapshots(snap_dir)
    opening_lines = get_opening_lines(all_snaps)

    # Group live odds by game_id → best line per side across books
    live_by_game: dict[str, list[dict]] = defaultdict(list)
    for o in live_odds:
        live_by_game[o["game_id"]].append(o)

    # ── 5. Injury cache (fetch once per team seen today) ──────────────────── #
    injury_cache: dict[str, list[dict]] = {}  # team_id -> players

    def get_injuries(team_id: str, team_name: str) -> list[dict]:
        if team_id not in injury_cache:
            injury_cache[team_id] = get_injury_report(team_id, team_name, division)
        return injury_cache[team_id]

    # ── 6. Build per-game overview ──────────────────────────────────────────
    game_cards: list[dict] = []

    for g in all_upcoming:
        home_id   = g["home_id"]
        away_id   = g["away_id"]
        home_name = g["home_name"]
        away_name = g["away_name"]

        # Model probabilities
        home_elo = engine.rating(home_id)
        away_elo = engine.rating(away_id)
        model_home = engine.win_prob(home_id, away_id, neutral=g.get("neutral", False))
        model_away = 1.0 - model_home

        home_rank = rankings.get(home_id, 999)
        away_rank = rankings.get(away_id, 999)

        # Match to odds API game
        matched_odds_id: str | None = None
        for oid, odds_list in live_by_game.items():
            if not odds_list:
                continue
            s = odds_list[0]
            h_match = match_team(s["home_team"], index)
            a_match = match_team(s["away_team"], index)
            if h_match == home_id and a_match == away_id:
                matched_odds_id = oid
                break
            if h_match == away_id and a_match == home_id:
                matched_odds_id = oid
                break

        # Best current line per side
        best_home_ml: int | None = None
        best_away_ml: int | None = None
        best_home_prob: float | None = None
        best_away_prob: float | None = None
        current_lines: dict[str, dict] = {}  # bookmaker -> {home_ml, away_ml, home_prob}

        if matched_odds_id:
            for o in live_by_game[matched_odds_id]:
                book = o["bookmaker"]
                # Determine orientation (odds API may flip home/away vs ESPN)
                s0 = live_by_game[matched_odds_id][0]
                h_match = match_team(s0["home_team"], index)
                flipped = (h_match == away_id)
                h_prob = o["away_prob"] if flipped else o["home_prob"]
                a_prob = o["home_prob"] if flipped else o["away_prob"]
                h_ml   = o["away_ml"]   if flipped else o["home_ml"]
                a_ml   = o["home_ml"]   if flipped else o["away_ml"]
                current_lines[book] = {"home_ml": h_ml, "away_ml": a_ml,
                                       "home_prob": h_prob, "away_prob": a_prob}
                if best_home_prob is None or h_prob < best_home_prob:  # lower prob = better price for bettor (longer odds)
                    pass  # find best price (most favorable ML)
                if best_home_ml is None or (h_ml < 0 and h_ml > (best_home_ml or -9999)) or \
                   (h_ml >= 0 and (best_home_ml is None or best_home_ml < 0 or h_ml > best_home_ml)):
                    best_home_ml = h_ml
                    best_home_prob = h_prob
                if best_away_ml is None or (a_ml < 0 and a_ml > (best_away_ml or -9999)) or \
                   (a_ml >= 0 and (best_away_ml is None or best_away_ml < 0 or a_ml > best_away_ml)):
                    best_away_ml = a_ml
                    best_away_prob = a_prob

        # Opening lines (from snapshots)
        open_home_ml: int | None = None
        open_home_prob: float | None = None
        for book in BOOKS:
            if matched_odds_id:
                ok = (matched_odds_id, book)
                if ok in opening_lines:
                    snap = opening_lines[ok]
                    s0 = live_by_game.get(matched_odds_id, [{}])[0]
                    h_match = match_team(s0.get("home_team", ""), index) if s0 else None
                    flipped = (h_match == away_id)
                    open_home_ml   = snap["away_ml"]   if flipped else snap["home_ml"]
                    open_home_prob = snap["away_prob"]  if flipped else snap["home_prob"]
                    break

        # Edge calculations (use best available current line, or any line)
        current_home_prob = best_home_prob
        current_away_prob = best_away_prob

        edge_home = (model_home - current_home_prob) if current_home_prob else None
        edge_away = (model_away - current_away_prob) if current_away_prob else None

        # Line movement
        line_move_str = ""
        if open_home_prob is not None and current_home_prob is not None:
            delta = current_home_prob - open_home_prob
            if abs(delta) >= 0.005:
                direction = "toward home" if delta > 0 else "toward away"
                line_move_str = f"{direction} ({delta:+.1%})"
            else:
                line_move_str = "stable"

        # Injuries
        home_injuries = get_injuries(home_id, home_name)
        away_injuries = get_injuries(away_id, away_name)

        # Tip time
        detail = g.get("status_detail", "")
        status = g["status"]

        # Max edge (for sorting)
        max_edge = max(
            abs(edge_home) if edge_home is not None else 0,
            abs(edge_away) if edge_away is not None else 0,
        )
        bet_side = None
        if edge_home is not None and edge_away is not None:
            if edge_home >= min_edge:
                bet_side = "HOME"
            elif edge_away >= min_edge:
                bet_side = "AWAY"

        game_cards.append({
            "home_name": home_name,
            "away_name": away_name,
            "home_id":   home_id,
            "away_id":   away_id,
            "home_elo":  home_elo,
            "away_elo":  away_elo,
            "home_rank": home_rank,
            "away_rank": away_rank,
            "model_home": model_home,
            "model_away": model_away,
            "current_home_prob": current_home_prob,
            "current_away_prob": current_away_prob,
            "best_home_ml":  best_home_ml,
            "best_away_ml":  best_away_ml,
            "open_home_ml":  open_home_ml,
            "open_home_prob": open_home_prob,
            "edge_home":  edge_home,
            "edge_away":  edge_away,
            "max_edge":   max_edge,
            "bet_side":   bet_side,
            "current_lines": current_lines,
            "line_move":  line_move_str,
            "home_injuries": home_injuries,
            "away_injuries": away_injuries,
            "tip_detail": detail,
            "status":     status,
        })

    # Sort by edge descending
    game_cards.sort(key=lambda x: x["max_edge"], reverse=True)
    if top_n:
        game_cards = game_cards[:top_n]

    # ── 7. Print cards ─────────────────────────────────────────────────────── #

    bet_count = 0
    for card in game_cards:
        edge_h = card["edge_home"]
        edge_a = card["edge_away"]
        bet    = card["bet_side"]
        status = card["status"]

        # Header
        status_tag = color(" [LIVE] ", RED + BOLD) if status == "in" else ""
        matchup = f"{card['away_name']}  @  {card['home_name']}"
        print(color("─" * 72, DIM))
        print(f"  {color(matchup, BOLD)}{status_tag}   {color(card['tip_detail'], DIM)}")

        # Elo ratings
        home_r = f"#{card['home_rank']} {card['home_name']} ({card['home_elo']:.0f})"
        away_r = f"#{card['away_rank']} {card['away_name']} ({card['away_elo']:.0f})"
        print(f"  {color('Elo:', DIM)} {away_r}  vs  {home_r}")

        # Model probs
        print(f"  {color('Model:', DIM)}  {card['away_name']}: {fmt_prob(card['model_away'])}   "
              f"{card['home_name']}: {fmt_prob(card['model_home'])}")

        # Odds table
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

        # Edge
        if edge_h is not None:
            eh_str = color(fmt_edge(edge_h), GREEN if edge_h >= min_edge else RESET)
            ea_str = color(fmt_edge(edge_a), GREEN if edge_a >= min_edge else RESET)
            h_flag = edge_flag(edge_h) if edge_h >= min_edge else ""
            a_flag = edge_flag(edge_a) if edge_a >= min_edge else ""
            print(f"  {color('Edge:', DIM)}    "
                  f"{card['away_name']}: {ea_str} {a_flag}   "
                  f"{card['home_name']}: {eh_str} {h_flag}")

        # Injuries
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

        # Recommendation
        if bet:
            bet_count += 1
            bet_name = card["home_name"] if bet == "HOME" else card["away_name"]
            bet_ml   = card["best_home_ml"] if bet == "HOME" else card["best_away_ml"]
            bet_edge = edge_h if bet == "HOME" else edge_a
            flag     = edge_flag(bet_edge)
            ml_str   = f" @ {fmt_ml(bet_ml)}" if bet_ml is not None else ""
            print(f"  {color('► BET', GREEN + BOLD)}  {color(bet_name + ml_str, BOLD)}  "
                  f"edge {color(fmt_edge(bet_edge), GREEN)}  {flag}")
        else:
            print(f"  {color('  PASS', DIM)}  (no side clears {min_edge:.0%} edge threshold)")

        print()

    # ── 8. Summary ─────────────────────────────────────────────────────────── #
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
            bet_ml   = c["best_home_ml"] if c["bet_side"] == "HOME" else c["best_away_ml"]
            bet_edge = c["edge_home"] if c["bet_side"] == "HOME" else c["edge_away"]
            ml_str   = fmt_ml(bet_ml) if bet_ml is not None else "?"
            print(f"    {edge_flag(bet_edge):10s}  {bet_name:40s}  {ml_str:>6}  edge {fmt_edge(bet_edge)}")
    print(color("━" * 72, CYAN))
    print()


# ── CLI ───────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bettor's daily overview — today's CBB slate with edge analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--division",  default="mens", choices=["mens", "womens"],
                        help="mens or womens D1 (default: mens)")
    parser.add_argument("--min-edge",  type=float, default=0.03,
                        help="Minimum model edge %% to flag as a bet (default: 3%%)")
    parser.add_argument("--top",       type=int, default=None,
                        help="Show only top N games by edge (default: all)")
    parser.add_argument("--no-fetch",  action="store_true",
                        help="Skip live odds API call, use cached snapshots only")
    args = parser.parse_args()
    run(
        division=args.division,
        min_edge=args.min_edge / 100 if args.min_edge > 1 else args.min_edge,
        top_n=args.top,
        no_fetch=args.no_fetch,
    )
