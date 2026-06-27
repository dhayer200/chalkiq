"""
Daily autonomous pipeline — fetch games, poll odds, injuries, newsletter, landing export.

Invoked by GitHub Actions at midnight ET. No manual CLI required.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from src.automation.config import division_in_season, divisions_for_collection, is_pre_season_prep
from src.automation import ACTIVE_DIVISIONS, MIN_EDGE
from src.automation.backfill import run_proactive_backfill, fill_recent_gaps
from src.automation.briefing import build_briefing
from src.automation.notify import send_briefing
from src.odds.budget import QuotaBudget, active_sports_today
from src.slate.generate import load_engine, generate_slate, DIVISION_LABELS, ODDS_SPORT
from src.utils.data import fetch_day


def _cache_dir(division: str) -> Path:
    p = _ROOT / "data" / "raw" / division
    p.mkdir(parents=True, exist_ok=True)
    return p


def collect_game_results(for_date: date | None = None) -> dict[str, int]:
    """Fetch yesterday + today for in-season divisions; always refresh recent gaps."""
    today = for_date or date.today()
    yesterday = today - timedelta(days=1)
    counts: dict[str, int] = {}

    # Always refresh last 7 days for both divisions (late score corrections)
    for division in ACTIVE_DIVISIONS:
        fill_recent_gaps(division, lookback_days=7)

    for division in divisions_for_collection(today):
        cache = _cache_dir(division)
        n = 0
        for d in (yesterday, today):
            games = fetch_day(d, division=division)
            path = cache / f"{d.strftime('%Y%m%d')}.json"
            path.write_text(json.dumps(games))
            n += len(games)
            print(f"  [{division}] {d}: {len(games)} games cached")
        counts[division] = n

    return counts


def poll_odds_and_clv() -> None:
    from scripts.poll_odds import poll_once, compute_pending_clv
    from src.odds.api import SPORT_NCAAB
    from src.odds.match import build_index

    budget = QuotaBudget()
    active = active_sports_today(list(ACTIVE_DIVISIONS))
    if not active:
        print("  [odds] no games today — skipping")
        return

    print(f"  [odds] active: {', '.join(active)}  {budget.summary()}")
    for division in active:
        sport = ODDS_SPORT.get(division, SPORT_NCAAB)
        engine = load_engine(division)
        index = build_index(engine.names)
        poll_once(engine, sport, division, budget)
        compute_pending_clv(engine, sport, division, index, budget)


def scan_injuries() -> None:
    from src.signals.injuries import scan_injuries, _OUT_STATUSES, estimate_impact
    from src.odds.store import save_alert as save_odds_alert

    for division in divisions_for_collection():
        engine = load_engine(division)
        alerts = scan_injuries(division=division)
        if not alerts:
            print(f"  [injuries/{division}] no changes")
            continue
        for a in alerts:
            impact = estimate_impact(
                player_name=a["player_name"],
                position=a["position"],
                team_elo=engine.rating(a["team_id"]),
                status=a["new_status"],
            )
            save_odds_alert({**a, "type": "injury", "division": division, "elo_impact": impact})
            if a["new_status"] in _OUT_STATUSES:
                print(f"  [injuries/{division}] OUT: {a['player_name']}")


def generate_newsletter_content() -> None:
    active = active_sports_today(list(ACTIVE_DIVISIONS))
    sports_data = []

    if not active:
        content = {"date": date.today().isoformat(), "sports": []}
        _save_newsletter(content)
        print("  [newsletter] no games today — empty content saved")
        return

    for division in active:
        engine = load_engine(division)
        result = generate_slate(
            division,
            engine=engine,
            min_edge=MIN_EDGE.get(division, 0.025),
            no_fetch=False,
        )
        bet_sheet = []
        for card in result.bet_sheet:
            bet_name = card["home_name"] if card["bet_side"] == "HOME" else card["away_name"]
            bet_ml = card["best_home_ml"] if card["bet_side"] == "HOME" else card["best_away_ml"]
            bet_edge = card["edge_home"] if card["bet_side"] == "HOME" else card["edge_away"]
            opp_name = card["away_name"] if card["bet_side"] == "HOME" else card["home_name"]
            bet_sheet.append({
                "bet_name": bet_name,
                "opp_name": opp_name,
                "bet_side": card["bet_side"],
                "bet_ml": bet_ml,
                "edge": round(bet_edge, 4) if bet_edge else None,
                "model_prob": round(
                    card["model_home"] if card["bet_side"] == "HOME" else card["model_away"], 4
                ),
                "home_name": card["home_name"],
                "away_name": card["away_name"],
                "tip_detail": card["tip_detail"],
            })

        sports_data.append({
            "division": division,
            "label": DIVISION_LABELS.get(division, division),
            "n_upcoming": result.n_upcoming,
            "n_bets": len(bet_sheet),
            "bet_sheet": bet_sheet,
            "clv_stats": result.clv_stats,
            "model_accuracy": result.model_accuracy,
            "n_final": result.n_final,
        })
        print(f"  [newsletter/{division}] {result.n_upcoming} games, {len(bet_sheet)} bets")

    _save_newsletter({"date": date.today().isoformat(), "sports": sports_data})


def _save_newsletter(content: dict) -> None:
    try:
        sys.path.insert(0, str(_ROOT / "api"))
        from _shared.db import save_newsletter_content
        save_newsletter_content(content)
        print("  [newsletter] saved to Postgres")
    except Exception as e:
        out = _ROOT / "data" / "newsletter_latest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(content, indent=2))
        print(f"  [newsletter] Postgres failed ({e}) — wrote {out}")


def export_landing_assets() -> None:
    from src.odds.store import load_clv_records

    out_path = _ROOT / "web" / "assets" / "data.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = load_engine("mens")
    rankings = []
    for i, (_, name, elo) in enumerate(engine.rankings()[:25], 1):
        rankings.append({"rank": i, "name": name, "elo": round(elo)})

    records = load_clv_records()
    by_date: dict[str, float] = {}
    for r in sorted(records, key=lambda x: x.get("recorded_at", "")):
        clv = r.get("clv_vs_closing") or 0
        d = (r.get("recorded_at") or "")[:10]
        if d:
            by_date[d] = by_date.get(d, 0) + clv * 100

    running = 0.0
    cumulative = []
    for d in sorted(by_date):
        running += by_date[d]
        cumulative.append({"date": d, "cum_clv": round(running, 2)})

    out_path.write_text(json.dumps({"cumulative_clv": cumulative, "rankings": rankings}, indent=2))
    print(f"  [landing] wrote {out_path}")


def fetch_player_boxscores() -> None:
    from src.players.fetch_boxscores import fetch_incremental

    limit = 80 if is_pre_season_prep() else 40
    for division in ACTIVE_DIVISIONS:
        seasons = [2026] if division == "mens" else [2024, 2025]
        n = fetch_incremental(division, seasons=seasons, limit=limit)
        print(f"  [boxscores/{division}] {n} new games fetched (limit {limit})")


def update_player_effectiveness() -> None:
    from src.players.effectiveness import update_player_ratings
    update_player_ratings(list(ACTIVE_DIVISIONS))


def run_daily_collect(*, skip_odds: bool = False, skip_sms: bool = False) -> None:
    print(f"\n=== ChalkIQ daily collect  {date.today()} ===")

    print("\n[1/8] Proactive backfill (never start cold)...")
    health = run_proactive_backfill(list(ACTIVE_DIVISIONS))

    print("\n[2/8] Game results...")
    counts = collect_game_results()
    print(f"  done: {counts}")

    if not skip_odds:
        print("\n[3/8] Odds + CLV...")
        poll_odds_and_clv()
    else:
        print("\n[3/8] Odds skipped")

    print("\n[4/8] Injuries...")
    scan_injuries()

    print("\n[5/8] Player box scores...")
    fetch_player_boxscores()

    print("\n[6/8] Player effectiveness (0–100)...")
    update_player_effectiveness()

    print("\n[7/8] Newsletter + landing...")
    generate_newsletter_content()
    export_landing_assets()

    if not skip_sms:
        print("\n[8/8] SMS briefing...")
        body = build_briefing(health)
        print(f"  ---\n{body}\n  ---")
        send_briefing(body)
    else:
        print("\n[8/8] SMS skipped")

    print("\n=== Daily collect complete ===\n")


def run_game_day_odds() -> None:
    print(f"\n=== ChalkIQ game-day odds  {date.today()} ===")
    poll_odds_and_clv()
    generate_newsletter_content()
    health = run_proactive_backfill(list(ACTIVE_DIVISIONS))
    body = build_briefing(health)
    send_briefing(body)
    print("=== Game-day odds complete ===\n")
