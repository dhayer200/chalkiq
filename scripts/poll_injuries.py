#!/usr/bin/env python3
"""
Poll ESPN injury reports and flag status changes.

Run continuously:
    python scripts/poll_injuries.py

Or once:
    python scripts/poll_injuries.py --once

The script:
  1. Fetches injury status for every D1 team.
  2. Compares to last-known status.
  3. Flags any player who changed status (especially -> Out/Doubtful).
  4. Estimates Elo impact of the absence.
  5. Logs all alerts to data/signals/injury_alerts.jsonl

Options:
    --interval N    Scan every N minutes (default: 10)
    --division      mens or womens (default: mens)
    --once          Scan once and exit
    --alerts        Print recent alerts and exit
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.signals.injuries import scan_injuries, load_alerts, estimate_impact, _OUT_STATUSES
from src.live.feed        import fetch_other_games
from src.ratings.elo      import EloEngine
from src.utils.data        import fetch_season


def has_games_today(division: str) -> bool:
    """Return True if there are any games scheduled or live today."""
    games = fetch_other_games(division=division, for_date=date.today())
    return len(games) > 0


def build_engine(division: str = "mens") -> EloEngine:
    configs = {
        "mens":   ("data/raw/mens",   date(2025, 11, 4)),
        "womens": ("data/raw/womens", date(2025, 11, 4)),
    }
    cache_dir, season_start = configs.get(division, configs["mens"])
    games  = fetch_season(season_start, date.today(), cache_dir=cache_dir, division=division, verbose=False)
    engine = EloEngine(k=24.0, home_advantage=100.0)
    engine.process_games(games)
    return engine


def print_alerts(n: int = 20) -> None:
    alerts = load_alerts()
    if not alerts:
        print("No injury alerts recorded yet.")
        return

    print(f"\n{'='*60}")
    print(f"  RECENT INJURY ALERTS (last {min(n, len(alerts))})")
    print(f"{'='*60}")
    for a in alerts[-n:]:
        severity = a.get("severity", "")
        print(
            f"  [{severity:7s}] {a['player_name']:<25} "
            f"({a['position']:<3}) "
            f"{a['prev_status']:<12} -> {a['new_status']:<12} "
            f"  {a['detected_at'][:16]}"
        )
    print(f"{'='*60}\n")


def scan_once(engine: EloEngine, division: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    print(f"\n[{now}] scanning all 360 teams ({division})...")
    alerts = scan_injuries(division=division)

    if not alerts:
        print("  no status changes detected")
        return

    print(f"\n  {len(alerts)} status change(s) detected:")
    for a in alerts:
        team_elo = engine.rating(a["team_id"])
        impact   = estimate_impact(
            player_name=a["player_name"],
            position=a["position"],
            team_elo=team_elo,
            status=a["new_status"],
        )
        if a["new_status"] in _OUT_STATUSES:
            print(
                f"\n  *** {a['severity']} ALERT ***\n"
                f"  Player   : {a['player_name']} ({a['position']})\n"
                f"  Status   : {a['prev_status']} -> {a['new_status']}\n"
                f"  Team Elo : {team_elo:.0f}\n"
                f"  Est. Elo impact: {impact:+.0f} pts\n"
                f"  Detail   : {a.get('detail', 'N/A')}\n"
            )
        else:
            print(
                f"  {a['player_name']} ({a['position']}): "
                f"{a['prev_status']} -> {a['new_status']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll ESPN injury reports")
    parser.add_argument("--interval", type=int, default=10, help="Scan interval in minutes")
    parser.add_argument("--division", default="mens")
    parser.add_argument("--once",     action="store_true")
    parser.add_argument("--alerts",   action="store_true", help="Print recent alerts and exit")
    args = parser.parse_args()

    if args.alerts:
        print_alerts()
        return

    print("Building Elo engine...")
    engine = build_engine(args.division)

    if args.once:
        scan_once(engine, args.division)
        return

    print("Scanning all 360 teams. Interval: 10 min on game days, 60 min otherwise.")
    print("Ctrl+C to stop.")
    while True:
        try:
            scan_once(engine, args.division)
        except Exception as e:
            print(f"  [error] {e}")

        # Game days: scan every 10 min (lines are moving, news breaks fast)
        # Non-game days: scan every 60 min (practice injuries, slower news cycle)
        if has_games_today(args.division):
            interval = 10
        else:
            interval = 60

        print(f"  sleeping {interval} min ({'game day' if interval == 10 else 'no games today'})...")
        time.sleep(interval * 60)


if __name__ == "__main__":
    main()
