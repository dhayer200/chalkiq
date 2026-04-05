#!/usr/bin/env python3
"""
Poll ESPN injury reports and flag status changes.

Supports multi-sport polling (NCAA, NBA, MLB).

Run continuously:
    python scripts/poll_injuries.py --divisions mens,nba,mlb

Or once:
    python scripts/poll_injuries.py --divisions mens,nba --once

Options:
    --interval N    Scan every N minutes (default: 10)
    --divisions     Comma-separated: mens,womens,nba,mlb (default: mens)
    --once          Scan once and exit
    --alerts        Print recent alerts and exit
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Fix TLS cert resolution: certifi may cache a stale absolute path if the
# venv was copied from another machine.  Setting SSL_CERT_FILE forces
# requests/urllib3 to use the cert bundle that actually exists on disk.
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.signals.injuries import scan_injuries, load_alerts, estimate_impact, _OUT_STATUSES
from src.odds.budget      import active_sports_today
from src.odds.store       import save_alert as save_odds_alert
from src.ratings.elo      import EloEngine
from src.slate.generate   import load_engine


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
        div = a.get("division", "?")
        print(
            f"  [{div:6s}] [{severity:7s}] {a['player_name']:<25} "
            f"({a['position']:<3}) "
            f"{a['prev_status']:<12} -> {a['new_status']:<12} "
            f"  {a['detected_at'][:16]}"
        )
    print(f"{'='*60}\n")


def scan_once(engine: EloEngine, division: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    print(f"\n[{now}] scanning {division}...")
    alerts = scan_injuries(division=division)

    if not alerts:
        print(f"  [{division}] no status changes detected")
        return

    print(f"\n  [{division}] {len(alerts)} status change(s) detected:")
    for a in alerts:
        team_elo = engine.rating(a["team_id"])
        impact = estimate_impact(
            player_name=a["player_name"],
            position=a["position"],
            team_elo=team_elo,
            status=a["new_status"],
        )

        save_odds_alert({
            **a,
            "type":       "injury",
            "division":   division,
            "player":     a.get("player_name", ""),
            "team":       a.get("team_id", ""),
            "status":     a["new_status"],
            "elo_impact": impact,
        })

        if a["new_status"] in _OUT_STATUSES:
            print(
                f"\n  *** {a['severity']} ALERT ***\n"
                f"  Player   : {a['player_name']} ({a['position']})\n"
                f"  Division : {division}\n"
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

    # Injury detected — retrigger odds + props for this division
    root = Path(__file__).resolve().parents[1]
    print(f"\n  [injury trigger] re-polling odds + props for {division}...")
    subprocess.Popen([sys.executable, "scripts/poll_odds.py", "--divisions", division, "--once"], cwd=root)
    subprocess.Popen([sys.executable, "scripts/poll_props.py", "--divisions", division, "--once"], cwd=root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll ESPN injury reports (multi-sport)")
    parser.add_argument("--interval",   type=int, default=10, help="Scan interval in minutes")
    parser.add_argument("--divisions",  default="mens",
                        help="Comma-separated divisions: mens,womens,nba,mlb")
    parser.add_argument("--once",       action="store_true")
    parser.add_argument("--alerts",     action="store_true", help="Print recent alerts and exit")
    args = parser.parse_args()

    if args.alerts:
        print_alerts()
        return

    requested_divisions = [d.strip() for d in args.divisions.split(",")]

    # Cache engines per division
    engines: dict[str, EloEngine] = {}

    def get_engine(division: str) -> EloEngine:
        if division not in engines:
            print(f"  Building Elo engine for {division}...")
            engines[division] = load_engine(division)
        return engines[division]

    def scan_all():
        active = active_sports_today(requested_divisions)
        if not active:
            print("  No games today for any requested division")
            return

        print(f"  Active sports: {', '.join(active)}")
        for division in active:
            engine = get_engine(division)
            scan_once(engine, division)

    if args.once:
        scan_all()
        return

    print(f"Scanning {args.divisions}. Interval: {args.interval} min. Ctrl+C to stop.")
    while True:
        try:
            scan_all()
        except Exception as e:
            print(f"  [error] {e}")

        print(f"  sleeping {args.interval} min...")
        time.sleep(args.interval * 60)


if __name__ == "__main__":
    main()
