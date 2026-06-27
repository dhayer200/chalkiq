#!/usr/bin/env python3
"""Entry point for scheduled automation (GitHub Actions, cron)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.automation.daily import run_daily_collect, run_game_day_odds


def main() -> None:
    parser = argparse.ArgumentParser(description="ChalkIQ autonomous data pipeline")
    parser.add_argument("--game-day", action="store_true", help="Odds + newsletter only")
    parser.add_argument("--skip-odds", action="store_true", help="Skip Odds API calls")
    parser.add_argument("--skip-sms", action="store_true", help="Skip SMS briefing")
    args = parser.parse_args()

    if args.game_day:
        run_game_day_odds()
    else:
        run_daily_collect(skip_odds=args.skip_odds, skip_sms=args.skip_sms)


if __name__ == "__main__":
    main()
