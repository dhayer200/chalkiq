#!/usr/bin/env python3
"""
Generate newsletter content from live model data and save to DB.

Run this before the newsletter cron fires (e.g., 13:30 UTC for 14:00 UTC send).
It calls generate_slate() for each active sport and packages the results as JSON
in the newsletter_content table for the Vercel send_newsletter function to read.

Usage:
    python scripts/generate_newsletter_data.py
    python scripts/generate_newsletter_data.py --divisions mens,nba,mlb

Also called by maintain.sh as part of the daily maintenance cycle.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.odds.budget    import active_sports_today
from src.slate.generate import generate_slate, load_engine, DIVISION_LABELS


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate newsletter content from model data")
    parser.add_argument("--divisions", default="mens,nba,mlb,epl,mls,ufc",
                        help="Comma-separated divisions")
    args = parser.parse_args()

    requested = [d.strip() for d in args.divisions.split(",")]
    active = active_sports_today(requested)

    if not active:
        print("No games today — generating empty newsletter content")
        content = {"date": date.today().isoformat(), "sports": []}
        _save(content)
        return

    print(f"Active sports: {', '.join(active)}")
    sports_data = []

    for division in active:
        print(f"\nGenerating slate for {division}...")
        engine = load_engine(division)
        print(f"  {len(engine.ratings)} teams rated")

        result = generate_slate(
            division,
            engine=engine,
            min_edge=0.03,
            no_fetch=True,  # use cached odds (don't burn API calls)
        )

        # Serialize game cards for newsletter (strip non-serializable fields)
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
                "model_prob": round(card["model_home"] if card["bet_side"] == "HOME" else card["model_away"], 4),
                "home_name": card["home_name"],
                "away_name": card["away_name"],
                "tip_detail": card["tip_detail"],
            })

        sport_entry = {
            "division": division,
            "label": DIVISION_LABELS.get(division, division),
            "n_upcoming": result.n_upcoming,
            "n_bets": len(bet_sheet),
            "bet_sheet": bet_sheet,
            "clv_stats": result.clv_stats,
            "model_accuracy": result.model_accuracy,
            "n_final": result.n_final,
        }
        sports_data.append(sport_entry)

        print(f"  {result.n_upcoming} upcoming games, {len(bet_sheet)} bets")
        if result.clv_stats:
            stats = result.clv_stats
            print(f"  CLV: {stats.get('avg_clv', 0):+.2%} avg, "
                  f"{stats.get('beat_closing_pct', 0):.1%} beat close, "
                  f"{stats.get('n_games', 0)} games tracked")

    content = {
        "date": date.today().isoformat(),
        "sports": sports_data,
    }

    _save(content)
    print(f"\nNewsletter content saved ({len(sports_data)} sports)")


def _save(content: dict) -> None:
    """Save to newsletter_content table."""
    # Try DB first, fall back to local JSON
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "api"))
        from _shared.db import save_newsletter_content
        save_newsletter_content(content)
        print("  Saved to newsletter_content table")
    except Exception as e:
        # Fallback: save as local JSON for debugging
        import json
        out_path = Path(__file__).resolve().parents[1] / "data" / "newsletter_latest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(content, indent=2))
        print(f"  DB save failed ({e}) — saved to {out_path}")


if __name__ == "__main__":
    main()
