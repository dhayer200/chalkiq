"""Generate concise league briefings for SMS delivery."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from src.automation.config import division_in_season, days_until_season_start
from src.slate.generate import load_engine, DIVISION_LABELS
from src.odds.store import load_clv_records
from src.odds.clv import clv_summary, CLVResult


def _clv_snapshot() -> str:
    records = load_clv_records()
    if not records:
        return "CLV: no records yet"
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
    s = clv_summary(results)
    return f"CLV {s['avg_clv']:+.1%} avg | {s['beat_closing_pct']:.0%} beat close ({s['n_games']}g)"


def _top_teams(division: str, n: int = 5) -> list[str]:
    try:
        engine = load_engine(division)
    except Exception:
        return []
    lines = []
    for i, (_, name, elo) in enumerate(engine.rankings()[:n], 1):
        school = name.split()[0]
        lines.append(f"{i}.{school} {round(elo)}")
    return lines


def _top_players(division: str, n: int = 2) -> list[str]:
    path = Path(__file__).resolve().parents[2] / "data" / "players" / f"{division}_effectiveness.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        lines = []
        for p in data.get("players", [])[:n]:
            lines.append(f"{p['name'].split()[-1]} {p['effectiveness']}")
        return lines
    except (json.JSONDecodeError, OSError, KeyError):
        return []


def build_briefing(
    health: dict[str, dict] | None = None,
    *,
    for_date: date | None = None,
) -> str:
    """Plain-text briefing, kept under ~480 chars for single SMS when possible."""
    today = for_date or date.today()
    lines = [f"ChalkIQ {today.strftime('%b %d')}"]

    for division in ("mens", "cfb"):
        label = "CBB" if division == "mens" else "CFB"
        in_season = division_in_season(division, today)

        if in_season:
            status = "in season"
        else:
            days = days_until_season_start(division, today)
            status = f"kickoff in {days}d" if days else "off-season"

        h = (health or {}).get(division, {})
        cov = h.get("coverage_pct", "?")
        lines.append(f"{label}: {status} | data {cov}%")

        tops = _top_teams(division, n=3)
        if tops:
            lines.append(f"  Top: {', '.join(tops)}")
        players = _top_players(division, n=2)
        if players:
            lines.append(f"  Players: {', '.join(players)}")

    lines.append(_clv_snapshot())
    return "\n".join(lines)
