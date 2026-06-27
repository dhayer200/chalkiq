"""Fetch box scores for mens CBB and CFB — callable from automation."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import requests

from src.players.gamescores import parse_player
from src.players.cfb_boxscore import parse_cfb_player
from src.utils.data import fetch_season

_ROOT = Path(__file__).resolve().parents[2]
_DELAY = 0.4

SEASONS: dict[str, dict[int, tuple[date, date]]] = {
    "mens": {
        2025: (date(2024, 11, 4), date(2025, 4, 7)),
        2026: (date(2025, 11, 4), date.today()),
    },
    "cfb": {
        2024: (date(2024, 8, 24), date(2025, 1, 20)),
        2025: (date(2025, 8, 23), date(2026, 1, 19)),
    },
}

ESPN = {
    "mens": ("basketball", "mens-college-basketball"),
    "cfb":  ("football", "college-football"),
}


def _scoreboard(division: str, date_str: str) -> list[dict]:
    cat, league = ESPN[division]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{cat}/{league}/scoreboard"
    try:
        resp = requests.get(url, params={"dates": date_str.replace("-", ""), "limit": 200}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return []
    out = []
    for event in resp.json().get("events", []):
        comp = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue
        out.append({
            "event_id": event["id"],
            "home_espn_id": home.get("team", {}).get("id", ""),
            "away_espn_id": away.get("team", {}).get("id", ""),
        })
    return out


def _summary(division: str, event_id: str) -> list[dict] | None:
    cat, league = ESPN[division]
    url = f"https://site.api.espn.com/apis/site/v2/sports/{cat}/{league}/summary"
    try:
        resp = requests.get(url, params={"event": event_id}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    bs = resp.json().get("boxscore", {})
    blocks = bs.get("players", [])
    if not blocks:
        return None

    players: list[dict] = []
    for team_block in blocks:
        team_id = team_block.get("team", {}).get("id", "")
        for stat_group in team_block.get("statistics", []):
            labels = stat_group.get("labels", [])
            category = stat_group.get("name", stat_group.get("type", ""))
            for ath in stat_group.get("athletes", []):
                if division == "cfb":
                    parsed = parse_cfb_player(ath, team_id, labels, category)
                else:
                    parsed = parse_player(ath, team_id, labels)
                if parsed:
                    players.append(parsed)
    return players or None


def fetch_incremental(division: str, seasons: list[int] | None = None, limit: int = 50) -> int:
    """
    Fetch box scores for games not yet cached. Returns count written.
    Default limit caps ESPN calls per daily run.
    """
    if seasons is None:
        seasons = [2024, 2025] if division == "cfb" else [2026]
    out_dir = _ROOT / "data" / "raw" / division / "boxscores"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.stem for f in out_dir.glob("*.json")}

    all_games: list[dict] = []
    for season in seasons:
        if season not in SEASONS.get(division, {}):
            continue
        start, end = SEASONS[division][season]
        cache = str(_ROOT / "data" / "raw" / division)
        games = fetch_season(start, min(end, date.today()), cache_dir=cache, division=division, verbose=False)
        all_games.extend(games)

    by_date: dict[str, list] = defaultdict(list)
    for g in all_games:
        by_date[g["date"]].append(g)

    written = 0
    for game_date in sorted(by_date.keys(), reverse=True):  # recent first
        for g in by_date[game_date]:
            if written >= limit:
                return written
            key = f"{game_date}_{g['home_id']}_{g['away_id']}"
            if key in existing:
                continue
            time.sleep(_DELAY)
            sb = _scoreboard(division, game_date)
            lookup = {(e["home_espn_id"], e["away_espn_id"]): e["event_id"] for e in sb}
            event_id = lookup.get((g["home_id"], g["away_id"])) or lookup.get((g["away_id"], g["home_id"]))
            if not event_id:
                continue
            time.sleep(_DELAY)
            players = _summary(division, event_id)
            if not players:
                continue
            record = {
                "game_date": game_date,
                "event_id": event_id,
                "home_id": g["home_id"],
                "home_name": g["home_name"],
                "away_id": g["away_id"],
                "away_name": g["away_name"],
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
                "players": players,
            }
            (out_dir / f"{key}.json").write_text(json.dumps(record))
            existing.add(key)
            written += 1
    return written
