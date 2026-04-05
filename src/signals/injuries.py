"""
ESPN injury report polling for college basketball, NBA, and MLB.

Polls the ESPN unofficial API every N minutes, detects status changes,
and flags games where a key player has gone from available to out/doubtful.

ESPN injury statuses (roughly):
    "Active"        — fully healthy
    "Day-To-Day"    — questionable
    "Questionable"  — uncertain
    "Doubtful"      — likely out
    "Out"           — confirmed out
    "Injured Reserve" / "Suspension" — definitely out
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

_ESPN_INJURIES_TMPL = (
    "https://site.api.espn.com/apis/site/v2/sports"
    "/{category}/{league}/teams/{team_id}/injuries"
)
_ESPN_TEAMS_TMPL = (
    "https://site.api.espn.com/apis/site/v2/sports"
    "/{category}/{league}/teams"
)

# (category, league) tuples for ESPN API paths
_SPORTS = {
    "mens":   ("basketball", "mens-college-basketball"),
    "womens": ("basketball", "womens-college-basketball"),
    "nba":    ("basketball", "nba"),
    "mlb":    ("baseball",   "mlb"),
    "epl":    ("soccer",     "eng.1"),
    "mls":    ("soccer",     "usa.1"),
    "ufc":    ("mma",        "ufc"),
}

# Statuses that mean a player is likely missing the game
_OUT_STATUSES = {"Out", "Injured Reserve", "Suspension", "Doubtful"}
_CONCERN_STATUSES = {"Questionable", "Day-To-Day"} | _OUT_STATUSES

_DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "signals"


def fetch_team_ids(division: str = "mens") -> list[dict]:
    """
    Fetch all team IDs for a division from ESPN.
    Returns [{team_id, name}].
    """
    category, league = _SPORTS.get(division, _SPORTS["mens"])
    url = _ESPN_TEAMS_TMPL.format(category=category, league=league)
    try:
        resp = requests.get(url, params={"limit": 500}, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [injuries] failed to fetch teams: {e}")
        return []

    teams = []
    for team in resp.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
        t = team.get("team", {})
        teams.append({"team_id": t.get("id", ""), "name": t.get("displayName", "")})
    return teams


def fetch_team_injuries(team_id: str, division: str = "mens") -> list[dict]:
    """
    Fetch injury report for a single team.
    Returns list of player injury dicts.
    """
    category, league = _SPORTS.get(division, _SPORTS["mens"])
    url = _ESPN_INJURIES_TMPL.format(category=category, league=league, team_id=team_id)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    players = []
    for item in resp.json().get("injuries", []):
        athlete = item.get("athlete", {})
        players.append({
            "player_id":   athlete.get("id", ""),
            "player_name": athlete.get("displayName", ""),
            "position":    athlete.get("position", {}).get("abbreviation", ""),
            "status":      item.get("status", "Active"),
            "detail":      item.get("longComment", ""),
            "team_id":     team_id,
        })
    return players


def load_known_statuses(data_dir: Path = _DEFAULT_DIR) -> dict[str, str]:
    """
    Load last-known injury statuses from disk.
    Returns {player_id: status}.
    """
    p = data_dir / "injury_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_known_statuses(statuses: dict[str, str], data_dir: Path = _DEFAULT_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "injury_state.json").write_text(json.dumps(statuses, indent=2))


def append_alert(alert: dict, data_dir: Path = _DEFAULT_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    p = data_dir / "injury_alerts.jsonl"
    with open(p, "a") as f:
        f.write(json.dumps(alert) + "\n")


def load_alerts(data_dir: Path = _DEFAULT_DIR) -> list[dict]:
    p = data_dir / "injury_alerts.jsonl"
    if not p.exists():
        return []
    alerts = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    alerts.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return alerts


def scan_injuries(
    division: str = "mens",
    team_ids: list[str] | None = None,
    data_dir: Path = _DEFAULT_DIR,
    delay: float = 0.3,
) -> list[dict]:
    """
    Scan all teams for injury status changes.

    Compares current ESPN statuses to last-known statuses.
    Returns list of change alerts for newly injured/returned players.

    delay: seconds to sleep between team requests (be polite to ESPN).
    """
    known = load_known_statuses(data_dir)
    updated = dict(known)
    alerts: list[dict] = []

    if team_ids is None:
        teams = fetch_team_ids(division)
        team_ids = [t["team_id"] for t in teams]

    for tid in team_ids:
        players = fetch_team_injuries(tid, division)
        for p in players:
            pid    = p["player_id"]
            status = p["status"]
            prev   = known.get(pid, "Active")

            if status != prev:
                severity = "OUT" if status in _OUT_STATUSES else "CONCERN"
                alert = {
                    **p,
                    "prev_status":  prev,
                    "new_status":   status,
                    "severity":     severity,
                    "detected_at":  datetime.now(timezone.utc).isoformat(),
                }
                alerts.append(alert)
                append_alert(alert, data_dir)
                print(
                    f"  [injury] {severity}: {p['player_name']} "
                    f"({p['position']}) {prev} -> {status}"
                )

            updated[pid] = status
        time.sleep(delay)

    save_known_statuses(updated, data_dir)
    return alerts


def estimate_impact(
    player_name: str,
    position: str,
    team_elo: float,
    status: str,
) -> float:
    """
    Estimate the Elo point impact of a player being out.

    This is a heuristic. A proper model would use minutes/usage data.
    Calibrated for college basketball:
      - Star guard/forward (G/F): ~30-50 Elo pts
      - Starter (C/PF/SF): ~15-30 Elo pts
      - Role player: ~5-15 Elo pts

    Returns negative Elo adjustment (penalty to apply to team's effective rating).
    """
    if status not in _OUT_STATUSES:
        return 0.0

    # Position-based baseline impact
    impact_map = {
        "PG": 40.0,
        "SG": 35.0,
        "SF": 30.0,
        "PF": 20.0,
        "C":  20.0,
        "G":  35.0,
        "F":  25.0,
    }
    base = impact_map.get(position.upper(), 20.0)

    # Scale by team quality: losing a starter hurts stronger teams more
    # (they rely on rotation depth less)
    quality_factor = min(1.5, max(0.8, team_elo / 1500.0))
    return -base * quality_factor
