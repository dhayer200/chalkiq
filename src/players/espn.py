"""
ESPN CBB player statistics fetcher.

Fetches season stat leaders from ESPN's unofficial API and computes
a composite Player Rating (PR) modelled after game-score metrics.

Player Rating:
    PR = PPG + 0.4*RPG + 0.7*APG + 1.0*SPG + 0.7*BPG - 0.7*TPG

Usage:
    from src.players.espn import fetch_player_leaders, STAT_CONFIG
    players = fetch_player_leaders(division="mens", limit=100)
    # [{"player_id": ..., "name": ..., "team_name": ...,
    #   "stats": {"pointsPerGame": 19.4, ...}, "player_rating": 22.1}, ...]
"""

from __future__ import annotations

import requests

_ESPN_LEADERS_TMPL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/{sport}/leaders"
)

_SPORTS = {
    "mens":   "mens-college-basketball",
    "womens": "womens-college-basketball",
}

# stat_name -> (display_label, PR_weight)
# weight=0.0 means display-only (not used in PR calculation)
STAT_CONFIG: dict[str, tuple[str, float]] = {
    "pointsPerGame":          ("PPG",  1.0),
    "reboundsPerGame":        ("RPG",  0.4),
    "assistsPerGame":         ("APG",  0.7),
    "stealsPerGame":          ("SPG",  1.0),
    "blocksPerGame":          ("BPG",  0.7),
    "turnoversPerGame":       ("TPG", -0.7),
    "fieldGoalPct":           ("FG%",  0.0),
    "threePointFieldGoalPct": ("3P%",  0.0),
    "freeThrowPct":           ("FT%",  0.0),
}


def _parse_float(s: str) -> float:
    """Parse a display value like '19.4' or '58.3%' to float."""
    try:
        return float(str(s).replace("%", "").strip())
    except (ValueError, AttributeError):
        return 0.0


def player_rating(stats: dict[str, float]) -> float:
    """Composite Player Rating from season-average counting stats."""
    pr = 0.0
    for stat_name, (_, weight) in STAT_CONFIG.items():
        if weight != 0.0:
            pr += weight * stats.get(stat_name, 0.0)
    return round(pr, 2)


def fetch_player_leaders(
    division: str = "mens",
    limit: int = 100,
) -> list[dict]:
    """
    Fetch top college basketball players by season averages.

    Returns a list of dicts sorted by player_rating descending:
    {
        "player_id":     str,
        "name":          str,
        "team_id":       str,
        "team_name":     str,
        "stats":         dict[str, float],   # stat_name -> value
        "player_rating": float,              # composite PR
    }

    Returns an empty list on network error or if the API returns no data.
    """
    sport = _SPORTS.get(division, _SPORTS["mens"])
    url   = _ESPN_LEADERS_TMPL.format(sport=sport)

    try:
        resp = requests.get(url, params={"limit": limit}, timeout=12)
        resp.raise_for_status()
    except requests.RequestException:
        return []

    players: dict[str, dict] = {}

    for cat in resp.json().get("categories", []):
        cat_name = cat.get("name", "")
        if cat_name not in STAT_CONFIG:
            continue

        for entry in cat.get("leaders", []):
            ath = entry.get("athlete", {})
            pid = ath.get("id", "")
            if not pid:
                continue

            if pid not in players:
                team = ath.get("team", {})
                players[pid] = {
                    "player_id": pid,
                    "name":      ath.get("displayName", ""),
                    "team_id":   team.get("id", ""),
                    "team_name": team.get("displayName", ""),
                    "stats":     {},
                }

            val = _parse_float(entry.get("displayValue", "0"))
            # ESPN percentage stats are sometimes returned as 0–1 proportions;
            # normalise to 0–100 for display consistency.
            if "Pct" in cat_name and val <= 1.0:
                val *= 100.0
            players[pid]["stats"][cat_name] = val

    result = []
    for p in players.values():
        p["player_rating"] = player_rating(p["stats"])
        result.append(p)

    return sorted(result, key=lambda x: x["player_rating"], reverse=True)
