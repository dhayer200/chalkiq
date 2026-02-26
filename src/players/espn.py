"""
ESPN basketball player statistics — box-score aggregation.

ESPN's /leaders endpoint returns 404 for college basketball.  Instead we
aggregate player stats from individual game box-score summaries fetched via
the /summary?event=<id> endpoint, which reliably exposes full per-player
box scores for completed games.

Player Rating (PR) formula — inspired by Hollinger Game Score:
    PR = PPG + 0.4×RPG + 0.7×APG + 1.0×SPG + 0.7×BPG − 0.7×TPG

Usage:
    from src.players.espn import fetch_player_leaders
    players = fetch_player_leaders(division="mens", days_back=21, limit=100)
    # [{"player_id": ..., "name": ..., "team_name": ...,
    #   "stats": {"pointsPerGame": 19.4, ...}, "player_rating": 22.1}, ...]
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import requests

_SCOREBOARD_TMPL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/{sport}/scoreboard"
)
_SUMMARY_TMPL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/{sport}/summary"
)

_SPORTS = {
    "mens":   "mens-college-basketball",
    "womens": "womens-college-basketball",
    "nba":    "nba",
}

_FINAL_STATUSES = {"STATUS_FINAL", "STATUS_FINAL_OT", "STATUS_FINAL_OVERTIME"}

# stat_name -> (display_label, PR weight)
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
    "gamesPlayed":            ("GP",   0.0),
}


def player_rating(stats: dict[str, float]) -> float:
    """Composite Player Rating from season-average counting stats."""
    pr = 0.0
    for stat_name, (_, weight) in STAT_CONFIG.items():
        if weight != 0.0:
            pr += weight * stats.get(stat_name, 0.0)
    return round(pr, 2)


def _get_game_ids(
    sport: str,
    days_back: int,
    max_games: int,
    max_per_day: int = 8,
) -> list[str]:
    """
    Collect ESPN game IDs for completed games over the last `days_back` days.
    Takes at most `max_per_day` games per day so the sample spans multiple days
    (CBB has 100+ games/day which would otherwise exhaust max_games in one day).
    """
    url = _SCOREBOARD_TMPL.format(sport=sport)
    game_ids: list[str] = []
    today = date.today()

    for d in range(1, days_back + 1):
        check_date = today - timedelta(days=d)
        try:
            resp = requests.get(
                url,
                params={"dates": check_date.strftime("%Y%m%d"), "limit": 200},
                timeout=8,
            )
            resp.raise_for_status()
        except requests.RequestException:
            continue

        day_count = 0
        for event in resp.json().get("events", []):
            stype = event.get("status", {}).get("type", {}).get("name", "")
            if stype in _FINAL_STATUSES:
                gid = event.get("id", "")
                if gid:
                    game_ids.append(gid)
                    day_count += 1
                    if day_count >= max_per_day:
                        break

        if len(game_ids) >= max_games:
            break

    return game_ids[:max_games]


def _parse_ratio(s: str) -> tuple[int, int]:
    """Parse 'made-attempted' ESPN stat string → (made, attempted)."""
    parts = str(s).split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return 0, 0


def _aggregate_from_boxscores(
    sport: str,
    game_ids: list[str],
) -> dict[str, dict]:
    """
    Fetch game summaries and aggregate per-player box-score stats.
    Returns a dict keyed by player_id.
    """
    url = _SUMMARY_TMPL.format(sport=sport)
    player_acc: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()   # (player_id, game_id) dedup

    for gid in game_ids:
        try:
            resp = requests.get(url, params={"event": gid}, timeout=8)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException:
            continue
        time.sleep(0.05)   # polite pacing to avoid ESPN rate limiting

        box = data.get("boxscore", {})
        for team_data in box.get("players", []):
            team_info = team_data.get("team", {})
            team_id   = team_info.get("id", "")
            team_name = team_info.get("displayName", "")

            for stat_cat in team_data.get("statistics", []):
                keys = stat_cat.get("keys", [])

                def _idx(name: str) -> int:
                    try:
                        return keys.index(name)
                    except ValueError:
                        return -1

                i_pts = _idx("points")
                i_reb = _idx("rebounds")
                i_ast = _idx("assists")
                i_tov = _idx("turnovers")
                i_stl = _idx("steals")
                i_blk = _idx("blocks")
                i_fg  = _idx("fieldGoalsMade-fieldGoalsAttempted")
                i_tp  = _idx("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
                i_ft  = _idx("freeThrowsMade-freeThrowsAttempted")

                for ath_entry in stat_cat.get("athletes", []):
                    ath = ath_entry.get("athlete", {})
                    pid = ath.get("id", "")
                    if not pid:
                        continue

                    pair = (pid, gid)
                    if pair in seen:
                        continue
                    seen.add(pair)

                    sarr = ath_entry.get("stats", [])

                    def gf(i: int) -> float:
                        if i < 0 or i >= len(sarr):
                            return 0.0
                        try:
                            return float(sarr[i])
                        except (ValueError, TypeError):
                            return 0.0

                    def gr(i: int) -> tuple[int, int]:
                        if i < 0 or i >= len(sarr):
                            return 0, 0
                        return _parse_ratio(sarr[i])

                    if pid not in player_acc:
                        player_acc[pid] = {
                            "player_id": pid,
                            "name":      ath.get("displayName", ""),
                            "team_id":   team_id,
                            "team_name": team_name,
                            "g":   0,
                            "pts": 0.0, "reb": 0.0, "ast": 0.0,
                            "tov": 0.0, "stl": 0.0, "blk": 0.0,
                            "fgm": 0,   "fga": 0,
                            "tpm": 0,   "tpa": 0,
                            "ftm": 0,   "fta": 0,
                        }

                    p = player_acc[pid]
                    p["g"]   += 1
                    p["pts"] += gf(i_pts)
                    p["reb"] += gf(i_reb)
                    p["ast"] += gf(i_ast)
                    p["tov"] += gf(i_tov)
                    p["stl"] += gf(i_stl)
                    p["blk"] += gf(i_blk)
                    fgm, fga = gr(i_fg)
                    p["fgm"] += fgm;  p["fga"] += fga
                    tpm, tpa = gr(i_tp)
                    p["tpm"] += tpm;  p["tpa"] += tpa
                    ftm, fta = gr(i_ft)
                    p["ftm"] += ftm;  p["fta"] += fta

    return player_acc


def fetch_player_leaders(
    division: str = "mens",
    limit: int = 100,
    days_back: int = 21,
    max_games: int = 60,
    min_games: int = 3,
) -> list[dict]:
    """
    Fetch top players by aggregating recent game box scores from ESPN.

    Parameters
    ----------
    division  : "mens" | "womens" | "nba"
    limit     : max players to return (sorted by PR descending)
    days_back : how many days of games to search
    max_games : cap on number of game summaries fetched (controls speed)
    min_games : minimum games played to be included

    Returns list of dicts (same format expected by app.py):
    {
        "player_id":     str,
        "name":          str,
        "team_id":       str,
        "team_name":     str,
        "stats":         dict[str, float],
        "player_rating": float,
    }
    """
    sport    = _SPORTS.get(division, _SPORTS["mens"])
    game_ids = _get_game_ids(sport, days_back, max_games)

    if not game_ids:
        return []

    player_acc = _aggregate_from_boxscores(sport, game_ids)

    result: list[dict] = []
    for p in player_acc.values():
        if p["g"] < min_games:
            continue
        g = p["g"]
        ppg = p["pts"] / g
        rpg = p["reb"] / g
        apg = p["ast"] / g
        tpg = p["tov"] / g
        spg = p["stl"] / g
        bpg = p["blk"] / g
        fgpct = (p["fgm"] / p["fga"] * 100) if p["fga"] > 0 else 0.0
        tpct  = (p["tpm"] / p["tpa"] * 100) if p["tpa"] > 0 else 0.0
        ftpct = (p["ftm"] / p["fta"] * 100) if p["fta"] > 0 else 0.0

        stats = {
            "pointsPerGame":          round(ppg, 1),
            "reboundsPerGame":        round(rpg, 1),
            "assistsPerGame":         round(apg, 1),
            "stealsPerGame":          round(spg, 1),
            "blocksPerGame":          round(bpg, 1),
            "turnoversPerGame":       round(tpg, 1),
            "fieldGoalPct":           round(fgpct, 1),
            "threePointFieldGoalPct": round(tpct, 1),
            "freeThrowPct":           round(ftpct, 1),
            "gamesPlayed":            g,
        }
        pr = player_rating(stats)
        result.append({
            "player_id":     p["player_id"],
            "name":          p["name"],
            "team_id":       p["team_id"],
            "team_name":     p["team_name"],
            "stats":         stats,
            "player_rating": pr,
        })

    return sorted(result, key=lambda x: x["player_rating"], reverse=True)[:limit]
