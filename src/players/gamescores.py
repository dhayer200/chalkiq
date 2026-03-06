"""
Game Score calculation and ESPN box score parsing.

Game Score (Hollinger):
    GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM)
         + 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK
         - 0.4*PF - TO

Typical values:
    Elite game  : 25+
    Good game   : 15-25
    Average     : ~8
    Bad game    : <5
    DNP / trash : 0
"""

from __future__ import annotations

import math

# League-wide averages used to normalize GmSc → [0,1] outcome
LEAGUE_AVG_GMSC: float = 8.0
LEAGUE_STD_GMSC: float = 8.0


def _parse_split(s: str) -> tuple[int, int]:
    """Parse '8-12' → (8, 12). Returns (0, 0) on failure."""
    try:
        m, a = s.split("-")
        return int(m), int(a)
    except Exception:
        return 0, 0


def _parse_int(s: str) -> int:
    try:
        return int(s)
    except Exception:
        return 0


def game_score(
    pts: int,
    fg_m: int, fg_a: int,
    ft_m: int, ft_a: int,
    oreb: int, dreb: int,
    stl: int, ast: int, blk: int,
    pf: int, to_: int,
) -> float:
    """Hollinger Game Score."""
    return (
        pts
        + 0.4 * fg_m
        - 0.7 * fg_a
        - 0.4 * (ft_a - ft_m)
        + 0.7 * oreb
        + 0.3 * dreb
        + stl
        + 0.7 * ast
        + 0.7 * blk
        - 0.4 * pf
        - to_
    )


def gmsc_to_outcome(
    gmsc: float,
    avg: float = LEAGUE_AVG_GMSC,
    sigma: float = LEAGUE_STD_GMSC,
) -> float:
    """
    Normalize game score to [0, 1] outcome via logistic function.
    Mirrors the win probability outcome used in team Elo.
    avg=8 means an average player scores exactly 0.5 (coin flip).
    """
    return 1.0 / (1.0 + math.exp(-(gmsc - avg) / sigma))


def parse_player(athlete_dict: dict, team_id: str, labels: list[str]) -> dict | None:
    """
    Parse a single ESPN athlete dict from the boxscore into a clean player record.
    Returns None for DNPs or players with <5 minutes.
    """
    ath   = athlete_dict.get("athlete", {})
    stats = athlete_dict.get("stats", [])

    # DNP sentinel: all dashes
    if not stats or all(s in ("--", "", "0") for s in stats[:3]):
        return None

    idx = {label: i for i, label in enumerate(labels)}

    def _s(label: str) -> str:
        i = idx.get(label)
        return stats[i] if (i is not None and i < len(stats)) else "0"

    min_         = _parse_int(_s("MIN"))
    if min_ < 5:
        return None  # garbage time / DNP

    pts          = _parse_int(_s("PTS"))
    fg_m, fg_a   = _parse_split(_s("FG"))
    fg3_m, fg3_a = _parse_split(_s("3PT"))
    ft_m, ft_a   = _parse_split(_s("FT"))
    oreb         = _parse_int(_s("OREB"))
    dreb         = _parse_int(_s("DREB"))
    reb          = _parse_int(_s("REB"))
    ast          = _parse_int(_s("AST"))
    to_          = _parse_int(_s("TO"))
    stl          = _parse_int(_s("STL"))
    blk          = _parse_int(_s("BLK"))
    pf           = _parse_int(_s("PF"))

    gmsc = game_score(pts, fg_m, fg_a, ft_m, ft_a, oreb, dreb, stl, ast, blk, pf, to_)

    return {
        "player_id": ath.get("id", ""),
        "name":      ath.get("displayName", ""),
        "team_id":   team_id,
        "position":  ath.get("position", {}).get("abbreviation", ""),
        "starter":   athlete_dict.get("starter", False),
        "min":       min_,
        "pts":       pts,
        "fg_m":      fg_m,  "fg_a":  fg_a,
        "fg3_m":     fg3_m, "fg3_a": fg3_a,
        "ft_m":      ft_m,  "ft_a":  ft_a,
        "oreb":      oreb,  "dreb":  dreb, "reb": reb,
        "ast":       ast,   "to":    to_,
        "stl":       stl,   "blk":   blk,
        "pf":        pf,
        "game_score": round(gmsc, 2),
        "off_score":  round(pts + 0.7 * ast - to_, 2),
    }
