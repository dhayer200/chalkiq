"""
ESPN college football box score parsing and position composites.

Each player gets a cfb_composite float per game used for 0–100 effectiveness.
"""

from __future__ import annotations


def _parse_int(s: str) -> int:
    try:
        return int(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return 0


def _parse_float(s: str) -> float:
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return 0.0


def _split_made_att(s: str) -> tuple[int, int]:
    try:
        a, b = str(s).split("-")
        return int(a), int(b)
    except (ValueError, AttributeError):
        return 0, 0


def _stat(stats: list, labels: list[str], label: str) -> str:
    idx = {lb: i for i, lb in enumerate(labels)}
    i = idx.get(label)
    if i is None or i >= len(stats):
        return "0"
    return stats[i]


def parse_cfb_player(athlete_dict: dict, team_id: str, labels: list[str], category: str) -> dict | None:
    """
    Parse one ESPN CFB athlete row.
    category: passing | rushing | receiving | defensive | ...
    """
    ath = athlete_dict.get("athlete", {})
    stats = athlete_dict.get("stats", [])
    if not stats or all(s in ("--", "", "0") for s in stats[:2]):
        return None

    pos = (ath.get("position", {}) or {}).get("abbreviation", "")
    if not pos and isinstance(ath.get("position"), str):
        pos = ath.get("position", "")

    name = ath.get("displayName") or ath.get("shortName", "")
    pid = str(ath.get("id", ""))
    if not pid or not name:
        return None

    composite = _position_composite(stats, labels, category, pos)
    if composite is None:
        return None

    return {
        "player_id": pid,
        "name": name,
        "team_id": team_id,
        "position": pos,
        "category": category,
        "cfb_composite": composite,
    }


def _position_composite(stats: list, labels: list[str], category: str, pos: str) -> float | None:
    cat = category.lower()

    if "pass" in cat:
        cmp_, att = _split_made_att(_stat(stats, labels, "C/ATT"))
        if att < 5:
            return None
        yds = _parse_int(_stat(stats, labels, "YDS"))
        td = _parse_int(_stat(stats, labels, "TD"))
        ints = _parse_int(_stat(stats, labels, "INT"))
        cmp_pct = cmp_ / att if att else 0
        # NCAA passer efficiency proxy
        return yds * 0.04 + td * 4.0 - ints * 3.0 + cmp_pct * 30

    if "rush" in cat:
        car = _parse_int(_stat(stats, labels, "CAR"))
        if car < 3:
            return None
        yds = _parse_int(_stat(stats, labels, "YDS"))
        td = _parse_int(_stat(stats, labels, "TD"))
        ypc = yds / car if car else 0
        return ypc * 8 + td * 6 + min(yds, 150) * 0.15

    if "rec" in cat or "receiv" in cat:
        rec = _parse_int(_stat(stats, labels, "REC"))
        if rec < 2:
            return None
        yds = _parse_int(_stat(stats, labels, "YDS"))
        td = _parse_int(_stat(stats, labels, "TD"))
        ypr = yds / rec if rec else 0
        return ypr * 2.5 + rec * 1.5 + td * 7

    if "def" in cat or "tack" in cat:
        tot = _parse_int(_stat(stats, labels, "TOT"))
        solo = _parse_int(_stat(stats, labels, "SOLO"))
        sacks = _parse_float(_stat(stats, labels, "SACKS"))
        tfl = _parse_int(_stat(stats, labels, "TFL"))
        pd = _parse_int(_stat(stats, labels, "PD"))
        if tot < 2 and sacks == 0:
            return None
        return tot * 1.2 + solo * 0.5 + sacks * 8 + tfl * 3 + pd * 2

    return None
