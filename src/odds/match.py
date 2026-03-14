"""
Team name matching between ESPN display names and The Odds API team names.

ESPN returns full names like "Duke Blue Devils", "North Carolina Tar Heels".
The Odds API returns short names like "Duke", "North Carolina".

Strategy:
  1. Check manual overrides first (known mismatches).
  2. Normalize: lowercase, strip common suffixes.
  3. Fuzzy match with difflib against all known ESPN names.
"""

from __future__ import annotations

import difflib
import re

# Manual overrides: Odds API name -> ESPN display name substring
_OVERRIDES: dict[str, str] = {
    # NCAA basketball
    "uconn":              "connecticut",
    "connecticut":        "connecticut",
    "lsu":                "lsu",
    "smu":                "smu",
    "tcu":                "tcu",
    "vcu":                "vcu",
    "usc":                "southern california",
    "ucf":                "ucf",
    "uab":                "uab",
    "ole miss":           "mississippi",
    "mississippi":        "mississippi",
    "pitt":               "pittsburgh",
    "miami (oh)":         "miami (oh)",
    "miami (fl)":         "miami",
    "st. john's":         "st. john",
    "saint john's":       "st. john",
    "unc":                "north carolina",
    "nc state":           "nc state",
    "north carolina st":  "nc state",
    # MLB
    "arizona diamondbacks":   "diamondbacks",
    "d-backs":                "diamondbacks",
    "chi white sox":          "white sox",
    "chicago white sox":      "white sox",
    "chi cubs":               "cubs",
    "chicago cubs":           "cubs",
    "la dodgers":             "dodgers",
    "los angeles dodgers":    "dodgers",
    "la angels":              "angels",
    "los angeles angels":     "angels",
    "ny yankees":             "yankees",
    "new york yankees":       "yankees",
    "ny mets":                "mets",
    "new york mets":          "mets",
    "sf giants":              "giants",
    "san francisco giants":   "giants",
    "sd padres":              "padres",
    "san diego padres":       "padres",
    "st. louis cardinals":    "cardinals",
    "st louis cardinals":     "cardinals",
    "tb rays":                "rays",
    "tampa bay rays":         "rays",
    "kc royals":              "royals",
    "kansas city royals":     "royals",
    # NBA
    "la lakers":              "lakers",
    "los angeles lakers":     "lakers",
    "la clippers":            "clippers",
    "los angeles clippers":   "clippers",
    "ny knicks":              "knicks",
    "new york knicks":        "knicks",
    "brooklyn nets":          "nets",
    "golden state warriors":  "warriors",
    "okc thunder":            "thunder",
    "oklahoma city thunder":  "thunder",
    "sa spurs":               "spurs",
    "san antonio spurs":      "spurs",
    "portland trail blazers": "trail blazers",
    "minnesota timberwolves": "timberwolves",
    "milwaukee bucks":        "bucks",
    "indiana pacers":         "pacers",
    "philadelphia 76ers":     "76ers",
    "phoenix suns":           "suns",
}

# Suffixes to strip when normalizing ESPN display names
_SUFFIXES = [
    "blue devils", "tar heels", "wolfpack", "demon deacons", "fighting irish",
    "hoyas", "bulldogs", "wildcats", "huskies", "gators", "longhorns",
    "buckeyes", "wolverines", "spartans", "hawkeyes", "boilermakers",
    "crimson tide", "tigers", "bears", "eagles", "cardinals", "knights",
    "trojans", "bruins", "jayhawks", "razorbacks", "commodores", "volunteers",
    "aggies", "cowboys", "cyclones", "hawkeyes", "cornhuskers", "terrapins",
    "badgers", "illini", "hoosiers", "nittany lions", "mountaineers",
    "seminoles", "gators", "orange", "panthers", "golden eagles", "rams",
    "flyers", "friars", "musketeers", "bearcats", "retrievers", "toreros",
]


def _normalize(name: str) -> str:
    """Lowercase, strip mascot suffixes, collapse whitespace."""
    n = name.lower().strip()
    for suffix in _SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
            break
    n = re.sub(r"\s+", " ", n)
    return n


def build_index(espn_names: dict[str, str]) -> dict[str, str]:
    """
    Build a lookup from normalized ESPN display name -> ESPN team ID.
    espn_names: {team_id: display_name} from engine.names
    """
    return {_normalize(name): tid for tid, name in espn_names.items()}


def match_team(
    odds_name: str,
    index: dict[str, str],   # normalized_name -> espn_id
    cutoff: float = 0.6,
) -> str | None:
    """
    Match an Odds API team name to an ESPN team ID.
    Returns None if no confident match found.
    """
    key = odds_name.lower().strip()

    # Check manual overrides
    for override_key, target in _OVERRIDES.items():
        if override_key in key or key in override_key:
            # Find the ESPN ID whose normalized name contains the target
            for norm, tid in index.items():
                if target in norm:
                    return tid

    # Normalize and try exact match
    norm = _normalize(odds_name)
    if norm in index:
        return index[norm]

    # Fuzzy match
    candidates = list(index.keys())
    matches = difflib.get_close_matches(norm, candidates, n=1, cutoff=cutoff)
    if matches:
        return index[matches[0]]

    return None


def match_game(
    home_odds_name: str,
    away_odds_name: str,
    index: dict[str, str],
) -> tuple[str | None, str | None]:
    """Match both teams in an Odds API event to ESPN IDs."""
    return (
        match_team(home_odds_name, index),
        match_team(away_odds_name, index),
    )
