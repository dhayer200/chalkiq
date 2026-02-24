"""
Pregame matchup probability from Elo ratings.

Thin wrapper around EloEngine.win_prob that adds convenience helpers:
  - matchup table for a list of teams
  - upset probability (P(lower-rated team wins))
"""

from src.ratings.elo import EloEngine


def matchup_prob(engine: EloEngine, team_a: str, team_b: str) -> dict:
    """
    Return a dict describing the pregame matchup (neutral court assumed).

    {
        "team_a": "Duke Blue Devils",
        "team_b": "Houston Cougars",
        "prob_a":  0.61,
        "prob_b":  0.39,
        "rating_a": 1623.4,
        "rating_b": 1581.2,
        "rating_diff": 42.2,
        "favorite": "Duke Blue Devils",
    }
    """
    p_a = engine.win_prob(team_a, team_b, neutral=True)
    name_a = engine.names.get(team_a, team_a)
    name_b = engine.names.get(team_b, team_b)
    r_a = engine.rating(team_a)
    r_b = engine.rating(team_b)
    return {
        "team_a_id": team_a,
        "team_b_id": team_b,
        "team_a": name_a,
        "team_b": name_b,
        "prob_a": round(p_a, 4),
        "prob_b": round(1.0 - p_a, 4),
        "rating_a": round(r_a, 1),
        "rating_b": round(r_b, 1),
        "rating_diff": round(r_a - r_b, 1),
        "favorite": name_a if p_a >= 0.5 else name_b,
    }


def matchup_table(engine: EloEngine, team_ids: list[str]) -> list[dict]:
    """
    Return matchup info for every pair in team_ids (n*(n-1)/2 rows).
    Sorted by absolute rating difference descending.
    """
    rows = []
    for i in range(len(team_ids)):
        for j in range(i + 1, len(team_ids)):
            rows.append(matchup_prob(engine, team_ids[i], team_ids[j]))
    return sorted(rows, key=lambda r: abs(r["rating_diff"]), reverse=True)
