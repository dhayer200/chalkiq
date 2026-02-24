"""
NCAA tournament bracket structure and S-curve seeding.

Assigns the top 64 teams (by Elo) to 4 regions of 16 using the
standard NCAA S-curve seeding pattern.

S-curve: seeds 1-4 fill one slot per region in order East/West/South/Midwest,
then seeds 5-8 fill in reverse Midwest/South/West/East, alternating snake-style.
Each group of 4 overall seeds maps to one regional seed level.
"""

REGIONS = ["East", "West", "South", "Midwest"]

# Order teams appear in the bracket (first-round matchup pairs):
# 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
BRACKET_SLOT_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

# Round labels used in tables
ROUND_LABELS = {
    1: "R64",
    2: "R32",
    3: "S16",
    4: "E8",
    5: "FF",
    6: "Champ",
}


def _scurve(overall_seed: int) -> tuple[int, int]:
    """Return (region_index 0-3, regional_seed 1-16) for an overall seed 1-64."""
    group = (overall_seed - 1) // 4      # 0 = seed-1 group, 1 = seed-2 group, ...
    pos   = (overall_seed - 1) % 4       # position within group (0-3)
    regional_seed = group + 1
    region_idx = pos if group % 2 == 0 else 3 - pos   # snake left-right-left
    return region_idx, regional_seed


def assign_seeds(rankings: list[tuple]) -> dict[str, dict[int, tuple]]:
    """
    Assign overall seeds 1-64 to regions using the NCAA S-curve.

    rankings: [(team_id, team_name, elo_rating), ...]  (sorted best-first)

    Returns:
        {
            "East":    {1: (tid, name, rating), 2: ..., ..., 16: ...},
            "West":    {...},
            "South":   {...},
            "Midwest": {...},
        }
    """
    top64 = rankings[:64]
    regions: dict[str, dict[int, tuple]] = {r: {} for r in REGIONS}

    for overall_seed, (tid, name, rating) in enumerate(top64, start=1):
        region_idx, regional_seed = _scurve(overall_seed)
        regions[REGIONS[region_idx]][regional_seed] = (tid, name, rating)

    return regions


def region_bracket_order(seed_map: dict[int, tuple]) -> list[str]:
    """
    Return team_ids in bracket slot order for one region's seed_map.
    This ordering is what the simulator expects — adjacent pairs play R1.

    slot order: 1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15
    """
    return [seed_map[s][0] for s in BRACKET_SLOT_ORDER if s in seed_map]


def final_four_order(regions: dict, adv_odds: dict) -> list[str]:
    """
    Return [East_champ_id, West_champ_id, South_champ_id, Midwest_champ_id]
    where each champ is the team with highest P(reach Final Four) in that region.
    """
    result = []
    for region in REGIONS:
        seed_map = regions[region]
        best_tid = max(
            (tid for _, (tid, _, _) in seed_map.items()),
            key=lambda t: adv_odds.get(t, {}).get(5, 0),
        )
        result.append(best_tid)
    return result
