"""
Monte Carlo Grand Slam draw simulator.

Simulates a 128-player (or smaller) bracket by repeatedly running the draw
and computing per-player win probabilities for each round.

Usage:
    from src.bracket.tennis_draw import simulate_draw
    from src.ratings.tennis import TennisEloEngine

    engine = TennisEloEngine()
    # ... populate engine with match history ...

    draw = ["Carlos Alcaraz", "Jannik Sinner", ..., "qualifier"]  # 128 players in seed/draw order
    results = simulate_draw(draw, engine, surface="grass", n_sims=10_000)

    # results["Carlos Alcaraz"] = {"R128": 0.98, "R64": 0.87, ..., "win": 0.34}
"""

from __future__ import annotations

import random
from collections import defaultdict

from src.ratings.tennis import TennisEloEngine

ROUND_LABELS = ["R128", "R64", "R32", "R16", "QF", "SF", "F", "win"]


def _sim_match(player_a: str, player_b: str, engine: TennisEloEngine, surface: str) -> str:
    p = engine.win_prob(player_a, player_b, surface)
    return player_a if random.random() < p else player_b


def _sim_bracket(draw: list[str], engine: TennisEloEngine, surface: str) -> dict[str, list[str]]:
    """
    Simulate one complete bracket. Returns {player: [rounds_reached]}.
    draw must have length that is a power of 2.
    """
    rounds_reached: dict[str, list[str]] = defaultdict(list)
    n = len(draw)
    n_rounds = n.bit_length() - 1       # 128 players → 7 rounds
    labels = ROUND_LABELS[:n_rounds + 1]  # e.g. R128 ... win

    current_round = list(draw)
    for label in labels[:-1]:
        # Mark everyone still alive as having reached this round
        for player in current_round:
            rounds_reached[player].append(label)

        # Play all matches in this round
        next_round: list[str] = []
        for i in range(0, len(current_round), 2):
            winner = _sim_match(current_round[i], current_round[i + 1], engine, surface)
            next_round.append(winner)
        current_round = next_round

    # Champion
    champion = current_round[0]
    rounds_reached[champion].append("win")

    return rounds_reached


def simulate_draw(
    draw: list[str],
    engine: TennisEloEngine,
    surface: str = "hard",
    n_sims: int = 10_000,
) -> dict[str, dict[str, float]]:
    """
    Run n_sims bracket simulations and return round-by-round win probabilities.

    draw: ordered list of players (seed 1 in position 0, seed 2 in position
          len(draw)-1, as per Grand Slam seeding convention). Length must be
          a power of 2 (pad with "qualifier" for byes).
    surface: "hard", "grass", or "clay"

    Returns {player: {round_label: probability}}
    """
    n = len(draw)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"draw length must be a power of 2, got {n}")

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _ in range(n_sims):
        reached = _sim_bracket(draw, engine, surface)
        for player, rounds in reached.items():
            for r in rounds:
                counts[player][r] += 1

    return {
        player: {r: cnt / n_sims for r, cnt in round_counts.items()}
        for player, round_counts in counts.items()
    }


def top_contenders(
    results: dict[str, dict[str, float]],
    round_label: str = "win",
    n: int = 16,
) -> list[tuple[str, float]]:
    """Return top-n players by win probability for a given round."""
    probs = [
        (player, rounds.get(round_label, 0.0))
        for player, rounds in results.items()
    ]
    return sorted(probs, key=lambda x: x[1], reverse=True)[:n]
