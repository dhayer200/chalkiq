"""
Monte Carlo tournament bracket simulator.

Simulates a single-elimination bracket N times, drawing each game outcome
from the Elo win probability. Bracket is padded to the next power of 2
with bye slots so any number of teams works.

Usage:
    from src.bracket.simulator import simulate_bracket
    odds = simulate_bracket(
        seeded_teams=["150", "248", ...],   # team IDs, best seed first
        win_prob_fn=engine.win_prob,         # fn(a, b) -> P(a beats b)
        n_sims=10_000,
        seed=42,
    )
    # odds = {"150": 0.31, "248": 0.18, ...}
"""

import random


def simulate_bracket(
    seeded_teams: list[str],
    win_prob_fn,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Simulate the bracket n_sims times. Returns {team_id: championship_probability}.

    seeded_teams: team IDs ordered from best (seed 1) to worst.
    win_prob_fn:  callable(team_a_id, team_b_id) -> float, assumed neutral site.
    """
    n = len(seeded_teams)
    bracket_size = 1
    while bracket_size < n:
        bracket_size *= 2

    rng = random.Random(seed)
    counts: dict[str, int] = {t: 0 for t in seeded_teams}

    for _ in range(n_sims):
        bracket = seeded_teams[:] + [None] * (bracket_size - n)

        while len(bracket) > 1:
            next_round = []
            for i in range(0, len(bracket), 2):
                a, b = bracket[i], bracket[i + 1]
                if a is None:
                    next_round.append(b)
                elif b is None:
                    next_round.append(a)
                else:
                    p = win_prob_fn(a, b)
                    next_round.append(a if rng.random() < p else b)
            bracket = next_round

        counts[bracket[0]] += 1

    return {t: counts[t] / n_sims for t in seeded_teams}


def round_advancement_odds(
    seeded_teams: list[str],
    win_prob_fn,
    n_sims: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[int, float]]:
    """
    Simulate the bracket n_sims times and track how far each team advances.

    Returns {team_id: {round_number: probability_of_reaching_that_round}}
    where round 1 = first round, round log2(bracket_size) = championship.
    """
    n = len(seeded_teams)
    bracket_size = 1
    while bracket_size < n:
        bracket_size *= 2
    n_rounds = bracket_size.bit_length() - 1

    rng = random.Random(seed)
    reached: dict[str, list[int]] = {t: [] for t in seeded_teams}

    for _ in range(n_sims):
        bracket = seeded_teams[:] + [None] * (bracket_size - n)
        round_num = 0

        while len(bracket) > 1:
            round_num += 1
            next_round = []
            for i in range(0, len(bracket), 2):
                a, b = bracket[i], bracket[i + 1]
                if a is None:
                    next_round.append(b)
                elif b is None:
                    next_round.append(a)
                else:
                    p = win_prob_fn(a, b)
                    winner = a if rng.random() < p else b
                    loser = b if winner == a else a
                    reached[loser].append(round_num - 1)
                    next_round.append(winner)
            bracket = next_round

        reached[bracket[0]].append(round_num)

    result: dict[str, dict[int, float]] = {}
    for t in seeded_teams:
        rounds_reached = reached[t]
        result[t] = {
            r: sum(1 for x in rounds_reached if x >= r) / n_sims
            for r in range(n_rounds + 1)
        }
    return result
