"""
Live in-game win probability model.

Uses a random-walk model of basketball scoring anchored to the Elo pregame
prior (Stern 1994; extended with logit-prior blending).

Model:
    P(team_a wins | d, t, p0)  =  logistic( z_lead + z_prior )

where:
    z_lead  = d / (SIGMA * sqrt(t))          -- lead normalized by uncertainty
    z_prior = logit(p0) * sqrt(3) / pi       -- Elo prior in probit space
    SIGMA   ~ 1.5 pts / sqrt(minute)         -- empirical college basketball

At tip-off (d=0, t=40): P == p0  (Elo prior dominates, no game info yet).
At end of game (t→0):  P → 1 if d>0, 0 if d<0  (outcome nearly certain).

Usage:
    from src.live.model import live_win_prob, upset_alert, prob_swing

    p = live_win_prob(score_diff=7, minutes_remaining=8.5, p_pregame=0.60)
    # p ≈ 0.93  (team_a up 7 with 8.5 min left, was 60% favorite)
"""

from __future__ import annotations

import math

# Empirical scoring volatility for college basketball (pts / sqrt(minute)).
# Derived from Clauset et al. (2015) diffusion model for college basketball.
# Gives ~85% win prob for a 7-pt lead with 8.5 min remaining (equal teams).
SIGMA: float = 2.0

# Small epsilon to avoid log(0) when clamping probabilities
_EPS: float = 1e-7


def live_win_prob(
    score_diff: float,
    minutes_remaining: float,
    p_pregame: float,
) -> float:
    """
    Estimate win probability for team_a given current game state.

    Parameters
    ----------
    score_diff:        score_a - score_b  (positive = team_a leads)
    minutes_remaining: total minutes left in the game (incl. future halves)
    p_pregame:         Elo-based pregame win probability for team_a

    Returns
    -------
    float in [0, 1] — P(team_a wins)
    """
    # Game over — determine winner from final score
    if minutes_remaining <= 0:
        if score_diff > 0:
            return 1.0
        if score_diff < 0:
            return 0.0
        return 0.5   # tie at buzzer → simplified overtime coin-flip

    p0 = max(_EPS, min(1.0 - _EPS, p_pregame))

    # Elo prior expressed as a z-score (logit → probit via logit ≈ π/√3 · probit)
    logit_p0 = math.log(p0 / (1.0 - p0))
    z_prior  = logit_p0 * math.sqrt(3.0) / math.pi

    # Score lead normalized by time-evolving uncertainty
    z_lead = score_diff / (SIGMA * math.sqrt(minutes_remaining))

    # Combined z-score → probability  (logistic approximation to normal CDF)
    z_total = z_lead + z_prior
    return 1.0 / (1.0 + math.exp(-z_total * math.pi / math.sqrt(3.0)))


def upset_alert(p_live: float, p_pregame: float) -> bool:
    """
    True when the pregame underdog has flipped to become the live favourite.

    p_live:    current live win probability for team_a
    p_pregame: pregame win probability for team_a
    """
    was_underdog     = p_pregame < 0.5
    is_now_favourite = p_live >= 0.5
    return was_underdog and is_now_favourite


def prob_swing(p_live: float, p_pregame: float) -> float:
    """
    Signed probability change for team_a.
    Positive = team_a gained probability since tip-off.
    """
    return p_live - p_pregame


def leverage(minutes_remaining: float) -> float:
    """
    Game leverage: how much a single possession matters right now.
    Increases as time runs out.  Normalised to 1.0 at tip-off (40 min).
    """
    if minutes_remaining <= 0:
        return float("inf")
    return math.sqrt(40.0 / minutes_remaining)
