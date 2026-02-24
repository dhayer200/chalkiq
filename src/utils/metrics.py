"""
Proper scoring rules for evaluating probabilistic forecasts.

Both log loss and Brier score are "proper" — meaning a forecaster maximizes
their expected score by reporting their true beliefs, not by gaming the metric.

Lower is better for both.

Usage:
    from src.utils.metrics import evaluate
    results = evaluate(engine.history)
    print(results)
"""

import math
from collections import defaultdict


def log_loss(predictions: list[tuple[float, int]]) -> float:
    """
    Average log loss (cross-entropy) over a list of (prob, outcome) pairs.

    Formula per game:  −[y·log(p) + (1−y)·log(1−p)]
    Averaged over n games.

    Punishes overconfident wrong predictions very hard.
    Perfect model: 0. Coin-flip baseline: log(2) ≈ 0.693.
    """
    if not predictions:
        return float("nan")
    total = 0.0
    for p, y in predictions:
        p = max(1e-9, min(1.0 - 1e-9, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(predictions)


def brier_score(predictions: list[tuple[float, int]]) -> float:
    """
    Average Brier score (mean squared error on probabilities).

    Formula per game:  (p − y)²
    Averaged over n games.

    Less punishing than log loss on extreme mistakes.
    Perfect model: 0. Coin-flip baseline: 0.25.
    """
    if not predictions:
        return float("nan")
    return sum((p - y) ** 2 for p, y in predictions) / len(predictions)


def calibration_bins(
    predictions: list[tuple[float, int]], n_bins: int = 10
) -> list[dict]:
    """
    Group predictions into probability buckets and compute observed win rate.

    Returns a list of dicts:
        {"bin_mid": 0.25, "predicted_avg": 0.24, "observed": 0.26, "n": 142}

    A well-calibrated model has predicted_avg ≈ observed in every bin.
    """
    bins: dict[int, list] = defaultdict(list)
    for p, y in predictions:
        b = min(int(p * n_bins), n_bins - 1)
        bins[b].append((p, y))

    result = []
    for b in range(n_bins):
        items = bins.get(b, [])
        if not items:
            continue
        ps = [x[0] for x in items]
        ys = [x[1] for x in items]
        result.append(
            {
                "bin_mid": (b + 0.5) / n_bins,
                "predicted_avg": sum(ps) / len(ps),
                "observed": sum(ys) / len(ys),
                "n": len(items),
            }
        )
    return result


def evaluate(history: list[dict]) -> dict:
    """
    Compute all metrics from the EloEngine's history list.

    Returns a dict with:
        n_games, log_loss, brier_score,
        baseline_log_loss, baseline_brier_score,
        calibration (list of bin dicts)
    """
    if not history:
        return {}

    preds = [(g["pregame_prob_home"], int(g["outcome"])) for g in history]
    baseline = [(0.5, int(g["outcome"])) for g in history]

    return {
        "n_games": len(preds),
        "log_loss": log_loss(preds),
        "brier_score": brier_score(preds),
        "baseline_log_loss": log_loss(baseline),
        "baseline_brier_score": brier_score(baseline),
        "calibration": calibration_bins(preds),
    }
