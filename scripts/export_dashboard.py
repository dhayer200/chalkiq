#!/usr/bin/env python3
"""
Export pre-computed dashboard data to a single JSON file.

Usage:
    python scripts/export_dashboard.py
    python scripts/export_dashboard.py --division mens
    python scripts/export_dashboard.py --division nba --division mlb

Outputs:
    web/assets/dashboard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.ratings.elo import EloEngine, SPORT_CONFIGS
from src.utils.data import fetch_season
from src.utils.metrics import evaluate
from src.utils.rating_history import build_rating_histories, market_movers, ncaa361_spread


SEASONS: dict[str, dict[int, tuple[date, date]]] = {
    "mens": {2026: (date(2025, 11, 4), date.today())},
    "womens": {2026: (date(2025, 11, 4), date.today())},
    "mlb": {2026: (date(2026, 2, 20), date.today())},
    "nba": {2026: (date(2025, 10, 21), date.today())},
}

CACHE_DIRS: dict[str, str] = {
    "mens": "data/raw/mens",
    "womens": "data/raw/womens",
    "mlb": "data/raw/mlb",
    "nba": "data/raw/nba",
}

DIVISION_LABELS: dict[str, str] = {
    "mens": "Men's CBB",
    "womens": "Women's CBB",
    "mlb": "MLB",
    "nba": "NBA",
}


def build_division(division: str) -> dict:
    """Build all dashboard data for a single division."""
    print(f"\n  [{division}] Loading games...")
    season_info = SEASONS[division]
    season = max(season_info.keys())
    start, end = season_info[season]
    cache_dir = CACHE_DIRS[division]

    games = fetch_season(start, end, cache_dir=cache_dir, division=division, verbose=False)
    print(f"  [{division}] {len(games):,} games loaded")

    if not games:
        print(f"  [{division}] No games found, skipping")
        return {}

    # Build Elo engine
    cfg = SPORT_CONFIGS.get(division, SPORT_CONFIGS["mens"])
    engine = EloEngine(
        k=cfg["k"],
        home_advantage=cfg["home_advantage"],
        scale=cfg.get("scale", 300.0),
        season_regress=cfg["season_regress"],
        tempo_adjust=cfg["tempo_adjust"],
        season_boundary=cfg.get("season_boundary", "ncaa"),
    )
    engine.process_games(games)
    print(f"  [{division}] Elo engine processed {len(engine.history):,} games")

    # Rankings (top 100)
    rankings_raw = engine.rankings()[:100]
    rankings = [
        {"rank": i + 1, "name": name, "elo": round(elo, 1), "team_id": tid}
        for i, (tid, name, elo) in enumerate(rankings_raw)
    ]

    # Metrics
    metrics = evaluate(engine.history)
    # Round floats for compact JSON
    if metrics:
        for key in ("log_loss", "brier_score", "baseline_log_loss", "baseline_brier_score"):
            if key in metrics:
                metrics[key] = round(metrics[key], 6)
        if "calibration" in metrics:
            for bucket in metrics["calibration"]:
                for k_cal in ("bin_mid", "predicted_avg", "observed"):
                    if k_cal in bucket:
                        bucket[k_cal] = round(bucket[k_cal], 4)

    # Rating histories and movers
    histories = build_rating_histories(engine.history)
    gainers, losers = market_movers(histories, engine.names, window_days=30, top_n=10)
    movers = {
        "gainers": [
            {"name": m["name"], "current": m["current"], "change": m["change"]}
            for m in gainers
        ],
        "losers": [
            {"name": m["name"], "current": m["current"], "change": m["change"]}
            for m in losers
        ],
    }

    # Recent games (last 50)
    recent_history = engine.history[-50:]
    recent_games = []
    for g in recent_history:
        prob = g["pregame_prob_home"]
        home_won = g["home_score"] > g["away_score"]
        if prob > 0.5:
            correct = home_won
        elif prob < 0.5:
            correct = not home_won
        else:
            correct = True  # 50/50 is always "correct"
        recent_games.append({
            "date": g["date"],
            "home_name": g["home_name"],
            "away_name": g["away_name"],
            "home_score": g["home_score"],
            "away_score": g["away_score"],
            "pregame_prob_home": round(prob, 4),
            "correct": correct,
        })

    # Rating spread (volatility over time)
    spread_raw = ncaa361_spread(histories)
    rating_spread = [{"date": d, "spread": s} for d, s in spread_raw]

    # Cumulative accuracy (emit every 10 games)
    sorted_history = sorted(engine.history, key=lambda g: g["date"] or "")
    cumulative_accuracy = []
    correct_count = 0
    total_count = 0
    for g in sorted_history:
        prob = g["pregame_prob_home"]
        home_won = g["home_score"] > g["away_score"]
        if prob > 0.5:
            is_correct = home_won
        elif prob < 0.5:
            is_correct = not home_won
        else:
            is_correct = True
        if is_correct:
            correct_count += 1
        total_count += 1
        if total_count % 10 == 0:
            cumulative_accuracy.append({
                "date": g["date"],
                "n": total_count,
                "accuracy": round(correct_count / total_count, 4),
            })

    # Elo trajectories for top 30 teams (sampled for reasonable JSON size)
    top_ids = {r["team_id"] for r in rankings[:30]}
    elo_trajectories = {}
    for tid in top_ids:
        pts = histories.get(tid, [])
        if not pts:
            continue
        # Sample: keep every Nth point to stay under ~50 pts per team
        step = max(1, len(pts) // 50)
        sampled = pts[::step]
        if pts[-1] not in sampled:
            sampled.append(pts[-1])
        name = engine.names.get(tid, tid)
        elo_trajectories[tid] = {
            "name": name,
            "elo": round(engine.ratings.get(tid, 1500), 1),
            "points": [{"date": d, "elo": round(r, 1)} for d, r in sampled],
        }

    # Prediction distribution (binned win probabilities)
    pred_dist = [0] * 10  # 10 bins: 0-10%, 10-20%, ..., 90-100%
    for g in sorted_history:
        p = g["pregame_prob_home"]
        bucket = min(int(p * 10), 9)
        pred_dist[bucket] += 1

    # Accuracy by confidence bucket
    conf_buckets: dict[str, dict] = {}
    for g in sorted_history:
        p = g["pregame_prob_home"]
        conf = max(p, 1 - p)  # symmetrize: 0.5 to 1.0
        bucket_idx = min(int(conf * 10), 9)
        label = f"{bucket_idx * 10}-{bucket_idx * 10 + 10}%"
        if label not in conf_buckets:
            conf_buckets[label] = {"correct": 0, "total": 0}
        home_won = g["home_score"] > g["away_score"]
        if p > 0.5:
            is_correct = home_won
        elif p < 0.5:
            is_correct = not home_won
        else:
            is_correct = True
        conf_buckets[label]["total"] += 1
        if is_correct:
            conf_buckets[label]["correct"] += 1

    accuracy_by_conf = []
    for label, counts in sorted(conf_buckets.items()):
        if counts["total"] >= 5:
            accuracy_by_conf.append({
                "bucket": label,
                "accuracy": round(counts["correct"] / counts["total"], 4),
                "n": counts["total"],
            })

    print(f"  [{division}] Rankings: {len(rankings)}, Recent: {len(recent_games)}, "
          f"Spread pts: {len(rating_spread)}, Accuracy pts: {len(cumulative_accuracy)}, "
          f"Trajectories: {len(elo_trajectories)}")

    return {
        "label": DIVISION_LABELS.get(division, division),
        "rankings": rankings,
        "metrics": metrics,
        "movers": movers,
        "recent_games": recent_games,
        "rating_spread": rating_spread,
        "cumulative_accuracy": cumulative_accuracy,
        "elo_trajectories": elo_trajectories,
        "prediction_distribution": pred_dist,
        "accuracy_by_confidence": accuracy_by_conf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export dashboard data to JSON",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--division",
        action="append",
        choices=["mens", "womens", "mlb", "nba"],
        default=None,
        help="Division(s) to export. Omit for all.",
    )
    args = parser.parse_args()

    divisions = args.division or ["mens", "womens", "mlb", "nba"]

    print("  Exporting dashboard data...")
    output: dict = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "divisions": {},
    }

    for div in divisions:
        result = build_division(div)
        if result:
            output["divisions"][div] = result

    out_path = ROOT / "web" / "assets" / "dashboard.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=None)

    size_kb = out_path.stat().st_size / 1024
    print(f"\n  Wrote {out_path} ({size_kb:.0f} KB)")
    print(f"  Divisions: {', '.join(output['divisions'].keys())}")
    for div, data in output["divisions"].items():
        n_rank = len(data.get("rankings", []))
        n_games = data.get("metrics", {}).get("n_games", 0)
        print(f"    {div}: {n_rank} ranked teams, {n_games} games scored")


if __name__ == "__main__":
    main()
