"""
March Madness Quant — Milestone 0
==================================
Fetches the 2025-26 NCAA men's basketball season, builds Elo ratings,
evaluates forecast quality, and runs a bracket simulation.

Run:
    python main.py

Data is cached in data/raw/ so subsequent runs are instant.
"""

from datetime import date

from src.bracket.simulator import simulate_bracket
from src.predictions.pregame import matchup_prob
from src.ratings.elo import EloEngine
from src.utils.data import fetch_season
from src.utils.metrics import evaluate

# ── Configuration ────────────────────────────────────────────────────────────

SEASON_START = date(2025, 11, 4)
SEASON_END   = date(2026, 2, 23)
CACHE_DIR    = "data/raw"

ELO_K             = 24.0
HOME_ADVANTAGE    = 100.0   # Elo points granted to the home team
TOP_N             = 25      # teams to show in the rankings table
N_SIMS            = 10_000  # bracket simulation runs


# ── Helpers ──────────────────────────────────────────────────────────────────

def bar(p: float, width: int = 20) -> str:
    filled = round(p * width)
    return "█" * filled + "░" * (width - filled)

def fmt_pct(p: float) -> str:
    return f"{p * 100:5.1f}%"


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    # 1. Fetch data
    print("=" * 60)
    print("MARCH MADNESS QUANT — Milestone 0")
    print("=" * 60)
    print(f"\n[1/4] Fetching games {SEASON_START} → {SEASON_END} ...")
    games = fetch_season(SEASON_START, SEASON_END, cache_dir=CACHE_DIR)
    print(f"      {len(games)} completed games loaded.\n")

    # 2. Build Elo ratings
    print("[2/4] Building Elo ratings ...")
    engine = EloEngine(k=ELO_K, home_advantage=HOME_ADVANTAGE)
    engine.process_games(games)
    rankings = engine.rankings()
    print(f"      {len(rankings)} teams rated.\n")

    # 3. Evaluate
    print("[3/4] Evaluating forecast quality ...")
    metrics = evaluate(engine.history)
    print(f"      {metrics['n_games']} predictions evaluated.\n")

    # 4. Bracket simulation — use current top-64 as a stand-in
    print("[4/4] Simulating bracket (top-64 by Elo, neutral court) ...")
    top64_ids = [tid for tid, _, _ in rankings[:64]]
    champ_odds = simulate_bracket(
        seeded_teams=top64_ids,
        win_prob_fn=engine.win_prob,
        n_sims=N_SIMS,
        seed=42,
    )
    print(f"      {N_SIMS:,} simulations complete.\n")

    # ── Output ────────────────────────────────────────────────────────────────

    print("=" * 60)
    print(f"POWER RANKINGS — Top {TOP_N}  (season through {SEASON_END})")
    print("=" * 60)
    print(f"{'Rank':<5} {'Team':<30} {'Elo':>7}  {'Title %':>7}")
    print("-" * 60)
    for rank, (tid, name, rating) in enumerate(rankings[:TOP_N], start=1):
        odds = champ_odds.get(tid, 0.0)
        print(f"{rank:<5} {name:<30} {rating:>7.1f}  {fmt_pct(odds):>7}")

    print("\n" + "=" * 60)
    print("MODEL EVALUATION  (predictive backtest — pregame probabilities)")
    print("=" * 60)
    ll    = metrics["log_loss"]
    bs    = metrics["brier_score"]
    ll_b  = metrics["baseline_log_loss"]
    bs_b  = metrics["baseline_brier_score"]
    print(f"  Games evaluated : {metrics['n_games']:,}")
    print(f"  Log  Loss  — Elo : {ll:.4f}   Baseline (50/50): {ll_b:.4f}   "
          f"Δ {ll - ll_b:+.4f}")
    print(f"  Brier Score — Elo : {bs:.4f}   Baseline (50/50): {bs_b:.4f}   "
          f"Δ {bs - bs_b:+.4f}")

    print("\n" + "=" * 60)
    print("CALIBRATION — observed win rate vs predicted probability")
    print("=" * 60)
    print(f"{'Pred range':<14} {'Avg pred':>9} {'Actual %':>9} {'N games':>8}")
    print("-" * 44)
    for b in metrics["calibration"]:
        lo = b["bin_mid"] - 0.05
        hi = b["bin_mid"] + 0.05
        print(
            f"{lo:.0%}–{hi:.0%}      "
            f"{b['predicted_avg']:>8.1%} "
            f"{b['observed']:>8.1%} "
            f"{b['n']:>8,}"
        )

    print("\n" + "=" * 60)
    print("EXAMPLE MATCHUPS  (neutral court, from current Elo ratings)")
    print("=" * 60)
    # show top-5 vs next-5 in rankings
    top5  = [tid for tid, _, _ in rankings[:5]]
    next5 = [tid for tid, _, _ in rankings[5:10]]
    for a, b in zip(top5, next5):
        m = matchup_prob(engine, a, b)
        prob_bar = bar(m["prob_a"])
        print(f"  {m['team_a'][:22]:<22} vs {m['team_b'][:22]:<22}")
        print(f"  {fmt_pct(m['prob_a'])} {prob_bar} {fmt_pct(m['prob_b'])}")
        print(f"  (Elo: {m['rating_a']} vs {m['rating_b']}, diff = {m['rating_diff']:+.0f})")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
