"""
Backtesting engine.

Runs the Elo model over historical game data and simulates a betting strategy,
recording P&L, CLV proxies, and calibration data.

Since we don't have historical closing lines (that requires paid Odds API),
this backtests against standard book juice (-110 on both sides).

A team needs >52.4% win probability to have positive expected value at -110.
The backtest answers: "does our model find edges above that threshold?"

Usage:
    from src.backtest.engine import Backtester
    bt = Backtester(k=24, home_advantage=100)
    results = bt.run(games, min_edge=0.03, stake=100)
    bt.print_summary(results)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from src.ratings.elo import EloEngine


# Break-even win rate at standard -110 juice
BREAKEVEN_110 = 100 / (110 + 100)   # ≈ 0.5238


@dataclass
class BetRecord:
    game_date:      str
    home_name:      str
    away_name:      str
    bet_on_home:    bool       # True = we bet home, False = we bet away
    model_prob:     float      # our probability for the team we're betting
    edge:           float      # model_prob - breakeven (how much above juice)
    won:            bool       # did our bet win?
    pnl:            float      # profit/loss on this bet (stake units)
    pregame_home_prob: float   # for calibration


@dataclass
class BacktestResults:
    bets:          list[BetRecord] = field(default_factory=list)
    total_games:   int = 0
    bets_placed:   int = 0
    wins:          int = 0
    losses:        int = 0
    total_pnl:     float = 0.0
    roi:           float = 0.0
    win_rate:      float = 0.0
    avg_edge:      float = 0.0
    log_loss:      float = 0.0
    brier_score:   float = 0.0
    monthly:       dict[str, dict] = field(default_factory=dict)


class Backtester:
    def __init__(self, k: float = 24.0, home_advantage: float = 100.0, decay_half_life: float = 0.0):
        self.k    = k
        self.hca  = home_advantage
        self.decay = decay_half_life

    def run(
        self,
        games: list[dict],
        min_edge: float = 0.03,
        stake: float = 100.0,
        juice: float = -110,
        warmup_games: int = 50,
    ) -> BacktestResults:
        """
        Backtest the Elo model over a list of games.

        min_edge:     minimum edge over breakeven to place a bet (default 3%).
        stake:        flat dollar amount per bet.
        juice:        standard book juice (default -110).
        warmup_games: skip betting for the first N games (ratings not settled).

        Games must be dicts with keys: date, home_id, home_name, away_id,
        away_name, home_score, away_score, neutral.
        """
        engine = EloEngine(k=self.k, home_advantage=self.hca, decay_half_life=self.decay)
        sorted_games = sorted(games, key=lambda g: g["date"])

        breakeven = 100.0 / (abs(juice) + 100.0)
        payout    = 100.0 / abs(juice)   # win $payout per $1 risked at -110

        results = BacktestResults(total_games=len(sorted_games))
        log_losses: list[float] = []
        briers:     list[float] = []

        for i, g in enumerate(sorted_games):
            home_id   = g["home_id"]
            away_id   = g["away_id"]
            neutral   = g.get("neutral", False)
            home_won  = g["home_score"] > g["away_score"]

            # Get model's pregame probability BEFORE updating ratings
            p_home = engine.win_prob(home_id, away_id, neutral=neutral)

            # Update engine with result
            engine.names[home_id] = g.get("home_name", home_id)
            engine.names[away_id] = g.get("away_name", away_id)
            engine.update(
                home_id=home_id,
                away_id=away_id,
                home_score=g["home_score"],
                away_score=g["away_score"],
                neutral=neutral,
                date=g["date"],
            )

            # Scoring metrics (all games, after warmup)
            if i >= warmup_games:
                y = 1.0 if home_won else 0.0
                p_clipped = max(1e-7, min(1 - 1e-7, p_home))
                log_losses.append(-(y * math.log(p_clipped) + (1 - y) * math.log(1 - p_clipped)))
                briers.append((p_home - y) ** 2)

            # Betting decision (skip warmup period)
            if i < warmup_games:
                continue

            # Determine which side has edge
            edge_home = p_home - breakeven
            edge_away = (1 - p_home) - breakeven

            if edge_home >= min_edge:
                bet_home = True
                edge = edge_home
                model_prob = p_home
                won = home_won
            elif edge_away >= min_edge:
                bet_home = False
                edge = edge_away
                model_prob = 1 - p_home
                won = not home_won
            else:
                continue  # no edge, skip

            pnl = stake * payout if won else -stake

            record = BetRecord(
                game_date=g["date"],
                home_name=g.get("home_name", home_id),
                away_name=g.get("away_name", away_id),
                bet_on_home=bet_home,
                model_prob=model_prob,
                edge=edge,
                won=won,
                pnl=pnl,
                pregame_home_prob=p_home,
            )
            results.bets.append(record)

            # Monthly tracking
            month = g["date"][:7]
            if month not in results.monthly:
                results.monthly[month] = {"bets": 0, "wins": 0, "pnl": 0.0}
            results.monthly[month]["bets"] += 1
            results.monthly[month]["wins"] += int(won)
            results.monthly[month]["pnl"]  += pnl

        # Aggregate
        n = len(results.bets)
        if n:
            results.bets_placed = n
            results.wins        = sum(1 for b in results.bets if b.won)
            results.losses      = n - results.wins
            results.total_pnl   = sum(b.pnl for b in results.bets)
            results.roi         = results.total_pnl / (stake * n)
            results.win_rate    = results.wins / n
            results.avg_edge    = sum(b.edge for b in results.bets) / n

        if log_losses:
            results.log_loss    = sum(log_losses) / len(log_losses)
            results.brier_score = sum(briers) / len(briers)

        return results

    @staticmethod
    def print_summary(results: BacktestResults, stake: float = 100.0) -> None:
        n = results.bets_placed
        if not n:
            print("No bets placed.")
            return

        print(f"\n{'='*55}")
        print(f"  BACKTEST SUMMARY")
        print(f"{'='*55}")
        print(f"  Total games processed : {results.total_games:,}")
        print(f"  Bets placed           : {n:,}")
        print(f"  Win rate              : {results.win_rate:.1%}  (breakeven ~52.4%)")
        print(f"  Total P&L             : ${results.total_pnl:,.2f}")
        print(f"  ROI                   : {results.roi:+.2%}")
        print(f"  Avg edge per bet      : {results.avg_edge:+.1%}")
        print(f"  Log loss              : {results.log_loss:.4f}  (baseline 0.6931)")
        print(f"  Brier score           : {results.brier_score:.4f}  (baseline 0.2500)")
        print(f"\n  Monthly breakdown:")
        for month, m in sorted(results.monthly.items()):
            wr  = m["wins"] / m["bets"] if m["bets"] else 0
            roi = m["pnl"] / (stake * m["bets"]) if m["bets"] else 0
            print(f"    {month}  bets={m['bets']:3d}  wr={wr:.0%}  pnl=${m['pnl']:+,.0f}  roi={roi:+.1%}")
        print(f"{'='*55}\n")

    @staticmethod
    def cumulative_pnl(results: BacktestResults) -> list[tuple[str, float]]:
        """
        Returns [(game_date, cumulative_pnl)] for plotting.
        """
        running = 0.0
        curve   = []
        for b in results.bets:
            running += b.pnl
            curve.append((b.game_date, running))
        return curve

    @staticmethod
    def edge_buckets(results: BacktestResults) -> dict[str, dict]:
        """
        Group bets by edge bucket and compute win rate + ROI per bucket.
        Useful for validating that higher edge = higher win rate.
        """
        buckets: dict[str, list[BetRecord]] = {}
        for b in results.bets:
            key = f"{int(b.edge * 100 // 2) * 2}-{int(b.edge * 100 // 2) * 2 + 2}%"
            buckets.setdefault(key, []).append(b)

        return {
            k: {
                "n":       len(v),
                "win_rate": sum(1 for b in v if b.won) / len(v),
                "avg_edge": sum(b.edge for b in v) / len(v),
                "pnl":      sum(b.pnl for b in v),
            }
            for k, v in sorted(buckets.items())
        }
