"""
Closing Line Value (CLV) and Line Movement Awareness (LMA).

CLV tells you: did you beat the closing line?
LMA tells you: which direction is sharp money moving?

A model that consistently produces positive CLV is finding real edges.
A model that wins games but never beats closing is getting lucky.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── CLV ───────────────────────────────────────────────────────────────────── #

@dataclass
class CLVResult:
    game_id:           str
    home_team:         str
    away_team:         str
    bookmaker:         str
    model_prob_home:   float   # our Elo probability
    opening_prob_home: float   # vig-removed opening line probability
    closing_prob_home: float   # vig-removed closing line probability
    clv_vs_opening:    float   # model_prob - opening_prob
    clv_vs_closing:    float   # model_prob - closing_prob  ← the real CLV
    home_won:          bool | None = None

    @property
    def beat_closing(self) -> bool:
        """True if our model was better than the closing line."""
        return self.clv_vs_closing > 0

    @property
    def summary(self) -> str:
        direction = "+" if self.clv_vs_closing >= 0 else ""
        result = ""
        if self.home_won is not None:
            result = f"  outcome={'W' if self.home_won else 'L'}"
        return (
            f"{self.home_team} vs {self.away_team} [{self.bookmaker}]  "
            f"model={self.model_prob_home:.1%}  "
            f"open={self.opening_prob_home:.1%}  "
            f"close={self.closing_prob_home:.1%}  "
            f"CLV={direction}{self.clv_vs_closing:+.1%}"
            f"{result}"
        )


def compute_clv(
    model_prob_home: float,
    opening: dict,   # snapshot dict with home_prob, away_prob
    closing: dict,   # snapshot dict with home_prob, away_prob
    game_id: str = "",
    home_team: str = "",
    away_team: str = "",
    bookmaker: str = "",
    home_won: bool | None = None,
) -> CLVResult:
    """
    Compute CLV for a single game.

    model_prob_home: our Elo win probability for the home team.
    opening/closing: dicts from store snapshots (already vig-removed home_prob).
    """
    return CLVResult(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        bookmaker=bookmaker,
        model_prob_home=model_prob_home,
        opening_prob_home=opening["home_prob"],
        closing_prob_home=closing["home_prob"],
        clv_vs_opening=model_prob_home - opening["home_prob"],
        clv_vs_closing=model_prob_home - closing["home_prob"],
        home_won=home_won,
    )


def clv_summary(results: list[CLVResult]) -> dict:
    """Aggregate CLV stats across a batch of games."""
    if not results:
        return {}

    n            = len(results)
    beat_closing = sum(1 for r in results if r.beat_closing)
    avg_clv      = sum(r.clv_vs_closing for r in results) / n
    avg_vs_open  = sum(r.clv_vs_opening for r in results) / n

    completed = [r for r in results if r.home_won is not None]
    win_rate  = sum(1 for r in completed if r.home_won) / len(completed) if completed else None

    return {
        "n_games":          n,
        "beat_closing_pct": beat_closing / n,
        "avg_clv":          avg_clv,
        "avg_clv_vs_open":  avg_vs_open,
        "win_rate":         win_rate,
        "n_completed":      len(completed),
    }


# ── LMA ───────────────────────────────────────────────────────────────────── #

@dataclass
class LineMove:
    game_id:    str
    home_team:  str
    away_team:  str
    bookmaker:  str
    from_ml:    int     # American ML before move
    to_ml:      int     # American ML after move
    from_prob:  float   # vig-removed prob before
    to_prob:    float   # vig-removed prob after
    move_size:  float   # to_prob - from_prob (positive = home got more likely)
    fetched_at: str
    sharp:      bool    # True if move is likely sharp action (see classify_move)

    @property
    def direction(self) -> str:
        return "home" if self.move_size > 0 else "away"

    @property
    def summary(self) -> str:
        flag = " [SHARP]" if self.sharp else ""
        return (
            f"{self.home_team} vs {self.away_team}  "
            f"home ML: {self.from_ml:+d} -> {self.to_ml:+d}  "
            f"({self.from_prob:.1%} -> {self.to_prob:.1%})  "
            f"move={self.move_size:+.1%}{flag}"
        )


def detect_line_moves(
    history: list[dict],
    min_move: float = 0.02,
) -> list[LineMove]:
    """
    Detect meaningful line moves across a sequence of snapshots for one game.

    history: list of snapshot dicts in chronological order (same game+bookmaker).
    min_move: minimum probability shift to count as a line move (default 2%).

    Returns list of LineMove events.
    """
    moves: list[LineMove] = []
    for i in range(1, len(history)):
        prev = history[i - 1]
        curr = history[i]
        shift = curr["home_prob"] - prev["home_prob"]
        if abs(shift) >= min_move:
            moves.append(LineMove(
                game_id=curr.get("game_id", ""),
                home_team=curr.get("home_team", ""),
                away_team=curr.get("away_team", ""),
                bookmaker=curr.get("bookmaker", ""),
                from_ml=prev["home_ml"],
                to_ml=curr["home_ml"],
                from_prob=prev["home_prob"],
                to_prob=curr["home_prob"],
                move_size=shift,
                fetched_at=curr.get("fetched_at", ""),
                sharp=classify_move(shift),
            ))
    return moves


def classify_move(shift: float, threshold: float = 0.04) -> bool:
    """
    Heuristic: a move >= 4% in implied probability is likely sharp action.
    Public money causes smaller drifts; sharp bets cause sudden, large moves.
    """
    return abs(shift) >= threshold


def reverse_line_movement(
    line_move: LineMove,
    public_pct_home: float | None,
) -> bool:
    """
    True if there's reverse line movement: public bets heavily on home
    but the line moves away from home (toward away). This signals sharp fade.

    public_pct_home: fraction of bets on home (0-1). None if unknown.
    """
    if public_pct_home is None:
        return False
    # Public on home (>60%), but line moved away from home
    if public_pct_home > 0.60 and line_move.move_size < -0.02:
        return True
    # Public on away (>60%), but line moved toward home
    if public_pct_home < 0.40 and line_move.move_size > 0.02:
        return True
    return False
