"""
Surface-specific Elo ratings for individual tennis players.

Each player carries four ratings: hard, grass, clay, and overall.
Match updates apply fully to the surface played and partially to overall.

Usage:
    from src.ratings.tennis import TennisEloEngine

    engine = TennisEloEngine()
    engine.process_match("Carlos Alcaraz", "Jannik Sinner", surface="grass")
    p = engine.win_prob("Carlos Alcaraz", "Novak Djokovic", surface="grass")
"""

from __future__ import annotations

from dataclasses import dataclass, field

SURFACES = ("hard", "grass", "clay")

# Grand Slam K is higher — fewer matches, higher stakes
K_DEFAULT   = 32.0
K_SLAM      = 40.0
SCALE       = 400.0
INITIAL     = 1500.0
OVERALL_MIX = 0.30   # fraction of surface update that bleeds into overall rating


@dataclass
class TennisEloEngine:
    k: float = K_DEFAULT
    k_slam: float = K_SLAM
    scale: float = SCALE
    initial: float = INITIAL

    # {player_name: {"hard": float, "grass": float, "clay": float, "overall": float}}
    ratings: dict[str, dict[str, float]] = field(default_factory=dict)
    # {player_name: match_count} across all surfaces
    match_counts: dict[str, int] = field(default_factory=dict)

    def _get(self, player: str, surface: str) -> float:
        return self.ratings.get(player, {}).get(surface, self.initial)

    def _init_player(self, player: str) -> None:
        if player not in self.ratings:
            self.ratings[player] = {s: self.initial for s in SURFACES}
            self.ratings[player]["overall"] = self.initial
            self.match_counts[player] = 0

    def win_prob(self, player_a: str, player_b: str, surface: str = "hard") -> float:
        """P(A beats B) on given surface using logistic Elo formula."""
        ra = self._get(player_a, surface)
        rb = self._get(player_b, surface)
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / self.scale))

    def process_match(
        self,
        winner: str,
        loser: str,
        surface: str = "hard",
        grand_slam: bool = False,
    ) -> None:
        self._init_player(winner)
        self._init_player(loser)

        k = self.k_slam if grand_slam else self.k
        p_win = self.win_prob(winner, loser, surface)

        delta_surface = k * (1.0 - p_win)
        delta_overall = delta_surface * OVERALL_MIX

        self.ratings[winner][surface]   += delta_surface
        self.ratings[loser][surface]    -= delta_surface
        self.ratings[winner]["overall"] += delta_overall
        self.ratings[loser]["overall"]  -= delta_overall

        self.match_counts[winner] = self.match_counts.get(winner, 0) + 1
        self.match_counts[loser]  = self.match_counts.get(loser, 0) + 1

    def process_matches(self, matches: list[dict]) -> None:
        """
        Process a list of match dicts.
        Each dict: {winner, loser, surface, grand_slam (optional)}
        """
        for m in matches:
            self.process_match(
                m["winner"],
                m["loser"],
                surface=m.get("surface", "hard"),
                grand_slam=m.get("grand_slam", False),
            )

    def top_players(self, surface: str = "overall", n: int = 25) -> list[tuple[str, float]]:
        """Return top-n players sorted by rating on given surface."""
        ranked = [
            (name, ratings.get(surface, self.initial))
            for name, ratings in self.ratings.items()
            if self.match_counts.get(name, 0) >= 3   # minimum matches to appear
        ]
        return sorted(ranked, key=lambda x: x[1], reverse=True)[:n]

    def match_count(self, player: str) -> int:
        return self.match_counts.get(player, 0)
