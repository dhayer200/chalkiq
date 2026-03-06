"""
Player Elo rating engine.

Mirrors the team EloEngine structure exactly:
  - Each player starts at 1500
  - After each game, their Game Score is normalized to a [0,1] outcome
  - Rating updates: R' = R + K * (outcome - expected)
  - Expected outcome uses the standard Elo formula vs the league average (1500)

K=16 (lower than team K=24 because individual games are noisier).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass, field

from src.players.gamescores import gmsc_to_outcome

INITIAL_RATING: float = 1500.0
LEAGUE_RATING:  float = 1500.0
K:              float = 16.0
MIN_MINUTES:    int   = 5     # ignore players with fewer minutes


def _expected(r_player: float, r_league: float = LEAGUE_RATING) -> float:
    """Standard Elo expected outcome for player vs league average."""
    return 1.0 / (1.0 + 10.0 ** ((r_league - r_player) / 400.0))


@dataclass
class PlayerEloEngine:
    k:       float = K
    initial: float = INITIAL_RATING

    # internal state
    ratings:   dict[str, float] = field(default_factory=dict, repr=False)
    names:     dict[str, str]   = field(default_factory=dict, repr=False)
    teams:     dict[str, str]   = field(default_factory=dict, repr=False)
    positions: dict[str, str]   = field(default_factory=dict, repr=False)
    history:   list[dict]       = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def rating(self, player_id: str) -> float:
        return self.ratings.get(player_id, self.initial)

    def process_game(self, game: dict) -> None:
        """Process one boxscore game dict (from fetch_boxscores output)."""
        game_date = game.get("game_date", "")

        for player in game.get("players", []):
            pid  = player.get("player_id", "")
            gmsc = player.get("game_score", 0.0)
            min_ = player.get("min", 0)

            if not pid or min_ < MIN_MINUTES:
                continue

            self.names[pid]     = player.get("name", "")
            self.teams[pid]     = player.get("team_id", "")
            self.positions[pid] = player.get("position", "")

            r        = self.rating(pid)
            outcome  = gmsc_to_outcome(gmsc)
            expected = _expected(r)

            self.ratings[pid] = r + self.k * (outcome - expected)

            self.history.append({
                "player_id":   pid,
                "name":        self.names[pid],
                "team_id":     player.get("team_id", ""),
                "game_date":   game_date,
                "game_score":  gmsc,
                "rating_pre":  round(r, 2),
                "rating_post": round(self.ratings[pid], 2),
                "min":         min_,
                "pts":         player.get("pts", 0),
                "reb":         player.get("reb", 0),
                "ast":         player.get("ast", 0),
                "stl":         player.get("stl", 0),
                "blk":         player.get("blk", 0),
                "to":          player.get("to", 0),
                "starter":     player.get("starter", False),
            })

    def process_games(self, games: list[dict]) -> None:
        """Process a list of boxscore game dicts in chronological order."""
        for g in sorted(games, key=lambda x: x.get("game_date", "")):
            self.process_game(g)

    def top_players(
        self,
        n: int = 100,
        min_games: int = 5,
        team_id: str | None = None,
        position: str | None = None,
    ) -> list[tuple[str, str, float, str, str]]:
        """
        Return top N players by current rating.
        Returns list of (player_id, name, rating, team_id, position).
        """
        game_counts = Counter(r["player_id"] for r in self.history)
        players = [
            (pid, self.names.get(pid, ""), self.rating(pid),
             self.teams.get(pid, ""), self.positions.get(pid, ""))
            for pid in self.ratings
            if game_counts[pid] >= min_games
            and (team_id is None or self.teams.get(pid) == team_id)
            and (position is None or self.positions.get(pid) == position)
        ]
        return sorted(players, key=lambda x: -x[2])[:n]

    def player_history(self, player_id: str) -> list[dict]:
        """Game-by-game history for a player, chronological."""
        return [r for r in self.history if r["player_id"] == player_id]

    def player_game_counts(self) -> dict[str, int]:
        counts: dict[str, int] = Counter(r["player_id"] for r in self.history)  # type: ignore[assignment]
        return counts

    def season_trajectory(
        self, player_id: str, window: int = 5
    ) -> list[dict]:
        """
        Rolling average Game Score for a player over the season.
        Returns list of {date, game_score, rolling_avg, pts, reb, ast}.
        """
        hist = self.player_history(player_id)
        if not hist:
            return []

        result = []
        for i, rec in enumerate(hist):
            window_scores = [h["game_score"] for h in hist[max(0, i - window + 1): i + 1]]
            result.append({
                "date":         rec["game_date"],
                "game_score":   rec["game_score"],
                "rolling_avg":  round(sum(window_scores) / len(window_scores), 2),
                "pts":          rec["pts"],
                "reb":          rec["reb"],
                "ast":          rec["ast"],
                "rating":       rec["rating_post"],
                "min":          rec["min"],
            })
        return result

    def prop_edge(
        self,
        player_id: str,
        stat: str,
        line: float,
        last_n: int = 10,
        breakeven: float = 0.5238,
    ) -> dict:
        """
        Compute edge on an over/under prop bet using the player's recent stat distribution.

        stat: 'pts', 'reb', 'ast', 'stl', 'blk'
        line: the prop line from the book
        breakeven: 0.5238 = breakeven at -110 juice

        Returns dict with projection, std_dev, p_over, edge, and raw history.
        """
        hist = self.player_history(player_id)[-last_n:]
        if not hist:
            return {"error": "no history"}

        values = [h.get(stat, 0) for h in hist]
        mean   = sum(values) / len(values)

        if len(values) < 2:
            std = 3.0  # default when not enough data
        else:
            std = statistics.stdev(values)
            std = max(std, 1.0)  # floor to avoid division issues

        # P(stat > line) using normal approximation with continuity correction
        z      = (line + 0.5 - mean) / std
        p_over = 1.0 - _normal_cdf(z)
        edge   = p_over - breakeven

        return {
            "player_id":  player_id,
            "stat":       stat,
            "line":       line,
            "mean":       round(mean, 1),
            "std_dev":    round(std, 1),
            "p_over":     round(p_over, 4),
            "edge":       round(edge, 4),
            "n_games":    len(values),
            "last_values": values,
        }

    def injury_elo_impact(self, player_id: str) -> float:
        """
        Estimate Elo points the player contributes to their team's strength.

        impact = (player_rating - 1500) * minutes_share * position_weight

        A player rated 1700 who plays 35/200 team minutes contributes:
        (1700 - 1500) * (35/200) * pos_weight = 200 * 0.175 * 1.0 = 35 Elo pts
        """
        if player_id not in self.ratings:
            return 0.0

        player_r = self.rating(player_id)
        excess   = player_r - INITIAL_RATING

        hist = self.player_history(player_id)
        if not hist:
            return 0.0

        avg_min   = sum(r["min"] for r in hist) / len(hist)
        min_share = avg_min / 200.0

        pos = self.positions.get(player_id, "")
        pos_weight = {"C": 1.1, "F": 1.0, "G": 0.95}.get(pos, 1.0)

        return round(excess * min_share * pos_weight, 1)

    def team_injury_report(
        self, team_id: str, team_elo: float, min_games: int = 3
    ) -> list[dict]:
        """
        For each player on a team, return their injury impact and adjusted team Elo.
        Sorted by impact descending.
        """
        game_counts = self.player_game_counts()
        players = [
            pid for pid in self.ratings
            if self.teams.get(pid) == team_id
            and game_counts.get(pid, 0) >= min_games
        ]

        rows = []
        for pid in players:
            impact = self.injury_elo_impact(pid)
            hist   = self.player_history(pid)
            avg_min = sum(r["min"] for r in hist) / len(hist) if hist else 0
            rows.append({
                "player_id":    pid,
                "name":         self.names.get(pid, "?"),
                "position":     self.positions.get(pid, ""),
                "rating":       round(self.rating(pid), 1),
                "avg_min":      round(avg_min, 1),
                "games":        game_counts.get(pid, 0),
                "elo_impact":   impact,
                "adj_team_elo": round(team_elo - impact, 1),
            })

        return sorted(rows, key=lambda x: -x["elo_impact"])


# ── Helpers ────────────────────────────────────────────────────────────────── #

def _normal_cdf(z: float) -> float:
    """Standard normal CDF using math.erf."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def load_boxscores(cache_dir: str) -> list[dict]:
    """Load all cached boxscore JSON files from a directory."""
    import json
    from pathlib import Path

    path  = Path(cache_dir)
    games = []
    if not path.exists():
        return games

    for f in sorted(path.glob("*.json")):
        try:
            with open(f) as fp:
                games.append(json.load(fp))
        except Exception:
            continue
    return games
