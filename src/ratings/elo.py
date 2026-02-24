"""
Elo rating engine for NCAA basketball.

Key design choices:
  - All teams start at DEFAULT_RATING (1500).
  - Home-court advantage is modeled as a fixed Elo offset added to the
    home team's effective rating before computing win probability.
  - Ratings are updated after every game using the standard Elo formula:
        R' = R + K * (outcome - expected)
  - The engine records a full history of predictions + outcomes so that
    log loss and Brier score can be computed externally.

Usage:
    engine = EloEngine(k=24, home_advantage=100)
    engine.process_games(games)           # games from data.fetch_season
    print(engine.rankings()[:10])         # top 10
    p = engine.win_prob("150", "248")     # Duke vs Houston (neutral)
"""

from dataclasses import dataclass, field

DEFAULT_RATING = 1500.0
SCALE = 400.0          # Elo scale: a 400-point gap ≈ 90.9% win probability
HOME_ADVANTAGE = 100.0  # Elo points granted to the home team


@dataclass
class EloEngine:
    k: float = 24.0
    home_advantage: float = HOME_ADVANTAGE
    initial: float = DEFAULT_RATING

    # internal state — not constructor args
    ratings: dict[str, float] = field(default_factory=dict, repr=False)
    names: dict[str, str] = field(default_factory=dict, repr=False)
    history: list[dict] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def rating(self, team_id: str) -> float:
        """Current Elo rating for a team (defaults to initial if unseen)."""
        return self.ratings.get(team_id, self.initial)

    def win_prob(self, team_a: str, team_b: str, neutral: bool = True) -> float:
        """
        P(team_a beats team_b).

        neutral=True  → no home-court adjustment (tournament assumption).
        neutral=False → team_a is the home team; home_advantage is applied.
        """
        r_a = self.rating(team_a)
        r_b = self.rating(team_b)
        adj = 0.0 if neutral else self.home_advantage
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a - adj) / SCALE))

    def update(
        self,
        home_id: str,
        away_id: str,
        home_score: int,
        away_score: int,
        neutral: bool = False,
        date: str | None = None,
    ) -> float:
        """
        Process one finished game. Returns the pregame P(home wins).

        home_id / away_id: ESPN team ID strings
        """
        outcome = 1.0 if home_score > away_score else 0.0
        p_home = self.win_prob(home_id, away_id, neutral)

        r_home = self.rating(home_id)
        r_away = self.rating(away_id)

        self.ratings[home_id] = r_home + self.k * (outcome - p_home)
        self.ratings[away_id] = r_away + self.k * ((1.0 - outcome) - (1.0 - p_home))

        self.history.append(
            {
                "date": date,
                "home_id": home_id,
                "home_name": self.names.get(home_id, home_id),
                "away_id": away_id,
                "away_name": self.names.get(away_id, away_id),
                "home_score": home_score,
                "away_score": away_score,
                "neutral": neutral,
                "pregame_prob_home": p_home,
                "outcome": outcome,         # 1 = home won, 0 = away won
                "home_rating_after": self.ratings[home_id],
                "away_rating_after": self.ratings[away_id],
            }
        )
        return p_home

    def process_games(self, games: list[dict]) -> None:
        """
        Process a list of game dicts as returned by data.fetch_season.
        Games are sorted by date before processing.
        """
        for g in sorted(games, key=lambda x: x["date"]):
            # keep a name lookup table updated
            self.names[g["home_id"]] = g["home_name"]
            self.names[g["away_id"]] = g["away_name"]
            self.update(
                home_id=g["home_id"],
                away_id=g["away_id"],
                home_score=g["home_score"],
                away_score=g["away_score"],
                neutral=g["neutral"],
                date=g["date"],
            )

    def rankings(self) -> list[tuple[str, str, float]]:
        """
        Return [(team_id, team_name, rating)] sorted by rating descending.
        Only includes teams that have played at least one game.
        """
        return sorted(
            [(tid, self.names.get(tid, tid), r) for tid, r in self.ratings.items()],
            key=lambda x: x[2],
            reverse=True,
        )
