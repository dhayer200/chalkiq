"""
Elo rating engine for sports betting models.

Supports NCAA basketball, NBA, and MLB with sport-specific tuning.

Key design choices:
  - All teams start at DEFAULT_RATING (1500).
  - Home-court/field advantage is modeled as a fixed Elo offset added to the
    home team's effective rating before computing win probability.
  - Ratings are updated after every game using the standard Elo formula:
        R' = R + K * mov_factor * (outcome - expected)
  - Margin of victory scaling (FiveThirtyEight method):
        mov_factor = ln(|margin| + 1) * 2.2 / (rating_gap * 0.001 + 2.2)
    The second term corrects for autocorrelation — blowouts against weak
    opponents shouldn't move ratings as much as blowouts against equals.
  - The engine records a full history of predictions + outcomes so that
    log loss and Brier score can be computed externally.

Usage:
    engine = EloEngine(k=24, home_advantage=100)
    engine.process_games(games)           # games from data.fetch_season
    print(engine.rankings()[:10])         # top 10
    p = engine.win_prob("150", "248")     # Duke vs Houston (neutral)

    # MLB with sport-specific tuning:
    cfg = SPORT_CONFIGS["mlb"]
    engine = EloEngine(**cfg)
"""

import math
from dataclasses import dataclass, field

DEFAULT_RATING = 1500.0
SCALE = 300.0          # Elo scale: a 300-point gap ≈ 90.9% win probability
                       # Calibrated from 1,300+ games with odds data —
                       # 300 minimizes log loss (0.5161) and produces +3.5% CLV vs closing lines.
                       # Standard Elo uses 400 but that underestimates favorites in NCAAB.
HOME_ADVANTAGE = 100.0  # Elo points granted to the home team

# Sport-specific Elo configurations
SPORT_CONFIGS = {
    "mens": {
        "k": 24.0,
        "home_advantage": 100.0,
        "scale": 300.0,
        "season_regress": 0.33,
        "tempo_adjust": True,
        "season_boundary": "ncaa",   # Nov-Apr academic year
    },
    "womens": {
        "k": 24.0,
        "home_advantage": 100.0,
        "scale": 300.0,
        "season_regress": 0.33,
        "tempo_adjust": True,
        "season_boundary": "ncaa",
    },
    "nba": {
        "k": 12.0,
        "home_advantage": 70.0,
        "scale": 350.0,          # NBA has more parity than NCAAB
        "season_regress": 0.25,
        "tempo_adjust": True,
        "season_boundary": "ncaa",   # NBA also runs Oct-Jun
    },
    "mlb": {
        "k": 6.0,               # 162 games/season → need small K to avoid over-reaction
        "home_advantage": 40.0,  # MLB home field ≈ 54% win rate
        "scale": 400.0,          # Standard Elo scale — MLB has highest parity
        "season_regress": 0.20,  # MLB rosters more stable than college
        "tempo_adjust": False,   # 9 innings standard — no pace normalization
        "season_boundary": "mlb", # Calendar year: season runs late Mar → Oct
    },
    "epl": {
        "k": 8.0,               # 38-game season → moderate K
        "home_advantage": 50.0,  # Strong home advantage in soccer
        "scale": 400.0,          # High parity in draws, suits wider scale
        "season_regress": 0.15,  # Rosters fairly stable (transfers, not drafts)
        "tempo_adjust": False,   # 90 min standard
        "season_boundary": "epl", # Aug → May
    },
    "mls": {
        "k": 10.0,              # 34-game season
        "home_advantage": 60.0,  # MLS has very strong home advantage (travel, turf, altitude)
        "scale": 400.0,
        "season_regress": 0.20,  # More roster turnover than EPL
        "tempo_adjust": False,
        "season_boundary": "mlb", # Calendar year: Feb/Mar → Nov
    },
    "ufc": {
        "k": 40.0,              # Fighters fight 2-3x/year → need high K per bout
        "home_advantage": 0.0,   # No home advantage in MMA
        "scale": 400.0,
        "season_regress": 0.0,   # No seasons — continuous
        "tempo_adjust": False,
        "season_boundary": "none",
    },
    "nhl": {
        "k": 8.0,               # 82-game season, similar to NBA but lower scoring → less signal per game
        "home_advantage": 50.0,  # NHL home advantage ~55% historically (less than NBA)
        "scale": 400.0,          # High parity — any team can beat any team on a given night
        "season_regress": 0.25,  # Moderate roster turnover (trades, free agency)
        "tempo_adjust": False,   # 60-min regulation standard
        "season_boundary": "ncaa",  # NHL season runs Oct → Jun (same shape as NBA)
    },
}


@dataclass
class EloEngine:
    k: float = 24.0
    home_advantage: float = HOME_ADVANTAGE
    initial: float = DEFAULT_RATING
    decay_half_life: float = 0.0   # days; 0 = no decay (uniform weighting)
    season_regress: float = 0.33   # fraction to regress toward initial between seasons
                                   # 0.33 = regress 1/3 of distance (accounts for roster turnover)
    tempo_adjust: bool = True      # normalize MOV by pace (avg possessions ~70)
    scale: float = SCALE           # Elo scale factor (300 for NCAAB, 400 for MLB, etc.)
    season_boundary: str = "ncaa"  # "ncaa" = Nov-Apr academic year, "mlb" = calendar year (Mar-Oct)

    # internal state — not constructor args
    ratings: dict[str, float] = field(default_factory=dict, repr=False)
    names: dict[str, str] = field(default_factory=dict, repr=False)
    history: list[dict] = field(default_factory=list, repr=False)
    _last_season_year: int | None = field(default=None, repr=False)

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
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a - adj) / self.scale))

    def regress_ratings(self) -> None:
        """
        Regress all ratings toward initial by season_regress fraction.

        Called between seasons to account for roster turnover, transfers,
        graduating seniors, and incoming freshmen. A 0.33 regress means:
            new_rating = old_rating - 0.33 * (old_rating - 1500)
        So a 1650-rated team becomes 1600.5, and a 1350-rated team becomes 1400.5.
        """
        if self.season_regress <= 0:
            return
        for tid in self.ratings:
            diff = self.ratings[tid] - self.initial
            self.ratings[tid] -= self.season_regress * diff

    def _maybe_season_reset(self, game_date: str | None) -> None:
        """Check if we've crossed into a new season and regress if so.

        NCAA season runs Nov→Apr. Season year = spring calendar year.
        MLB season runs Mar→Oct. Season year = calendar year.
        We trigger a regress when we see the first game of a new season
        (i.e. season_year increases).
        """
        if not game_date or self.season_regress <= 0:
            return

        try:
            month = int(game_date[5:7])
            year  = int(game_date[:4])
        except (ValueError, IndexError):
            return

        if self.season_boundary == "none":
            # No seasons (e.g. UFC) — never regress
            return

        if self.season_boundary == "epl":
            # EPL: Aug-May. Season year = year of August start.
            season_year = year if month >= 8 else year - 1
        elif self.season_boundary == "mlb":
            # MLB/MLS: calendar year. Season starts Feb/Mar.
            season_year = year
        else:
            # NCAA/NBA: Nov/Dec of year Y → season Y+1; Jan-Apr of year Y → season Y
            season_year = year + 1 if month >= 10 else year

        if self._last_season_year is None:
            self._last_season_year = season_year
            return

        if season_year > self._last_season_year:
            self.regress_ratings()
            self._last_season_year = season_year

    @staticmethod
    def _tempo_normalize(home_score: int, away_score: int) -> float:
        """
        Normalize margin of victory by pace.

        Average D1 game has ~70 possessions per team. A 10-point margin in
        an 80-possession game is less impressive than in a 60-possession game.

        We estimate possessions from total points (rough but effective):
            est_possessions = total_points / 2.0  (each possession ~1 point avg)
        Then normalize: adj_margin = raw_margin * (70 / est_poss)

        This prevents fast-paced teams from inflating their Elo via large raw margins.
        """
        total = home_score + away_score
        if total == 0:
            return 0.0
        raw_margin = abs(home_score - away_score)
        # Estimate possessions: total points / ~2.0 points per possession
        # Average game: ~140 total points / 2.0 = ~70 possessions
        est_poss = total / 2.0
        # Normalize to 70-possession baseline
        return raw_margin * (70.0 / max(est_poss, 30.0))

    def update(
        self,
        home_id: str,
        away_id: str,
        home_score: int,
        away_score: int,
        neutral: bool = False,
        date: str | None = None,
        k_override: float | None = None,
    ) -> float:
        """
        Process one finished game. Returns the pregame P(home wins).

        home_id / away_id: ESPN team ID strings
        """
        # Check for season boundary → regress ratings
        self._maybe_season_reset(date)

        outcome = 1.0 if home_score > away_score else 0.0
        p_home = self.win_prob(home_id, away_id, neutral)

        r_home = self.rating(home_id)
        r_away = self.rating(away_id)

        # Margin of victory scaling (FiveThirtyEight method).
        # Larger margins move ratings more, but diminishing returns via log.
        # Autocorrelation correction: blowouts vs weak teams count less.
        if self.tempo_adjust:
            margin = self._tempo_normalize(home_score, away_score)
        else:
            margin = float(abs(home_score - away_score))
        winner_elo = r_home if outcome == 1.0 else r_away
        loser_elo  = r_away if outcome == 1.0 else r_home
        elo_diff   = abs(winner_elo - loser_elo)
        mov_factor = math.log(margin + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

        k = k_override if k_override is not None else self.k
        self.ratings[home_id] = r_home + k * mov_factor * (outcome - p_home)
        self.ratings[away_id] = r_away + k * mov_factor * ((1.0 - outcome) - (1.0 - p_home))

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

        If decay_half_life > 0, the K factor is scaled by 0.5^(days_ago/half_life)
        where days_ago is measured from the most recent game in the list.
        This makes recent games move ratings more than old games.
        """
        from datetime import date as _date

        sorted_games = sorted(games, key=lambda x: x["date"])

        # Precompute max date for decay reference
        max_date = None
        if self.decay_half_life > 0 and sorted_games:
            try:
                max_date = _date.fromisoformat(
                    max(g["date"] for g in sorted_games if g.get("date"))
                )
            except (ValueError, TypeError):
                max_date = None

        for g in sorted_games:
            self.names[g["home_id"]] = g["home_name"]
            self.names[g["away_id"]] = g["away_name"]

            k_override = None
            if self.decay_half_life > 0 and max_date and g.get("date"):
                try:
                    days_ago = (max_date - _date.fromisoformat(g["date"])).days
                    k_override = self.k * (0.5 ** (days_ago / self.decay_half_life))
                except (ValueError, TypeError):
                    pass

            self.update(
                home_id=g["home_id"],
                away_id=g["away_id"],
                home_score=g["home_score"],
                away_score=g["away_score"],
                neutral=g["neutral"],
                date=g["date"],
                k_override=k_override,
            )

    def win_prob_from_ratings(self, r_a: float, r_b: float, neutral: bool = True) -> float:
        """
        P(team_a beats team_b) given explicit ratings.
        Used when injury adjustments are applied ad-hoc without mutating state.
        """
        adj = 0.0 if neutral else self.home_advantage
        return 1.0 / (1.0 + 10.0 ** ((r_b - r_a - adj) / self.scale))

    def adjusted_copy(self, overrides: dict[str, float]) -> "EloEngine":
        """
        Return a shallow copy with Elo adjustments applied.

        overrides: {team_id: delta} where delta is typically negative (injury penalty).
        Only teams present in self.ratings are adjusted — new teams are ignored.
        """
        import copy
        clone = copy.copy(self)
        clone.ratings = dict(self.ratings)
        for team_id, delta in overrides.items():
            if team_id in clone.ratings:
                clone.ratings[team_id] += delta
        return clone

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
