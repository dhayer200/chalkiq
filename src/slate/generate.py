"""
Shared slate generation logic — used by both daily_slate.py CLI and the newsletter.

Produces a pure-data SlateResult that can be formatted as ANSI (CLI) or HTML (email).

Usage:
    from src.slate.generate import generate_slate, load_engine

    engine = load_engine("mens")
    result = generate_slate("mens", engine=engine)
    print(result.bet_sheet)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from src.live.feed import fetch_other_games
from src.odds.api import (
    fetch_odds,
    SPORT_NCAAB,
    SPORT_NCAAF,
)
from src.odds.clv import CLVResult, clv_summary
from src.odds.match import build_index, match_team
from src.odds.store import load_snapshots, load_clv_records
from src.ratings.elo import EloEngine, SPORT_CONFIGS
from src.signals.injuries import fetch_team_injuries, _CONCERN_STATUSES, _OUT_STATUSES, estimate_impact
from src.utils.data import fetch_season

_ROOT = Path(__file__).resolve().parents[2]

BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]

# Division → Odds API sport key (active: mens only; cfb added at launch)
ODDS_SPORT = {
    "mens": SPORT_NCAAB,
    "cfb":  SPORT_NCAAF,
}

DIVISION_LABELS = {
    "mens": "NCAA Men's Basketball",
    "cfb":  "NCAA Football (FBS)",
}


# ── Engine loader (handles all 4 divisions) ─────────────────────────────── #

def load_engine(division: str) -> EloEngine:
    """Build an Elo engine for any supported division."""
    today = date.today()

    if division == "cfb":
        yr = today.year if today.month >= 8 else today.year - 1
        season_start = date(yr, 8, 20)
    else:
        # Men's CBB: Nov → Apr
        yr = today.year if today.month >= 10 else today.year - 1
        season_start = date(yr, 11, 4)

    cache_dir = _ROOT / "data" / "raw" / division
    games = fetch_season(
        season_start, today,
        cache_dir=str(cache_dir), division=division, verbose=False,
    )

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
    return engine


# ── Injury helpers ──────────────────────────────────────────────────────── #

def _get_injury_report(team_id: str, division: str) -> list[dict]:
    raw = fetch_team_injuries(team_id, division=division)
    out = []
    for p in raw:
        status = p.get("status", "")
        if status in _CONCERN_STATUSES:
            out.append({
                "name": p.get("player_name", p.get("name", "?")),
                "pos": p.get("position", p.get("pos", "?")),
                "status": status,
                "detail": p.get("detail", ""),
            })
    return out


def _compute_injury_delta(injuries: list[dict], team_elo: float) -> float:
    """
    Sum Elo point penalties from all concern-status injuries.

    Out/Doubtful/IR/Suspension: full impact via estimate_impact().
    Questionable/Day-To-Day: 30% of full impact (likely plays).

    Returns a negative float (penalty) or 0.0.
    """
    delta = 0.0
    for inj in injuries:
        status = inj.get("status", "")
        pos = inj.get("pos", "?")
        name = inj.get("name", "")
        if status in _OUT_STATUSES:
            delta += estimate_impact(name, pos, team_elo, status)
        elif status in _CONCERN_STATUSES:
            # Questionable / Day-To-Day: partial impact
            full = estimate_impact(name, pos, team_elo, "Out")
            delta += full * 0.30
    return delta


def _injury_adjusted_prob(
    home_elo: float,
    away_elo: float,
    home_delta: float,
    away_delta: float,
    home_advantage: float,
    scale: float,
    neutral: bool,
) -> float:
    """Win probability for home team using injury-adjusted Elo ratings."""
    adj = 0.0 if neutral else home_advantage
    eff_home = home_elo + home_delta
    eff_away = away_elo + away_delta
    return 1.0 / (1.0 + 10.0 ** ((eff_away - eff_home - adj) / scale))


# ── Opening line lookup ─────────────────────────────────────────────────── #

def _get_opening_lines(snapshots: list[dict]) -> dict[tuple, dict]:
    seen: dict[tuple, dict] = {}
    for s in snapshots:
        key = (s.get("game_id", ""), s.get("bookmaker", ""))
        if key not in seen:
            seen[key] = s
    return seen


# ── Result dataclass ────────────────────────────────────────────────────── #

@dataclass
class SlateResult:
    division: str
    game_cards: list[dict] = field(default_factory=list)
    bet_sheet: list[dict] = field(default_factory=list)
    final_games: list[dict] = field(default_factory=list)
    clv_stats: dict = field(default_factory=dict)
    model_accuracy: float | None = None
    n_upcoming: int = 0
    n_live: int = 0
    n_final: int = 0


# ── Main generation function ────────────────────────────────────────────── #

def generate_slate(
    division: str,
    engine: EloEngine | None = None,
    min_edge: float = 0.015,
    top_n: int | None = None,
    no_fetch: bool = False,
    target_date: date | None = None,
    budget=None,
) -> SlateResult:
    """
    Generate a complete slate for a division. Returns pure data (no formatting).

    Args:
        division: "mens", "womens", "nba", or "mlb"
        engine: pre-built EloEngine (built if not provided)
        min_edge: minimum edge to flag as a bet (0.03 = 3%)
        top_n: limit to top N games by edge
        no_fetch: skip live odds API call, use cached snapshots
        target_date: date to generate for (default: today)
        budget: optional QuotaBudget for Odds API rate limiting
    """
    target = target_date or date.today()
    result = SlateResult(division=division)

    # 1. Fetch games from ESPN (free)
    all_games = fetch_other_games(division=division, for_date=target)
    if not all_games:
        return result

    scheduled = [g for g in all_games if g["status"] == "scheduled"]
    live_now = [g for g in all_games if g["status"] not in ("scheduled", "final")]
    final_games = [g for g in all_games if g["status"] == "final"]

    result.n_upcoming = len(scheduled) + len(live_now)
    result.n_live = len(live_now)
    result.n_final = len(final_games)

    all_upcoming = live_now + scheduled

    # 2. Elo engine
    if engine is None:
        engine = load_engine(division)
    rankings = {tid: rank for rank, (tid, _, _r) in enumerate(engine.rankings(), 1)}
    index = build_index(engine.names)

    # 3. Live odds
    live_odds: list[dict] = []
    if not no_fetch:
        odds_sport = ODDS_SPORT.get(division, SPORT_NCAAB)
        try:
            live_odds = fetch_odds(
                sport=odds_sport,
                bookmakers=",".join(BOOKS),
                budget=budget,
            )
        except EnvironmentError:
            pass

    # 4. Historical snapshots (opening lines)
    snap_dir = _ROOT / "data" / "odds"
    all_snaps = load_snapshots(snap_dir)
    opening_lines = _get_opening_lines(all_snaps)

    live_by_game: dict[str, list[dict]] = defaultdict(list)
    for o in live_odds:
        live_by_game[o["game_id"]].append(o)

    # 5. Injury cache
    injury_cache: dict[str, list[dict]] = {}

    def get_injuries(team_id: str) -> list[dict]:
        if team_id not in injury_cache:
            injury_cache[team_id] = _get_injury_report(team_id, division)
        return injury_cache[team_id]

    # 6. Build game cards
    game_cards: list[dict] = []

    for g in all_upcoming:
        home_id = g["home_id"]
        away_id = g["away_id"]
        home_name = g["home_name"]
        away_name = g["away_name"]

        home_rank = rankings.get(home_id, 999)
        away_rank = rankings.get(away_id, 999)

        # Fetch injuries early so we can adjust Elo before computing win prob
        home_injuries = get_injuries(home_id)
        away_injuries = get_injuries(away_id)

        home_elo = engine.rating(home_id)
        away_elo = engine.rating(away_id)
        home_delta = _compute_injury_delta(home_injuries, home_elo)
        away_delta = _compute_injury_delta(away_injuries, away_elo)

        neutral = g.get("neutral", False)
        model_home = _injury_adjusted_prob(
            home_elo, away_elo, home_delta, away_delta,
            engine.home_advantage, engine.scale, neutral,
        )
        model_away = 1.0 - model_home

        # Match to odds API game
        matched_odds_id: str | None = None
        for oid, odds_list in live_by_game.items():
            if not odds_list:
                continue
            s = odds_list[0]
            h_match = match_team(s["home_team"], index)
            a_match = match_team(s["away_team"], index)
            if (h_match == home_id and a_match == away_id) or \
               (h_match == away_id and a_match == home_id):
                matched_odds_id = oid
                break

        # Best current line per side
        best_home_ml: int | None = None
        best_away_ml: int | None = None
        best_home_prob: float | None = None
        best_away_prob: float | None = None
        current_lines: dict[str, dict] = {}

        if matched_odds_id:
            for o in live_by_game[matched_odds_id]:
                s0 = live_by_game[matched_odds_id][0]
                h_match = match_team(s0["home_team"], index)
                flipped = (h_match == away_id)
                h_prob = o["away_prob"] if flipped else o["home_prob"]
                a_prob = o["home_prob"] if flipped else o["away_prob"]
                h_ml = o["away_ml"] if flipped else o["home_ml"]
                a_ml = o["home_ml"] if flipped else o["away_ml"]
                book = o["bookmaker"]
                current_lines[book] = {
                    "home_ml": h_ml, "away_ml": a_ml,
                    "home_prob": h_prob, "away_prob": a_prob,
                }
                if best_home_ml is None or \
                   (h_ml < 0 and h_ml > (best_home_ml or -9999)) or \
                   (h_ml >= 0 and (best_home_ml is None or best_home_ml < 0 or h_ml > best_home_ml)):
                    best_home_ml = h_ml
                    best_home_prob = h_prob
                if best_away_ml is None or \
                   (a_ml < 0 and a_ml > (best_away_ml or -9999)) or \
                   (a_ml >= 0 and (best_away_ml is None or best_away_ml < 0 or a_ml > best_away_ml)):
                    best_away_ml = a_ml
                    best_away_prob = a_prob

        # Opening lines
        open_home_ml: int | None = None
        open_home_prob: float | None = None
        for book in BOOKS:
            if matched_odds_id:
                ok = (matched_odds_id, book)
                if ok in opening_lines:
                    snap = opening_lines[ok]
                    s0 = live_by_game.get(matched_odds_id, [{}])[0]
                    h_match = match_team(s0.get("home_team", ""), index) if s0 else None
                    flipped = (h_match == away_id)
                    open_home_ml = snap["away_ml"] if flipped else snap["home_ml"]
                    open_home_prob = snap["away_prob"] if flipped else snap["home_prob"]
                    break

        # Edge
        edge_home = (model_home - best_home_prob) if best_home_prob else None
        edge_away = (model_away - best_away_prob) if best_away_prob else None

        # Line movement
        line_move_str = ""
        if open_home_prob is not None and best_home_prob is not None:
            delta = best_home_prob - open_home_prob
            if abs(delta) >= 0.005:
                direction = "toward home" if delta > 0 else "toward away"
                line_move_str = f"{direction} ({delta:+.1%})"
            else:
                line_move_str = "stable"

        # Max edge for sorting
        max_edge = max(
            abs(edge_home) if edge_home is not None else 0,
            abs(edge_away) if edge_away is not None else 0,
        )
        bet_side = None
        if edge_home is not None and edge_away is not None:
            if edge_home >= min_edge:
                bet_side = "HOME"
            elif edge_away >= min_edge:
                bet_side = "AWAY"

        card = {
            "home_name": home_name,
            "away_name": away_name,
            "home_id": home_id,
            "away_id": away_id,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_injury_delta": home_delta,
            "away_injury_delta": away_delta,
            "home_rank": home_rank,
            "away_rank": away_rank,
            "model_home": model_home,
            "model_away": model_away,
            "current_home_prob": best_home_prob,
            "current_away_prob": best_away_prob,
            "best_home_ml": best_home_ml,
            "best_away_ml": best_away_ml,
            "open_home_ml": open_home_ml,
            "open_home_prob": open_home_prob,
            "edge_home": edge_home,
            "edge_away": edge_away,
            "max_edge": max_edge,
            "bet_side": bet_side,
            "current_lines": current_lines,
            "line_move": line_move_str,
            "home_injuries": home_injuries,
            "away_injuries": away_injuries,
            "tip_detail": g.get("status_detail", ""),
            "status": g["status"],
        }
        game_cards.append(card)

    # Sort by edge descending
    game_cards.sort(key=lambda x: x["max_edge"], reverse=True)
    if top_n:
        game_cards = game_cards[:top_n]

    result.game_cards = game_cards
    result.bet_sheet = [c for c in game_cards if c["bet_side"]]

    # 7. Completed games + model accuracy
    if final_games:
        for g in final_games:
            model_h = engine.win_prob(
                g["home_id"], g["away_id"],
                neutral=g.get("neutral", False),
            )
            g["model_home"] = model_h
            g["model_correct"] = (model_h >= 0.5) == (g["home_score"] > g["away_score"])

        result.final_games = final_games
        n_correct = sum(1 for g in final_games if g["model_correct"])
        result.model_accuracy = n_correct / len(final_games) if final_games else None

    # 8. CLV summary from all historical records
    clv_records = load_clv_records()
    if clv_records:
        clv_results = [
            CLVResult(
                game_id=r["game_id"],
                home_team=r["home_team"],
                away_team=r["away_team"],
                bookmaker=r["bookmaker"],
                model_prob_home=r["model_prob_home"],
                opening_prob_home=r["opening_home_prob"],
                closing_prob_home=r["closing_home_prob"],
                clv_vs_opening=r["clv_vs_opening"],
                clv_vs_closing=r["clv_vs_closing"],
                home_won=r.get("home_won"),
            )
            for r in clv_records
        ]
        result.clv_stats = clv_summary(clv_results)

    return result
