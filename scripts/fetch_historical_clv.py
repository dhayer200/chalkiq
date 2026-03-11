#!/usr/bin/env python3
"""
Backfill historical CLV records using The Odds API historical endpoint.

Requires a PAID Odds API plan — historical data is not available on the free tier.
One month (~$30) is enough to pull all the data you need; then cancel.

How it works:
  1. Loads all game data for the requested seasons from our ESPN cache.
  2. Builds walk-forward Elo probabilities (model_prob as it was BEFORE each game).
  3. Detects DNP-based injuries from boxscores and adjusts Elo predictions.
  4. Groups games by date and fetches opening + closing odds from Odds API.
  5. Matches Odds API team names to ESPN teams via fuzzy matching.
  6. Computes CLV = model_prob - closing_implied_prob and writes records.
  7. With --min-edge, only records games where model edge > threshold (selective).

Output: data/odds/historical_clv.jsonl  (same format as clv.jsonl + source=historical)

Usage:
    # Dry run first -- see what would be fetched without hitting the API
    python scripts/fetch_historical_clv.py --seasons 2025 2026 --dry-run

    # Real run with injury adjustments and selective 5% edge filter
    python scripts/fetch_historical_clv.py --seasons 2025 2026 --start-date 2024-11-04 --min-edge 0.05

    # No edge filter (record all games)
    python scripts/fetch_historical_clv.py --seasons 2025 2026 --start-date 2024-11-04

Environment:
    ODDS_API_KEY=your_paid_key   (set in .env or export)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from src.odds.api   import fetch_historical_odds, SPORT_NCAAB
from src.odds.clv   import compute_clv
from src.odds.match import build_index, match_game
from src.odds.store import append, read_all
from src.ratings.elo import EloEngine
from src.utils.data  import fetch_season


_OUT_FILE = "historical_clv.jsonl"
_CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "odds" / "_hist_cache"
_ROOT = Path(__file__).resolve().parents[1]

SEASONS: dict[int, tuple[date, date]] = {
    2023: (date(2022, 11, 7),  date(2023, 4,  3)),
    2024: (date(2023, 11, 6),  date(2024, 4,  8)),
    2025: (date(2024, 11, 4),  date(2025, 4,  7)),
    2026: (date(2025, 11, 4),  date.today()),
}

CACHE_DIRS: dict[str, dict[int, str]] = {
    "mens": {
        2023: "data/raw/mens/2023",
        2024: "data/raw/mens/2024",
        2025: "data/raw/mens/2025",
        2026: "data/raw/mens",
    },
    "womens": {
        2023: "data/raw/womens/2023",
        2024: "data/raw/womens/2024",
        2025: "data/raw/womens/2025",
        2026: "data/raw/womens",
    },
}

BOX_DIRS: dict[str, dict[int, str]] = {
    "mens": {
        2026: "data/raw/mens/boxscores",
    },
    "womens": {
        2026: "data/raw/womens/boxscores",
    },
}

# Bookmaker priority for CLV (prefer most liquid books)
_BOOK_PRIORITY = {"draftkings": 0, "fanduel": 1, "betmgm": 2, "caesars": 3}

# Break-even at -110 juice
_BREAKEVEN_110 = 100.0 / (110.0 + 100.0)   # ≈ 0.5238

# How many consecutive dates with zero matches before we bail
_MAX_CONSECUTIVE_MISSES = 10


def _cache_key(sport: str, date_iso: str) -> Path:
    """Return path for a cached API response."""
    h = hashlib.md5(f"{sport}:{date_iso}".encode()).hexdigest()[:12]
    safe_date = date_iso.replace(":", "").replace("-", "")
    return _CACHE_DIR / f"{safe_date}_{h}.json"


def fetch_historical_cached(sport: str, date_iso: str) -> list[dict]:
    """Fetch historical odds with local file caching to avoid re-fetching."""
    cache_path = _cache_key(sport, date_iso)
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text())
            print(f"  [cached] {date_iso}")
            return data
        except (json.JSONDecodeError, OSError):
            cache_path.unlink(missing_ok=True)

    result = fetch_historical_odds(sport, date_iso)

    # Cache the result (even empty ones, so we don't re-fetch)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result))

    return result


def best_odds_lookup(odds_records: list[dict]) -> dict[tuple[str, str], dict]:
    """
    Build {(home_team, away_team): best_record} preferring higher-priority books.
    Uses Odds API team names as keys.
    """
    lookup: dict[tuple[str, str], dict] = {}
    for r in odds_records:
        key = (r["home_team"], r["away_team"])
        existing = lookup.get(key)
        if existing is None or (
            _BOOK_PRIORITY.get(r["bookmaker"], 99)
            < _BOOK_PRIORITY.get(existing["bookmaker"], 99)
        ):
            lookup[key] = r
    return lookup


# ── Injury detection from boxscores ──────────────────────────────────────── #

def build_walk_forward_injury_timeline(
    division: str,
    seasons: list[int],
    team_elo_fn,
) -> dict[str, dict[str, float]]:
    """
    Build {game_date: {team_id: elo_penalty}} from boxscore DNP detection.

    Uses PlayerEloEngine to compute actual injury impact based on player ratings
    and minutes, rather than heuristic position-based estimates.
    """
    from src.players.engine import PlayerEloEngine, load_boxscores

    all_boxscores: list[dict] = []
    for season in seasons:
        box_dirs = BOX_DIRS.get(division, {})
        box_dir = box_dirs.get(season)
        if not box_dir:
            continue
        full_path = _ROOT / box_dir
        if full_path.exists():
            boxes = load_boxscores(str(full_path))
            all_boxscores.extend(boxes)

    if not all_boxscores:
        return {}

    print(f"  Loaded {len(all_boxscores)} boxscore files for injury detection")

    # Build player engine walk-forward
    player_engine = PlayerEloEngine(k=16.0)
    sorted_boxes = sorted(all_boxscores, key=lambda g: g.get("game_date", ""))

    # Track per-player minutes history for DNP detection
    player_minutes: dict[str, list[float]] = defaultdict(list)
    player_team: dict[str, str] = {}

    # Build {(game_date, team_id): set(player_ids who played)}
    played_lookup: dict[tuple[str, str], set[str]] = defaultdict(set)
    for g in sorted_boxes:
        game_date = g.get("game_date", "")
        for p in g.get("players", []):
            pid = p.get("player_id", "")
            mins = p.get("min", 0)
            if pid and mins >= 5:
                played_lookup[(game_date, p.get("team_id", ""))].add(pid)

    timeline: dict[str, dict[str, float]] = {}
    min_prior_games = 5
    min_avg_min = 15.0

    for g in sorted_boxes:
        game_date = g.get("game_date", "")

        # Process this game through player engine (updates ratings)
        player_engine.process_game(g)

        # Track minutes for players who played
        teams_in_game: set[str] = set()
        for p in g.get("players", []):
            pid = p.get("player_id", "")
            mins = p.get("min", 0)
            tid = p.get("team_id", "")
            if pid and mins >= 5:
                player_minutes[pid].append(float(mins))
                player_team[pid] = tid
                teams_in_game.add(tid)

        # Detect DNPs: players who should have played but didn't
        for team_id in teams_in_game:
            if not team_id:
                continue
            team_played = played_lookup.get((game_date, team_id), set())
            day_penalty = 0.0

            for pid, mins_hist in player_minutes.items():
                if player_team.get(pid) != team_id:
                    continue
                if len(mins_hist) < min_prior_games:
                    continue
                # Use last 10 games for avg minutes
                recent = mins_hist[-10:]
                avg_min = sum(recent) / len(recent)
                if avg_min < min_avg_min:
                    continue
                if pid in team_played:
                    continue  # they played, not a DNP

                # Surprise DNP — use player Elo engine's calibrated impact
                impact = player_engine.injury_elo_impact(pid)
                if impact > 0:
                    day_penalty -= impact  # negative = team gets weaker

            if day_penalty != 0.0:
                if game_date not in timeline:
                    timeline[game_date] = {}
                timeline[game_date][team_id] = timeline[game_date].get(team_id, 0.0) + day_penalty

    n_dates = len(timeline)
    n_adjustments = sum(len(v) for v in timeline.values())
    print(f"  Injury timeline: {n_adjustments} team-date adjustments across {n_dates} dates")
    return timeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical CLV from The Odds API (paid plan required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seasons",     nargs="+", type=int, default=[2026],
                        help="Season years to backfill (e.g. 2025 2026)")
    parser.add_argument("--sport",       default=SPORT_NCAAB,
                        help="Odds API sport key")
    parser.add_argument("--division",    default="mens", choices=["mens", "womens"],
                        help="ESPN division")
    parser.add_argument("--start-date",  default=None,
                        help="Skip dates before this (YYYY-MM-DD). Saves API calls for dates "
                             "where the Odds API has no NCAAB data.")
    parser.add_argument("--min-edge",    type=float, default=0.0,
                        help="Only record CLV for games where model edge > this threshold "
                             "(e.g. 0.05 = 5%%). Set to 0 for all games.")
    parser.add_argument("--open-offset", type=int, default=3,
                        help="Days before game date to fetch opening line")
    parser.add_argument("--close-hour",  type=int, default=22,
                        help="UTC hour on game date to fetch closing line (22=5pm ET)")
    parser.add_argument("--delay",       type=float, default=0.5,
                        help="Seconds to wait between API calls (avoid rate limiting)")
    parser.add_argument("--max-misses",  type=int, default=_MAX_CONSECUTIVE_MISSES,
                        help="Bail after this many consecutive dates with no NCAAB data")
    parser.add_argument("--no-injuries", action="store_true",
                        help="Disable injury adjustments (pure Elo only)")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Bypass local response cache (re-fetch everything)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print what would be fetched WITHOUT hitting the API")
    args = parser.parse_args()

    start_date_filter = date.fromisoformat(args.start_date) if args.start_date else None

    # ── Load all game data ──────────────────────────────────────────────────
    print("Loading ESPN game data...")
    all_games: list[dict] = []
    for season in sorted(args.seasons):
        if season not in SEASONS:
            print(f"  Unknown season {season}, skipping")
            continue
        start, end = SEASONS[season]
        cache_dirs = CACHE_DIRS.get(args.division, CACHE_DIRS["mens"])
        cache      = cache_dirs.get(season, f"data/raw/{args.division}/{season}")
        games = fetch_season(start, end, cache_dir=cache, division=args.division, verbose=False)
        print(f"  {season}: {len(games):,} games loaded")
        all_games.extend(games)

    if not all_games:
        print("No games found. Check your seasons/division args.")
        sys.exit(1)

    # ── Walk-forward Elo — capture model prob BEFORE each game ──────────────
    print(f"\nBuilding walk-forward Elo probs for {len(all_games):,} games...")
    sorted_games = sorted(all_games, key=lambda g: g["date"])
    engine = EloEngine(k=24.0, home_advantage=100.0)

    # ── Build injury timeline from boxscores ─────────────────────────────────
    injury_timeline: dict[str, dict[str, float]] = {}
    if not args.no_injuries:
        print("\nBuilding injury timeline from boxscores...")
        injury_timeline = build_walk_forward_injury_timeline(
            division=args.division,
            seasons=args.seasons,
            team_elo_fn=lambda tid: engine.rating(tid),
        )
        if not injury_timeline:
            print("  No boxscores available — running without injury adjustments")

    # ── Compute pre-game probabilities (with injury adjustments) ─────────────
    pre_game_probs: dict[tuple[str, str, str], float] = {}
    n_injury_adjusted = 0

    for g in sorted_games:
        engine.names[g["home_id"]] = g["home_name"]
        engine.names[g["away_id"]] = g["away_name"]

        home_id = g["home_id"]
        away_id = g["away_id"]
        neutral = g.get("neutral", True)

        # Check for injury adjustments on this date
        day_adj = injury_timeline.get(g["date"], {})
        home_adj = day_adj.get(home_id, 0.0)
        away_adj = day_adj.get(away_id, 0.0)

        if home_adj != 0.0 or away_adj != 0.0:
            # Injury-adjusted probability
            r_home = engine.rating(home_id) + home_adj
            r_away = engine.rating(away_id) + away_adj
            prob = engine.win_prob_from_ratings(r_home, r_away, neutral=neutral)
            n_injury_adjusted += 1
        else:
            prob = engine.win_prob(home_id, away_id, neutral=neutral)

        key = (home_id, away_id, g["date"])
        pre_game_probs[key] = prob

        engine.update(
            home_id    = home_id,
            away_id    = away_id,
            home_score = g["home_score"],
            away_score = g["away_score"],
            neutral    = neutral,
            date       = g["date"],
        )

    print(f"  {n_injury_adjusted} games had injury-adjusted probabilities")

    espn_index = build_index(engine.names)

    games_by_date: dict[str, list[dict]] = defaultdict(list)
    for g in sorted_games:
        games_by_date[g["date"]].append(g)

    unique_dates = sorted(games_by_date.keys())

    # Apply --start-date filter
    if start_date_filter:
        before = len(unique_dates)
        unique_dates = [d for d in unique_dates if date.fromisoformat(d) >= start_date_filter]
        skipped = before - len(unique_dates)
        if skipped:
            print(f"\n--start-date {args.start_date}: skipping {skipped} earlier dates")

    print(f"Unique game dates : {len(unique_dates)}")
    print(f"Estimated API calls: {len(unique_dates) * 2} (open + close per date)")
    if args.min_edge > 0:
        print(f"Edge filter: only recording games with model edge > {args.min_edge:.1%}")

    # ── Load existing historical CLV to skip already-processed games ─────────
    existing_keys: set[str] = {
        r["game_key"]
        for r in read_all(_OUT_FILE)
        if r.get("game_key")
    }
    print(f"Already have {len(existing_keys)} historical CLV records")

    if args.dry_run:
        print("\n[DRY RUN] No API calls will be made.")
        print(f"Would fetch {len(unique_dates)} dates × 2 = {len(unique_dates)*2} API requests")
        print(f"First date: {unique_dates[0] if unique_dates else 'none'}")
        print(f"Last date:  {unique_dates[-1] if unique_dates else 'none'}")
        sample = unique_dates[:5]
        for d in sample:
            gd       = date.fromisoformat(d)
            open_ts  = (gd - timedelta(days=args.open_offset)).strftime("%Y-%m-%dT12:00:00Z")
            close_ts = gd.strftime(f"%Y-%m-%dT{args.close_hour:02d}:00:00Z")
            n        = len(games_by_date[d])
            print(f"  {d}  open={open_ts}  close={close_ts}  ({n} games)")
        if len(unique_dates) > 5:
            print(f"  ... and {len(unique_dates) - 5} more dates")
        return

    # ── Main fetch loop ─────────────────────────────────────────────────────
    total_written = 0
    total_skipped = 0
    total_filtered = 0   # games skipped due to --min-edge
    consecutive_misses = 0
    fetch_fn = fetch_historical_odds if args.no_cache else fetch_historical_cached

    for i, game_date_str in enumerate(unique_dates):
        game_date = date.fromisoformat(game_date_str)
        open_ts   = (game_date - timedelta(days=args.open_offset)).strftime("%Y-%m-%dT12:00:00Z")
        close_ts  = game_date.strftime(f"%Y-%m-%dT{args.close_hour:02d}:00:00Z")
        day_games = games_by_date[game_date_str]

        print(f"\n[{i+1}/{len(unique_dates)}] {game_date_str}  ({len(day_games)} ESPN games)")

        # Check if all games for this date are already processed
        day_keys = {
            f"{g['home_id']}_{g['away_id']}_{game_date_str}"
            for g in day_games
        }
        if day_keys.issubset(existing_keys):
            print(f"  all games already processed, skipping API calls")
            total_skipped += len(day_keys)
            consecutive_misses = 0
            continue

        # Fetch opening and closing odds
        time.sleep(args.delay)
        opening_odds = fetch_fn(args.sport, open_ts)
        time.sleep(args.delay)
        closing_odds = fetch_fn(args.sport, close_ts)

        if not closing_odds:
            print(f"  no closing odds returned")
            consecutive_misses += 1
            if consecutive_misses >= args.max_misses:
                print(f"\n  *** {args.max_misses} consecutive dates with no data — "
                      f"Odds API likely has no {args.sport} coverage this far back.")
                print(f"  *** Stopping early to save API quota. "
                      f"Try --start-date with a later date.")
                break
            continue

        open_lookup  = best_odds_lookup(opening_odds)
        close_lookup = best_odds_lookup(closing_odds)

        espn_by_id: dict[tuple[str, str], dict] = {
            (g["home_id"], g["away_id"]): g for g in day_games
        }

        matched = 0
        for (odds_home, odds_away), close_rec in close_lookup.items():
            espn_home_id, espn_away_id = match_game(odds_home, odds_away, espn_index)
            if not espn_home_id or not espn_away_id:
                continue

            espn_game = espn_by_id.get((espn_home_id, espn_away_id))
            if not espn_game:
                continue

            game_key = f"{espn_home_id}_{espn_away_id}_{game_date_str}"
            if game_key in existing_keys:
                continue

            open_rec = open_lookup.get((odds_home, odds_away)) or close_rec

            model_prob = pre_game_probs.get((espn_home_id, espn_away_id, game_date_str))
            if model_prob is None:
                continue

            # ── Selective betting: only record if model has edge ─────────
            if args.min_edge > 0:
                edge_home = model_prob - _BREAKEVEN_110
                edge_away = (1 - model_prob) - _BREAKEVEN_110
                best_edge = max(edge_home, edge_away)
                if best_edge < args.min_edge:
                    total_filtered += 1
                    continue

            hs = espn_game.get("home_score")
            as_ = espn_game.get("away_score")
            home_won = (hs > as_) if (hs is not None and as_ is not None) else None

            clv = compute_clv(
                model_prob_home = model_prob,
                opening         = open_rec,
                closing         = close_rec,
                game_id         = espn_game.get("game_id", game_key),
                home_team       = espn_game["home_name"],
                away_team       = espn_game["away_name"],
                bookmaker       = close_rec["bookmaker"],
                home_won        = home_won,
            )

            # Check if there was an injury adjustment for this game
            day_adj = injury_timeline.get(game_date_str, {})
            has_injury = (day_adj.get(espn_home_id, 0.0) != 0.0 or
                         day_adj.get(espn_away_id, 0.0) != 0.0)

            record = {
                "game_key":          game_key,
                "game_id":           espn_game.get("game_id", ""),
                "game_date":         game_date_str,
                "home_team":         espn_game["home_name"],
                "away_team":         espn_game["away_name"],
                "bookmaker":         close_rec["bookmaker"],
                "model_prob_home":   round(model_prob, 4),
                "opening_home_ml":   open_rec["home_ml"],
                "closing_home_ml":   close_rec["home_ml"],
                "opening_home_prob": round(open_rec["home_prob"], 4),
                "closing_home_prob": round(close_rec["home_prob"], 4),
                "clv_vs_opening":    round(clv.clv_vs_opening, 4),
                "clv_vs_closing":    round(clv.clv_vs_closing, 4),
                "home_won":          home_won,
                "injury_adjusted":   has_injury,
                "source":            "historical",
                "recorded_at":       datetime.now(timezone.utc).isoformat(),
            }

            append(record, _OUT_FILE)
            existing_keys.add(game_key)
            total_written += 1
            matched += 1

            inj_flag = " 🏥" if has_injury else ""
            result_str = "W" if home_won else ("L" if home_won is False else "?")
            print(
                f"  {espn_game['away_name']} @ {espn_game['home_name']}"
                f"  model={model_prob:.1%}"
                f"  close={close_rec['home_prob']:.1%}"
                f"  CLV={clv.clv_vs_closing:+.1%}"
                f"  [{result_str}]"
                f"  [{close_rec['bookmaker']}]{inj_flag}"
            )

        if matched == 0:
            consecutive_misses += 1
            print(f"  no NCAAB matches ({consecutive_misses}/{args.max_misses} consecutive misses)")
            if consecutive_misses >= args.max_misses:
                print(f"\n  *** {args.max_misses} consecutive dates with no NCAAB matches — "
                      f"stopping to save API quota.")
                print(f"  *** Try --start-date with a later date, or check that "
                      f"the Odds API has {args.sport} data for this period.")
                break
        else:
            consecutive_misses = 0
            print(f"  matched {matched} games")

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Done. Written: {total_written}  Skipped (existing): {total_skipped}  "
          f"Filtered (below edge): {total_filtered}")

    all_records = read_all(_OUT_FILE)
    if all_records:
        beat   = sum(1 for r in all_records if (r.get("clv_vs_closing") or 0) > 0)
        n      = len(all_records)
        avg    = sum(r.get("clv_vs_closing") or 0 for r in all_records) / n
        n_won  = sum(1 for r in all_records if r.get("home_won") is True)
        n_comp = sum(1 for r in all_records if r.get("home_won") is not None)
        n_inj  = sum(1 for r in all_records if r.get("injury_adjusted"))

        print(f"  Historical CLV records : {n}")
        print(f"  Beat closing line      : {beat}/{n} ({beat/n:.1%})")
        print(f"  Avg CLV vs close       : {avg:+.2%}")
        if n_comp:
            print(f"  Home win rate          : {n_won}/{n_comp} ({n_won/n_comp:.1%})")
        if n_inj:
            # Show separate stats for injury-adjusted games
            inj_records = [r for r in all_records if r.get("injury_adjusted")]
            inj_beat = sum(1 for r in inj_records if (r.get("clv_vs_closing") or 0) > 0)
            inj_avg  = sum(r.get("clv_vs_closing") or 0 for r in inj_records) / len(inj_records)
            print(f"  Injury-adjusted games  : {n_inj}")
            print(f"    Beat close (injured) : {inj_beat}/{n_inj} ({inj_beat/n_inj:.1%})")
            print(f"    Avg CLV (injured)    : {inj_avg:+.2%}")

    print(f"{'='*60}")
    print(f"  Output: data/odds/{_OUT_FILE}")


if __name__ == "__main__":
    main()
