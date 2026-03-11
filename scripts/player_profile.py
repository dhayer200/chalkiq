#!/usr/bin/env python3
"""
player_profile.py — Comprehensive player profile CLI for ChalkIQ.

Usage:
    python scripts/player_profile.py "Paige Bueckers"          # single player profile
    python scripts/player_profile.py --top 50                  # leaderboard (top 50)
    python scripts/player_profile.py --team "UConn"            # all players on a team
    python scripts/player_profile.py --top 50 --sort ppg       # leaderboard sorted by PPG
    python scripts/player_profile.py --position G --top 20     # top 20 guards
    python scripts/player_profile.py --division womens "Paige Bueckers"

Supports:
    --division mens|womens   (default: mens)
    --top N                  leaderboard mode
    --team "Name"            filter by team (fuzzy match)
    --position G|F|C         filter by position
    --min-games N            minimum games played (default: 5)
    --sort elo|off_eff|def_imp|ppg|game_score
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.players.engine import PlayerEloEngine, load_boxscores

# ── ANSI helpers ──────────────────────────────────────────────────────────── #

BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RESET  = "\033[0m"

def c(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


# ── Box-drawing table helpers ─────────────────────────────────────────────── #

def _hline(widths: list[int], left: str, mid: str, right: str, fill: str = "─") -> str:
    return left + mid.join(fill * (w + 2) for w in widths) + right


def _row(cells: list[str], widths: list[int], aligns: list[str] | None = None) -> str:
    """Build one row.  aligns: 'l' or 'r' per column (default left)."""
    if aligns is None:
        aligns = ["l"] * len(cells)
    parts = []
    for cell, w, a in zip(cells, widths, aligns):
        if a == "r":
            parts.append(f" {cell:>{w}} ")
        else:
            parts.append(f" {cell:<{w}} ")
    return "│" + "│".join(parts) + "│"


def print_table(headers: list[str], rows: list[list[str]],
                aligns: list[str] | None = None) -> None:
    """Print a box-drawn table to stdout."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            # strip ANSI for width calc
            plain = cell
            for esc in (BOLD, DIM, GREEN, RED, CYAN, YELLOW, RESET):
                plain = plain.replace(esc, "")
            widths[i] = max(widths[i], len(plain))

    print(_hline(widths, "┌", "┬", "┐"))
    print(_row(headers, widths, aligns))
    print(_hline(widths, "├", "┼", "┤"))
    for row in rows:
        print(_row(row, widths, aligns))
    print(_hline(widths, "└", "┴", "┘"))


# ── Data loading ──────────────────────────────────────────────────────────── #

def load_engine_and_boxscores(division: str) -> tuple[PlayerEloEngine, list[dict]]:
    """Load boxscores, build player Elo engine, return (engine, raw_boxscores)."""
    root = Path(__file__).resolve().parents[1]
    box_dir = root / "data" / "raw" / division / "boxscores"
    if not box_dir.exists():
        print(c(f"  [error] Boxscore dir not found: {box_dir}", RED))
        sys.exit(1)

    print(c("  Loading boxscores...", DIM), end="\r", flush=True)
    games = load_boxscores(str(box_dir))
    print(f"  Loaded {len(games)} boxscore files" + " " * 20)

    print(c("  Building player Elo ratings...", DIM), end="\r", flush=True)
    engine = PlayerEloEngine(k=16.0)
    engine.process_games(games)
    n_players = len(engine.ratings)
    print(f"  Player engine: {n_players} players rated" + " " * 20)

    return engine, games


def build_player_boxscore_index(games: list[dict]) -> dict[str, list[dict]]:
    """
    Build {player_id: [game_record, ...]} from raw boxscores.
    Each game_record includes all stat fields (fg_m, fg3_m, etc.) that
    the Elo engine history does not store.
    """
    index: dict[str, list[dict]] = defaultdict(list)
    for g in sorted(games, key=lambda x: x.get("game_date", "")):
        game_date = g.get("game_date", "")
        for p in g.get("players", []):
            pid = p.get("player_id", "")
            mins = p.get("min", 0)
            if not pid or mins < 5:
                continue
            rec = dict(p)
            rec["game_date"] = game_date
            index[pid].append(rec)
    return index


# ── Stats computation ─────────────────────────────────────────────────────── #

def compute_player_stats(engine: PlayerEloEngine, player_id: str,
                         box_index: dict[str, list[dict]]) -> dict | None:
    """Compute full stat profile for a single player."""
    hist = engine.player_history(player_id)
    boxes = box_index.get(player_id, [])
    if not hist or not boxes:
        return None

    gp = len(hist)
    name = engine.names.get(player_id, "?")
    team = engine.teams.get(player_id, "?")
    pos = engine.positions.get(player_id, "?")
    elo = engine.rating(player_id)

    # Per-game averages from engine history
    def avg(key: str) -> float:
        vals = [r.get(key, 0) for r in hist]
        return sum(vals) / len(vals) if vals else 0.0

    avg_min = avg("min")
    avg_pts = avg("pts")
    avg_reb = avg("reb")
    avg_ast = avg("ast")
    avg_stl = avg("stl")
    avg_blk = avg("blk")
    avg_to  = avg("to")
    avg_pf  = avg("pf")
    avg_dreb = avg("dreb")
    avg_gs  = avg("game_score")
    avg_off = avg("off_score")

    # Per-36 normalization
    scale = (36.0 / avg_min) if avg_min > 0 else 0.0
    p36_pts  = avg_pts * scale
    p36_reb  = avg_reb * scale
    p36_ast  = avg_ast * scale
    p36_stl  = avg_stl * scale
    p36_blk  = avg_blk * scale
    p36_to   = avg_to * scale
    p36_pf   = avg_pf * scale
    p36_dreb = avg_dreb * scale

    # Shooting splits from boxscore data
    tot_fg_m  = sum(b.get("fg_m", 0) for b in boxes)
    tot_fg_a  = sum(b.get("fg_a", 0) for b in boxes)
    tot_fg3_m = sum(b.get("fg3_m", 0) for b in boxes)
    tot_fg3_a = sum(b.get("fg3_a", 0) for b in boxes)
    tot_ft_m  = sum(b.get("ft_m", 0) for b in boxes)
    tot_ft_a  = sum(b.get("ft_a", 0) for b in boxes)
    tot_pts   = sum(b.get("pts", 0) for b in boxes)

    fg_pct  = (tot_fg_m / tot_fg_a * 100) if tot_fg_a > 0 else 0.0
    fg3_pct = (tot_fg3_m / tot_fg3_a * 100) if tot_fg3_a > 0 else 0.0
    ft_pct  = (tot_ft_m / tot_ft_a * 100) if tot_ft_a > 0 else 0.0
    ts_denom = 2 * (tot_fg_a + 0.44 * tot_ft_a)
    ts_pct  = (tot_pts / ts_denom * 100) if ts_denom > 0 else 0.0

    # Efficiency metrics
    off_eff = ts_pct + p36_ast - 1.5 * p36_to
    def_imp = 3 * p36_stl + 2 * p36_blk + 0.3 * p36_dreb - p36_pf

    # Injury impact
    injury_impact = engine.injury_elo_impact(player_id)

    return {
        "player_id": player_id,
        "name": name,
        "team": team,
        "pos": pos,
        "gp": gp,
        "elo": elo,
        "avg_min": avg_min,
        "avg_pts": avg_pts,
        "avg_reb": avg_reb,
        "avg_ast": avg_ast,
        "avg_stl": avg_stl,
        "avg_blk": avg_blk,
        "avg_to": avg_to,
        "avg_pf": avg_pf,
        "avg_dreb": avg_dreb,
        "avg_gs": avg_gs,
        "avg_off": avg_off,
        "p36_pts": p36_pts,
        "p36_reb": p36_reb,
        "p36_ast": p36_ast,
        "p36_stl": p36_stl,
        "p36_blk": p36_blk,
        "p36_to": p36_to,
        "p36_pf": p36_pf,
        "p36_dreb": p36_dreb,
        "fg_pct": fg_pct,
        "fg3_pct": fg3_pct,
        "ft_pct": ft_pct,
        "ts_pct": ts_pct,
        "off_eff": off_eff,
        "def_imp": def_imp,
        "injury_impact": injury_impact,
    }


def compute_all_rankings(engine: PlayerEloEngine, box_index: dict[str, list[dict]],
                         min_games: int) -> dict[str, dict]:
    """Compute stats for all qualifying players and return {player_id: stats_dict}."""
    game_counts = engine.player_game_counts()
    qualifying = [pid for pid, cnt in game_counts.items() if cnt >= min_games]

    all_stats: dict[str, dict] = {}
    for pid in qualifying:
        s = compute_player_stats(engine, pid, box_index)
        if s:
            all_stats[pid] = s

    # Compute ranks for key metrics
    for metric, key, reverse in [
        ("elo_rank", "elo", True),
        ("off_eff_rank", "off_eff", True),
        ("def_imp_rank", "def_imp", True),
        ("gs_rank", "avg_gs", True),
        ("ppg_rank", "avg_pts", True),
    ]:
        sorted_pids = sorted(all_stats.keys(), key=lambda p: all_stats[p][key], reverse=reverse)
        for rank, pid in enumerate(sorted_pids, 1):
            all_stats[pid][metric] = rank

    total = len(all_stats)
    for pid in all_stats:
        all_stats[pid]["elo_pctile"] = round(
            (1 - (all_stats[pid]["elo_rank"] - 1) / max(total - 1, 1)) * 100, 1
        )
        all_stats[pid]["total_ranked"] = total

    return all_stats


# ── Player search ─────────────────────────────────────────────────────────── #

def find_player(engine: PlayerEloEngine, query: str) -> list[str]:
    """Fuzzy search: case-insensitive substring match on player name."""
    q = query.lower().strip()
    matches = []
    for pid, name in engine.names.items():
        if q in name.lower():
            matches.append(pid)
    return matches


def find_team(query: str, games: list[dict]) -> str | None:
    """Fuzzy match a team name from game data. Returns team_id or None."""
    q = query.lower().strip()
    for g in games:
        if q in g.get("home_name", "").lower():
            return g.get("home_id")
        if q in g.get("away_name", "").lower():
            return g.get("away_id")
    return None


def find_team_players(engine: PlayerEloEngine, team_query: str,
                      games: list[dict], min_games: int) -> tuple[str, list[str]]:
    """Find all players on a team matching the query. Returns (team_id, [player_ids])."""
    q = team_query.lower().strip()

    # Build team_id -> team_name mapping from raw games
    team_names: dict[str, str] = {}
    for g in games:
        team_names[g.get("home_id", "")] = g.get("home_name", "")
        team_names[g.get("away_id", "")] = g.get("away_name", "")

    # Fuzzy match team
    matched_tid = None
    for tid, tname in team_names.items():
        if q in tname.lower() or q == tid:
            matched_tid = tid
            break

    if not matched_tid:
        return ("", [])

    game_counts = engine.player_game_counts()
    pids = [
        pid for pid in engine.ratings
        if engine.teams.get(pid) == matched_tid
        and game_counts.get(pid, 0) >= min_games
    ]
    return (matched_tid, pids)


# ── Display: single player profile ───────────────────────────────────────── #

def print_profile(engine: PlayerEloEngine, player_id: str,
                  all_stats: dict[str, dict],
                  games: list[dict]) -> None:
    """Print full player profile card."""
    s = all_stats.get(player_id)
    if not s:
        print(c("  Player stats not available.", RED))
        return

    # Resolve team name from raw games
    team_names: dict[str, str] = {}
    for g in games:
        team_names[g.get("home_id", "")] = g.get("home_name", "")
        team_names[g.get("away_id", "")] = g.get("away_name", "")
    team_display = team_names.get(s["team"], s["team"])

    w = 72

    # Header
    print()
    print(c("━" * w, CYAN))
    print(c(f"  {s['name']}", BOLD))
    print(f"  {s['pos']}  |  {team_display}  |  {s['gp']} games  |  {s['avg_min']:.1f} min/g")
    print(c("━" * w, CYAN))

    # Elo
    print()
    print(c("  ELO RATING", BOLD))
    print(f"  Rating:     {c(f'{s['elo']:.1f}', CYAN)}")
    print(f"  Rank:       #{s['elo_rank']} of {s['total_ranked']}  ({s['elo_pctile']:.0f}th percentile)")
    print()

    # Per-game averages
    print(c("  PER-GAME AVERAGES", BOLD))
    print(f"  PTS    REB    AST    STL    BLK    TO     MIN")
    print(f"  {s['avg_pts']:5.1f}  {s['avg_reb']:5.1f}  {s['avg_ast']:5.1f}  "
          f"{s['avg_stl']:5.1f}  {s['avg_blk']:5.1f}  {s['avg_to']:5.1f}  {s['avg_min']:5.1f}")
    print()

    # Per-36
    print(c("  PER-36 MINUTE STATS", BOLD))
    print(f"  PTS    REB    AST    STL    BLK    TO")
    print(f"  {s['p36_pts']:5.1f}  {s['p36_reb']:5.1f}  {s['p36_ast']:5.1f}  "
          f"{s['p36_stl']:5.1f}  {s['p36_blk']:5.1f}  {s['p36_to']:5.1f}")
    print()

    # Shooting splits
    print(c("  SHOOTING SPLITS", BOLD))
    print(f"  FG%     3PT%    FT%     TS%")
    print(f"  {s['fg_pct']:5.1f}   {s['fg3_pct']:5.1f}   {s['ft_pct']:5.1f}   {s['ts_pct']:5.1f}")
    print()

    # Efficiency
    print(c("  EFFICIENCY METRICS", BOLD))
    oe_color = GREEN if s["off_eff"] > 55 else YELLOW if s["off_eff"] > 45 else RESET
    di_color = GREEN if s["def_imp"] > 4 else YELLOW if s["def_imp"] > 2 else RESET
    gs_color = GREEN if s["avg_gs"] > 12 else YELLOW if s["avg_gs"] > 8 else RESET
    print(f"  Offensive Efficiency:  {c(f'{s['off_eff']:.1f}', oe_color)}"
          f"   (rank #{s['off_eff_rank']})")
    print(f"  Defensive Impact:      {c(f'{s['def_imp']:.1f}', di_color)}"
          f"   (rank #{s['def_imp_rank']})")
    print(f"  Hollinger Game Score:  {c(f'{s['avg_gs']:.1f}', gs_color)}"
          f"   (rank #{s['gs_rank']})")
    print(f"  Off Score (avg):       {s['avg_off']:.1f}")
    print()

    # Trend: last 5 vs season
    hist = engine.player_history(player_id)
    if len(hist) >= 6:
        print(c("  TREND (last 5 vs season)", BOLD))
        last5 = hist[-5:]
        l5_pts = sum(r["pts"] for r in last5) / 5
        l5_reb = sum(r["reb"] for r in last5) / 5
        l5_ast = sum(r["ast"] for r in last5) / 5
        l5_gs  = sum(r["game_score"] for r in last5) / 5

        def trend_arrow(recent: float, season: float) -> str:
            diff = recent - season
            pct = (diff / season * 100) if season != 0 else 0
            if pct > 10:
                return c(f"  +{pct:.0f}%", GREEN)
            elif pct < -10:
                return c(f"  {pct:.0f}%", RED)
            else:
                return c(f"  {pct:+.0f}%", DIM)

        print(f"  {'':12s}  {'Last 5':>7s}  {'Season':>7s}  {'Trend':>8s}")
        print(f"  {'PTS':12s}  {l5_pts:7.1f}  {s['avg_pts']:7.1f}  {trend_arrow(l5_pts, s['avg_pts'])}")
        print(f"  {'REB':12s}  {l5_reb:7.1f}  {s['avg_reb']:7.1f}  {trend_arrow(l5_reb, s['avg_reb'])}")
        print(f"  {'AST':12s}  {l5_ast:7.1f}  {s['avg_ast']:7.1f}  {trend_arrow(l5_ast, s['avg_ast'])}")
        print(f"  {'Game Score':12s}  {l5_gs:7.1f}  {s['avg_gs']:7.1f}  {trend_arrow(l5_gs, s['avg_gs'])}")

        overall = "IMPROVING" if l5_gs > s["avg_gs"] * 1.05 else \
                  "DECLINING" if l5_gs < s["avg_gs"] * 0.95 else "STEADY"
        overall_color = GREEN if overall == "IMPROVING" else RED if overall == "DECLINING" else YELLOW
        print(f"  Overall: {c(overall, overall_color + BOLD)}")
        print()

    # Prop lines
    print(c("  PROP LINES (last 10 games)", BOLD))
    common_lines = {
        "pts": [10.5, 12.5, 14.5, 16.5, 18.5, 20.5],
        "reb": [3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        "ast": [2.5, 3.5, 4.5, 5.5],
    }
    for stat, lines in common_lines.items():
        prop = engine.prop_edge(player_id, stat, line=0, last_n=10)
        if "error" in prop:
            continue
        vals = prop["last_values"]
        median = sorted(vals)[len(vals) // 2] if vals else 0
        mean = prop["mean"]

        # Find best line (closest to median from the common lines)
        best_line = min(lines, key=lambda x: abs(x - median))
        edge_data = engine.prop_edge(player_id, stat, line=best_line, last_n=10)

        stat_upper = stat.upper()
        val_str = " ".join(f"{v:>3}" for v in vals[-10:])
        print(f"  {stat_upper:4s}  Last 10: [{val_str}]")
        print(f"  {'':4s}  Median: {median}   Mean: {mean:.1f}   StdDev: {edge_data['std_dev']}")

        # Show edges for lines near the player's median
        relevant = [l for l in lines if abs(l - median) <= 4]
        if not relevant:
            relevant = [best_line]
        edge_parts = []
        for line in relevant:
            e = engine.prop_edge(player_id, stat, line=line, last_n=10)
            edge_pct = e["edge"] * 100
            p_over_pct = e["p_over"] * 100
            edge_color = GREEN if edge_pct > 3 else RED if edge_pct < -3 else DIM
            edge_parts.append(
                f"O/U {line}: {c(f'{edge_pct:+.1f}%', edge_color)} ({p_over_pct:.0f}% over)"
            )
        print(f"  {'':4s}  " + "   ".join(edge_parts))
        print()

    # Injury impact
    print(c("  INJURY IMPACT", BOLD))
    impact = s["injury_impact"]
    if impact > 0:
        impact_color = RED if impact > 40 else YELLOW if impact > 15 else DIM
        print(f"  If {s['name']} were injured:")
        print(f"  Estimated team Elo loss: {c(f'-{impact:.1f}', impact_color)} points")
        severity = "CRITICAL" if impact > 60 else "HIGH" if impact > 30 else \
                   "MODERATE" if impact > 10 else "LOW"
        sev_color = RED if severity in ("CRITICAL", "HIGH") else \
                    YELLOW if severity == "MODERATE" else DIM
        print(f"  Severity: {c(severity, sev_color + BOLD)}")
    else:
        print(f"  Minimal impact (below-average or low-minutes player)")

    print()
    print(c("━" * w, CYAN))
    print()


# ── Display: leaderboard ─────────────────────────────────────────────────── #

SORT_KEYS = {
    "elo":       ("elo", True),
    "off_eff":   ("off_eff", True),
    "def_imp":   ("def_imp", True),
    "ppg":       ("avg_pts", True),
    "game_score":("avg_gs", True),
}


def print_leaderboard(all_stats: dict[str, dict], top_n: int, sort_by: str,
                      team_filter: str | None, pos_filter: str | None,
                      games: list[dict]) -> None:
    """Print leaderboard table."""
    # Resolve team names
    team_names: dict[str, str] = {}
    for g in games:
        team_names[g.get("home_id", "")] = g.get("home_name", "")
        team_names[g.get("away_id", "")] = g.get("away_name", "")

    # Filter
    players = list(all_stats.values())
    if team_filter:
        tq = team_filter.lower()
        players = [p for p in players
                   if tq in team_names.get(p["team"], "").lower() or tq == p["team"]]
    if pos_filter:
        players = [p for p in players if p["pos"] == pos_filter.upper()]

    # Sort
    sort_key, sort_rev = SORT_KEYS.get(sort_by, ("elo", True))
    players.sort(key=lambda p: p[sort_key], reverse=sort_rev)
    players = players[:top_n]

    if not players:
        print(c("  No players match the filters.", YELLOW))
        return

    print()
    sort_label = sort_by.upper().replace("_", " ")
    print(c(f"  PLAYER LEADERBOARD — Top {len(players)} by {sort_label}", BOLD))
    print()

    headers = ["#", "Name", "Team", "Pos", "GP", "PPG", "RPG", "APG",
               "Elo", "Off Eff", "Def Imp", "GmSc"]
    aligns = ["r", "l", "l", "l", "r", "r", "r", "r", "r", "r", "r", "r"]
    rows = []
    for i, p in enumerate(players, 1):
        tname = team_names.get(p["team"]) or p["team"]
        # Truncate long team names
        if len(tname) > 18:
            tname = tname[:17] + "."
        rows.append([
            str(i),
            p["name"][:24],
            tname,
            p["pos"],
            str(p["gp"]),
            f"{p['avg_pts']:.1f}",
            f"{p['avg_reb']:.1f}",
            f"{p['avg_ast']:.1f}",
            f"{p['elo']:.0f}",
            f"{p['off_eff']:.1f}",
            f"{p['def_imp']:.1f}",
            f"{p['avg_gs']:.1f}",
        ])

    print_table(headers, rows, aligns)
    print()


# ── Display: team roster ──────────────────────────────────────────────────── #

def print_team_roster(team_id: str, player_ids: list[str],
                      all_stats: dict[str, dict], games: list[dict]) -> None:
    """Print all players on a team as a mini-leaderboard."""
    team_names: dict[str, str] = {}
    for g in games:
        team_names[g.get("home_id", "")] = g.get("home_name", "")
        team_names[g.get("away_id", "")] = g.get("away_name", "")
    team_display = team_names.get(team_id, team_id)

    players = [all_stats[pid] for pid in player_ids if pid in all_stats]
    players.sort(key=lambda p: p["elo"], reverse=True)

    if not players:
        print(c(f"  No qualifying players found for {team_display}.", YELLOW))
        return

    print()
    print(c(f"  ROSTER — {team_display}  ({len(players)} players)", BOLD))
    print()

    headers = ["#", "Name", "Pos", "GP", "MIN", "PPG", "RPG", "APG",
               "FG%", "TS%", "Elo", "Off Eff", "Def Imp", "GmSc", "Inj Impact"]
    aligns = ["r", "l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"]
    rows = []
    for i, p in enumerate(players, 1):
        rows.append([
            str(i),
            p["name"][:22],
            p["pos"],
            str(p["gp"]),
            f"{p['avg_min']:.1f}",
            f"{p['avg_pts']:.1f}",
            f"{p['avg_reb']:.1f}",
            f"{p['avg_ast']:.1f}",
            f"{p['fg_pct']:.1f}",
            f"{p['ts_pct']:.1f}",
            f"{p['elo']:.0f}",
            f"{p['off_eff']:.1f}",
            f"{p['def_imp']:.1f}",
            f"{p['avg_gs']:.1f}",
            f"{p['injury_impact']:.1f}",
        ])

    print_table(headers, rows, aligns)
    print()


# ── Main ──────────────────────────────────────────────────────────────────── #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChalkIQ Player Profile — recruiter + bettor overview",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/player_profile.py "Johni Broome"
  python scripts/player_profile.py --top 50
  python scripts/player_profile.py --top 20 --sort ppg
  python scripts/player_profile.py --team "Duke"
  python scripts/player_profile.py --position G --top 30
  python scripts/player_profile.py --division womens "Paige Bueckers"
""",
    )
    parser.add_argument("player", nargs="?", default=None,
                        help="Player name to search (fuzzy match)")
    parser.add_argument("--division", default="mens", choices=["mens", "womens"],
                        help="Division (default: mens)")
    parser.add_argument("--top", type=int, default=None,
                        help="Leaderboard mode: show top N players")
    parser.add_argument("--team", type=str, default=None,
                        help="Show all players on a team (fuzzy match)")
    parser.add_argument("--position", type=str, default=None, choices=["G", "F", "C"],
                        help="Filter by position")
    parser.add_argument("--min-games", type=int, default=5,
                        help="Minimum games played (default: 5)")
    parser.add_argument("--sort", type=str, default="elo",
                        choices=list(SORT_KEYS.keys()),
                        help="Sort leaderboard by (default: elo)")
    args = parser.parse_args()

    # Must specify at least one mode
    if not args.player and args.top is None and not args.team:
        parser.print_help()
        sys.exit(1)

    # Load data
    engine, games = load_engine_and_boxscores(args.division)
    box_index = build_player_boxscore_index(games)

    print(c("  Computing player stats and rankings...", DIM), end="\r", flush=True)
    all_stats = compute_all_rankings(engine, box_index, args.min_games)
    print(f"  Ranked {len(all_stats)} players (min {args.min_games} games)" + " " * 20)

    # ── Team mode ─────────────────────────────────────────────────────────── #
    if args.team:
        team_id, player_ids = find_team_players(engine, args.team, games, args.min_games)
        if not player_ids:
            print(c(f"  No team found matching '{args.team}'", RED))
            sys.exit(1)
        if args.position:
            player_ids = [pid for pid in player_ids
                          if engine.positions.get(pid) == args.position.upper()]
        print_team_roster(team_id, player_ids, all_stats, games)
        return

    # ── Leaderboard mode ──────────────────────────────────────────────────── #
    if args.top is not None:
        print_leaderboard(all_stats, args.top, args.sort,
                          team_filter=None, pos_filter=args.position,
                          games=games)
        return

    # ── Player profile mode ───────────────────────────────────────────────── #
    if args.player:
        matches = find_player(engine, args.player)
        # Filter to those in all_stats (meeting min-games)
        matches = [pid for pid in matches if pid in all_stats]

        if not matches:
            print(c(f"  No player found matching '{args.player}' "
                     f"(with >= {args.min_games} games)", RED))
            # Suggest close matches without min-games filter
            all_matches = find_player(engine, args.player)
            if all_matches:
                game_counts = engine.player_game_counts()
                print(c("  Possible matches (below min-games threshold):", DIM))
                for pid in all_matches[:5]:
                    gc = game_counts.get(pid, 0)
                    print(f"    {engine.names.get(pid, '?')} ({gc} games)")
            sys.exit(1)

        if len(matches) == 1:
            print_profile(engine, matches[0], all_stats, games)
        else:
            # Multiple matches — show disambiguation then profile for best match
            # Prefer exact match first
            exact = [pid for pid in matches
                     if engine.names.get(pid, "").lower() == args.player.lower()]
            if len(exact) == 1:
                print_profile(engine, exact[0], all_stats, games)
                return

            # Sort by Elo rating (most notable player first)
            matches.sort(key=lambda pid: engine.rating(pid), reverse=True)

            if len(matches) <= 3:
                # Show profiles for all
                for pid in matches:
                    print_profile(engine, pid, all_stats, games)
            else:
                print(c(f"  Multiple players match '{args.player}' — showing top 10:", YELLOW))
                print()
                for i, pid in enumerate(matches[:10], 1):
                    s = all_stats[pid]
                    team_names: dict[str, str] = {}
                    for g in games:
                        team_names[g.get("home_id", "")] = g.get("home_name", "")
                        team_names[g.get("away_id", "")] = g.get("away_name", "")
                    tname = team_names.get(s["team"], s["team"])
                    print(f"  {i:2d}. {s['name']:30s}  {s['pos']:3s}  {tname:20s}  "
                          f"Elo: {s['elo']:.0f}  PPG: {s['avg_pts']:.1f}")
                print()
                print(f"  Tip: use a more specific name to narrow results.")


if __name__ == "__main__":
    main()
