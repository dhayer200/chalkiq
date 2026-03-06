"""
Adjusted offensive/defensive efficiency + pace.

Pace proxy: average total points per game (home + away).
High-scoring games = fast pace. No box score data needed.

Efficiency is normalized to pace so teams are compared on a
per-pace basis — a team scoring 80 in a 60-pace game is more
efficient than one scoring 80 in an 80-pace game.

Methodology (KenPom-style iterative adjustment, GP-weighted):
  - Raw pace   = avg total points per game
  - Pace-adj off/def = raw off/def * (league_avg_pace / team_avg_pace)
    (normalizes scoring to a neutral pace environment)
  - Iterative opponent adjustment (20 passes, GP-weighted) on top of that
  - Renormalized each pass to prevent divergence
"""

from __future__ import annotations


def compute_efficiency(
    history: list[dict],
    n_iter: int = 20,
    min_games: int = 5,
) -> dict[str, dict]:
    """
    Returns dict[team_id, {gp, pace, raw_off, raw_def, adj_off, adj_def, net_adj}]

    pace     — avg total points per game (proxy for tempo)
    adj_off  — pace-adjusted, opponent-adjusted offensive efficiency
    adj_def  — pace-adjusted, opponent-adjusted defensive efficiency
    net_adj  — adj_off - adj_def
    """
    # Build per-team game list: (pts_scored, pts_allowed, total_pts, opponent_id)
    team_games: dict[str, list[tuple[float, float, float, str]]] = {}
    for g in history:
        h    = g["home_id"]
        a    = g["away_id"]
        hs   = float(g["home_score"])
        as_  = float(g["away_score"])
        tot  = hs + as_
        team_games.setdefault(h, []).append((hs, as_, tot, a))
        team_games.setdefault(a, []).append((as_, hs, tot, h))

    teams = [t for t, gs in team_games.items() if len(gs) >= min_games]
    if not teams:
        return {}

    gp: dict[str, int] = {t: len(team_games[t]) for t in teams}

    # Raw averages
    raw_off: dict[str, float] = {
        t: sum(g[0] for g in team_games[t]) / gp[t] for t in teams
    }
    raw_def: dict[str, float] = {
        t: sum(g[1] for g in team_games[t]) / gp[t] for t in teams
    }
    raw_pace: dict[str, float] = {
        t: sum(g[2] for g in team_games[t]) / gp[t] for t in teams
    }

    league_avg_pace = sum(raw_pace.values()) / len(raw_pace)
    league_avg_off  = sum(raw_off.values())  / len(raw_off)

    # Pace-adjust: normalize each team's scoring to league-average pace.
    # A slow team (low pace) gets their offense scaled up proportionally.
    pace_adj_off: dict[str, float] = {
        t: raw_off[t] * (league_avg_pace / raw_pace[t]) if raw_pace[t] else raw_off[t]
        for t in teams
    }
    pace_adj_def: dict[str, float] = {
        t: raw_def[t] * (league_avg_pace / raw_pace[t]) if raw_pace[t] else raw_def[t]
        for t in teams
    }

    # Iterative opponent-strength adjustment (GP-weighted)
    adj_off = dict(pace_adj_off)
    adj_def = dict(pace_adj_def)

    for _ in range(n_iter):
        new_off: dict[str, float] = {}
        new_def: dict[str, float] = {}
        for t in teams:
            valid_games = [(g[0], g[1], g[3]) for g in team_games[t] if g[3] in adj_def]
            if not valid_games:
                new_off[t] = pace_adj_off[t]
                new_def[t] = pace_adj_def[t]
                continue

            total_w     = sum(gp[opp] for _, _, opp in valid_games)
            opp_avg_def = sum(adj_def[opp] * gp[opp] for _, _, opp in valid_games) / total_w
            opp_avg_off = sum(adj_off[opp] * gp[opp] for _, _, opp in valid_games) / total_w

            new_off[t] = pace_adj_off[t] * (league_avg_off / opp_avg_def) if opp_avg_def else pace_adj_off[t]
            new_def[t] = pace_adj_def[t] * (league_avg_off / opp_avg_off) if opp_avg_off else pace_adj_def[t]

        # Renormalize each pass to prevent explosion
        off_mean = sum(new_off.values()) / len(new_off)
        def_mean = sum(new_def.values()) / len(new_def)
        adj_off = {t: v * league_avg_off / off_mean for t, v in new_off.items()}
        adj_def = {t: v * league_avg_off / def_mean for t, v in new_def.items()}

    return {
        t: {
            "gp":      gp[t],
            "pace":    round(raw_pace[t], 1),
            "raw_off": round(raw_off[t], 1),
            "raw_def": round(raw_def[t], 1),
            "adj_off": round(adj_off[t], 1),
            "adj_def": round(adj_def[t], 1),
            "net_adj": round(adj_off[t] - adj_def[t], 1),
        }
        for t in teams
    }
