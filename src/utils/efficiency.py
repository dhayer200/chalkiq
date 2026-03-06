"""
Adjusted offensive and defensive efficiency.

Methodology (KenPom-style iterative adjustment, games-played weighted):
  - Raw OE  = points scored per game
  - Raw DE  = points allowed per game
  - Adjusted OE = raw OE scaled by (league avg DE / weighted avg DE of opponents faced)
  - Adjusted DE = raw DE scaled by (league avg OE / weighted avg OE of opponents faced)
  - Opponent averages are weighted by that opponent's games played — teams with more
    data pull more weight, reducing noise from small-sample opponents.
  - Run N iterations so the adjustments converge, renormalizing each pass.

Only requires game history (home_id, away_id, home_score, away_score).
No possession data needed — uses points per game as the efficiency proxy.
"""

from __future__ import annotations


def compute_efficiency(
    history: list[dict],
    n_iter: int = 20,
    min_games: int = 5,
) -> dict[str, dict]:
    """
    Compute adjusted offensive/defensive efficiency for every team.

    Parameters
    ----------
    history   : list of game dicts from EloEngine.history
                (keys: home_id, away_id, home_score, away_score)
    n_iter    : number of adjustment iterations (20 is plenty for convergence)
    min_games : teams with fewer games are excluded

    Returns
    -------
    dict[team_id, {gp, raw_off, raw_def, adj_off, adj_def, net_adj}]
    """
    # Build per-team game list: (points_scored, points_allowed, opponent_id)
    team_games: dict[str, list[tuple[float, float, str]]] = {}
    for g in history:
        h   = g["home_id"]
        a   = g["away_id"]
        hs  = float(g["home_score"])
        as_ = float(g["away_score"])
        team_games.setdefault(h, []).append((hs, as_, a))
        team_games.setdefault(a, []).append((as_, hs, h))

    # Filter teams with enough games
    teams = [t for t, gs in team_games.items() if len(gs) >= min_games]
    if not teams:
        return {}

    # Games-played count used as weight (more games = more reliable data)
    gp: dict[str, int] = {t: len(team_games[t]) for t in teams}

    # Raw averages (unweighted — each game counts equally within a team's own record)
    raw_off: dict[str, float] = {
        t: sum(g[0] for g in team_games[t]) / gp[t]
        for t in teams
    }
    raw_def: dict[str, float] = {
        t: sum(g[1] for g in team_games[t]) / gp[t]
        for t in teams
    }

    # League average anchors the scale
    league_avg = sum(raw_off.values()) / len(raw_off)

    adj_off = dict(raw_off)
    adj_def = dict(raw_def)

    # Iterative adjustment with games-played weighting + normalization
    for _ in range(n_iter):
        new_off: dict[str, float] = {}
        new_def: dict[str, float] = {}
        for t in teams:
            games = team_games[t]
            # Only use opponents that passed the min_games filter
            valid = [(pts, opp_pts, opp) for pts, opp_pts, opp in games if opp in adj_def]
            if not valid:
                new_off[t] = raw_off[t]
                new_def[t] = raw_def[t]
                continue

            # Weight each opponent by their games played — more data = more trust
            total_w = sum(gp[opp] for _, _, opp in valid)
            opp_avg_def = sum(adj_def[opp] * gp[opp] for _, _, opp in valid) / total_w
            opp_avg_off = sum(adj_off[opp] * gp[opp] for _, _, opp in valid) / total_w

            new_off[t] = raw_off[t] * (league_avg / opp_avg_def) if opp_avg_def else raw_off[t]
            new_def[t] = raw_def[t] * (league_avg / opp_avg_off) if opp_avg_off else raw_def[t]

        # Renormalize to league_avg each iteration to prevent explosion
        off_mean = sum(new_off.values()) / len(new_off)
        def_mean = sum(new_def.values()) / len(new_def)
        adj_off = {t: v * league_avg / off_mean for t, v in new_off.items()}
        adj_def = {t: v * league_avg / def_mean for t, v in new_def.items()}

    return {
        t: {
            "gp":      gp[t],
            "raw_off": round(raw_off[t], 1),
            "raw_def": round(raw_def[t], 1),
            "adj_off": round(adj_off[t], 1),
            "adj_def": round(adj_def[t], 1),
            "net_adj": round(adj_off[t] - adj_def[t], 1),
        }
        for t in teams
    }
