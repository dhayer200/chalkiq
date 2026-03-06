"""
Extract per-team Elo rating histories from engine.history.

Each game in engine.history records the rating *after* the update, so we
can reconstruct the full price-like trajectory for any team.
"""

from __future__ import annotations
import statistics


def build_rating_histories(
    history: list[dict],
) -> dict[str, list[tuple[str, float]]]:
    """
    Returns {team_id: [(date, rating_after), ...]} sorted by date.
    Starting point (1500) is prepended using the season's first game date.
    """
    INITIAL = 1500.0
    team_hist: dict[str, list[tuple[str, float]]] = {}

    for g in history:
        d = g["date"] or ""
        team_hist.setdefault(g["home_id"], []).append((d, g["home_rating_after"]))
        team_hist.setdefault(g["away_id"], []).append((d, g["away_rating_after"]))

    result: dict[str, list[tuple[str, float]]] = {}
    for tid, pts in team_hist.items():
        pts_sorted = sorted(pts, key=lambda x: x[0])
        # Prepend the starting rating so the chart begins at 1500
        if pts_sorted:
            result[tid] = [(pts_sorted[0][0], INITIAL)] + pts_sorted
    return result


def market_movers(
    histories: dict[str, list[tuple[str, float]]],
    names: dict[str, str],
    window_days: int = 30,
    top_n: int = 10,
) -> tuple[list[dict], list[dict]]:
    """
    Return (gainers, losers) — each a list of dicts sorted by rating change
    over the last `window_days` calendar days.
    """
    from datetime import date, timedelta

    cutoff = (date.today() - timedelta(days=window_days)).isoformat()
    rows: list[dict] = []

    for tid, pts in histories.items():
        if len(pts) < 2:
            continue
        current = pts[-1][1]
        # Find the rating closest to (but before) the cutoff
        past_pts = [r for d, r in pts if d <= cutoff]
        past = past_pts[-1] if past_pts else pts[0][1]
        change = current - past
        rows.append({
            "team_id": tid,
            "name":    names.get(tid, tid),
            "current": round(current, 1),
            "past":    round(past, 1),
            "change":  round(change, 1),
        })

    rows.sort(key=lambda x: x["change"], reverse=True)
    return rows[:top_n], rows[-top_n:][::-1]


def ncaa361_spread(histories: dict[str, list[tuple[str, float]]]) -> list[tuple[str, float]]:
    """
    Daily rating spread (std dev across all teams) — the market's volatility index.
    Returns [(date_str, spread), ...] sorted by date.
    Uses the last known rating for each team on each date seen.
    """
    from collections import defaultdict

    # For each date, gather each team's most recent rating on or before that date
    all_dates = sorted({d for pts in histories.values() for d, _ in pts if d})
    if not all_dates:
        return []

    # Last known rating per team up to each date
    spread_series: list[tuple[str, float]] = []
    last_known: dict[str, float] = {}
    date_to_games: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for tid, pts in histories.items():
        for d, r in pts:
            if d:
                date_to_games[d].append((tid, r))

    for d in all_dates:
        for tid, r in date_to_games[d]:
            last_known[tid] = r
        if len(last_known) >= 10:
            spread_series.append((d, round(statistics.stdev(last_known.values()), 1)))

    return spread_series
