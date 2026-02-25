"""
ChalkIQ — the favorites win
==============================
Run from the project root:
    streamlit run frontend/app.py
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import html as _html
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.bracket.simulator import round_advancement_odds
from src.bracket.structure import (
    BRACKET_SLOT_ORDER,
    REGIONS,
    ROUND_LABELS,
    assign_seeds,
    final_four_order,
    region_bracket_order,
)
from src.live.feed import fetch_live_games
from src.live.model import live_win_prob, upset_alert, prob_swing
from src.predictions.pregame import matchup_prob
from src.ratings.elo import EloEngine
from src.utils.data import fetch_season
from src.utils.metrics import evaluate

# ── Constants ─────────────────────────────────────────────────────────────────

SEASON_START = date(2025, 11, 4)
SEASON_END   = date(2026, 2, 23)

ROOT = Path(__file__).parent.parent   # project root, works regardless of cwd

# Nord colour palette constants
NORD = {
    "bg":      "#2E3440",   # Nord0 — polar night dark
    "bg1":     "#3B4252",   # Nord1
    "bg2":     "#434C5E",   # Nord2
    "bg3":     "#4C566A",   # Nord3  (subtle borders / muted text)
    "snow0":   "#D8DEE9",   # Nord4  (primary text on dark)
    "snow1":   "#E5E9F0",   # Nord5
    "snow2":   "#ECEFF4",   # Nord6  (brightest text)
    "frost0":  "#8FBCBB",   # Nord7  (teal)
    "frost1":  "#88C0D0",   # Nord8  (light blue  — men's accent)
    "frost2":  "#81A1C1",   # Nord9  (blue)
    "frost3":  "#5E81AC",   # Nord10 (dark blue)
    "red":     "#BF616A",   # Nord11
    "orange":  "#D08770",   # Nord12
    "yellow":  "#EBCB8B",   # Nord13
    "green":   "#A3BE8C",   # Nord14
    "purple":  "#B48EAD",   # Nord15 — women's accent
}

DIVISION_CONFIG = {
    "mens": {
        "label":     "Men's",
        "cache_dir": str(ROOT / "data" / "raw" / "mens"),
        "emoji":     "🏀",
        "color":     NORD["frost1"],   # #88C0D0 — light blue
        "light":     NORD["bg1"],      # #3B4252 — dark panel bg
    },
    "womens": {
        "label":     "Women's",
        "cache_dir": str(ROOT / "data" / "raw" / "womens"),
        "emoji":     "🏀",
        "color":     NORD["purple"],   # #B48EAD — aurora purple
        "light":     NORD["bg1"],      # #3B4252
    },
}

N_SIMS = 100_000


# ── Data loading (cached per division) ────────────────────────────────────────

@st.cache_resource(show_spinner="Loading game data…")
def load_engine(division: str) -> EloEngine:
    cfg = DIVISION_CONFIG[division]
    games = fetch_season(
        SEASON_START, SEASON_END,
        cache_dir=cfg["cache_dir"],
        division=division,
        verbose=False,
    )
    engine = EloEngine(k=24.0, home_advantage=100.0)
    engine.process_games(games)
    return engine


@st.cache_data(show_spinner="Simulating bracket…")
def load_bracket_data(division: str):
    engine = load_engine(division)
    rankings = engine.rankings()
    regions  = assign_seeds(rankings)

    # Build bracket in proper seed order: East slots 0-15, West 16-31,
    # South 32-47, Midwest 48-63 — so #1 seed plays #16, not #2.
    bracket_order: list[str] = []
    for region_name in REGIONS:
        bracket_order.extend(region_bracket_order(regions[region_name]))

    adv_odds = round_advancement_odds(
        seeded_teams=bracket_order,
        win_prob_fn=engine.win_prob,
        n_sims=N_SIMS,
        seed=42,
    )
    # Round 6 = championship win probability
    champ_odds = {tid: adv_odds[tid].get(6, 0.0) for tid in bracket_order}

    return regions, adv_odds, champ_odds


@st.cache_data(ttl=120, show_spinner=False)
def load_live_games(division: str) -> list[dict]:
    """Fetch live games, cached for 2 minutes."""
    return fetch_live_games(division=division)


@st.cache_data(ttl=120, show_spinner=False)
def live_bracket_impact(
    division: str,
    home_id: str,
    away_id: str,
    home_wins: bool,
) -> dict[str, float]:
    """
    Simulate bracket odds if home_wins (or away wins) in a live regular-season game.
    Clones the Elo engine, applies the hypothetical result, reruns 5k sims.
    Returns {team_id: championship_probability}.
    """
    base = load_engine(division)

    # Shallow-clone: copy ratings + names, keep same hyper-params
    sim = EloEngine(k=base.k, home_advantage=base.home_advantage)
    sim.ratings = dict(base.ratings)
    sim.names   = dict(base.names)

    # Apply hypothetical result (scores don't affect Elo, only outcome does)
    if home_wins:
        sim.update(home_id, away_id, 70, 60, neutral=True)
    else:
        sim.update(home_id, away_id, 60, 70, neutral=True)

    rankings_sim = sim.rankings()
    regions_sim  = assign_seeds(rankings_sim)
    bracket_order: list[str] = []
    for rn in REGIONS:
        bracket_order.extend(region_bracket_order(regions_sim[rn]))

    adv = round_advancement_odds(
        seeded_teams=bracket_order,
        win_prob_fn=sim.win_prob,
        n_sims=5_000,
        seed=42,
    )
    return {tid: adv[tid].get(6, 0.0) for tid in bracket_order}


# ── Chart helpers ──────────────────────────────────────────────────────────────

def plotly_odds_bar(names: list[str], odds: list[float], color: str) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=odds,
        y=names,
        orientation="h",
        marker_color=color,
        text=[f"{o*100:.1f}%" for o in odds],
        textposition="outside",
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickformat=".0%", title="Championship probability"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=180, r=60, t=10, b=30),
        height=max(300, len(names) * 26),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


def plotly_calibration(cal_bins: list[dict], color: str) -> go.Figure:
    preds = [b["predicted_avg"] for b in cal_bins]
    obs   = [b["observed"]      for b in cal_bins]
    ns    = [b["n"]             for b in cal_bins]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        line=dict(color=NORD["bg3"], dash="dash", width=1),
        name="Perfect calibration",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=preds, y=obs,
        mode="markers+lines",
        marker=dict(size=[max(6, n / 20) for n in ns], color=color, opacity=0.8),
        line=dict(color=color, width=2),
        name="Elo model",
        hovertemplate="Predicted: %{x:.0%}<br>Observed: %{y:.0%}<br><extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(tickformat=".0%", title="Predicted win probability", range=[0, 1]),
        yaxis=dict(tickformat=".0%", title="Observed win rate",         range=[0, 1]),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=20, t=10, b=50),
        height=340,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def draw_final_four(regions, adv_odds, names, color) -> plt.Figure:
    """Draw a Final Four bracket: East/West on left, South/Midwest on right."""
    ff_teams = final_four_order(regions, adv_odds)
    team_names = [names.get(t, t) for t in ff_teams]

    # Canvas: 12 wide × 4 tall — Nord dark background
    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor(NORD["bg"])
    ax.set_facecolor(NORD["bg"])
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)

    BW, BH = 2.5, 0.62   # box width / height
    EDGE   = NORD["bg3"]  # #4C566A
    MID_Y  = 0.48         # y-center of champ box

    def box(x, y, text, highlight=False):
        fc = color if highlight else NORD["bg1"]
        tc = NORD["bg"] if highlight else NORD["snow0"]
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), BW, BH,
            boxstyle="round,pad=0.05",
            facecolor=fc, edgecolor=EDGE, linewidth=0.8, zorder=2,
        ))
        ax.text(x + BW / 2, y + BH / 2, text,
                ha="center", va="center", fontsize=7.5,
                color=tc, fontweight="bold" if highlight else "normal", zorder=3)

    def elbow(x1, y1, x2, y2):
        """L-shaped connector: horizontal then vertical."""
        mx = (x1 + x2) / 2
        ax.plot([x1, mx, mx, x2], [y1, y1, y2, y2],
                color=EDGE, linewidth=1.0, zorder=1)

    # ── Left semifinal: East (0) vs West (1) ────────────────────────────────
    Lx = 0.2                          # left team column
    box(Lx, 2.8, f"({REGIONS[0]})\n{team_names[0][:22]}")
    box(Lx, 1.5, f"({REGIONS[1]})\n{team_names[1][:22]}")

    best_left = max([ff_teams[0], ff_teams[1]],
                    key=lambda t: adv_odds.get(t, {}).get(5, 0))
    LWx = 3.1                         # left-winner column
    LWy = 2.15 - BH / 2              # vertically centered
    box(LWx, LWy, names.get(best_left, best_left)[:24], highlight=True)

    # connect East → winner, West → winner
    elbow(Lx + BW, 2.8 + BH / 2, LWx, LWy + BH / 2)
    elbow(Lx + BW, 1.5 + BH / 2, LWx, LWy + BH / 2)

    # ── Right semifinal: South (2) vs Midwest (3) ────────────────────────────
    Rx = 12 - 0.2 - BW               # right team column
    box(Rx, 2.8, f"({REGIONS[2]})\n{team_names[2][:22]}")
    box(Rx, 1.5, f"({REGIONS[3]})\n{team_names[3][:22]}")

    best_right = max([ff_teams[2], ff_teams[3]],
                     key=lambda t: adv_odds.get(t, {}).get(5, 0))
    RWx = 12 - 3.1 - BW              # right-winner column
    RWy = LWy
    box(RWx, RWy, names.get(best_right, best_right)[:24], highlight=True)

    elbow(Rx, 2.8 + BH / 2, RWx + BW, RWy + BH / 2)
    elbow(Rx, 1.5 + BH / 2, RWx + BW, RWy + BH / 2)

    # ── Championship ──────────────────────────────────────────────────────────
    Cx = 12 / 2 - BW / 2
    Cy = MID_Y
    champ = max([best_left, best_right],
                key=lambda t: adv_odds.get(t, {}).get(6, 0))
    box(Cx, Cy, f"🏆 {names.get(champ, champ)[:24]}", highlight=True)

    # left winner → champ, right winner → champ
    elbow(LWx + BW, LWy + BH / 2, Cx, Cy + BH / 2)
    elbow(RWx,      RWy + BH / 2, Cx + BW, Cy + BH / 2)

    ax.set_title("Projected Final Four & Champion  (Elo Monte Carlo)",
                 fontsize=10, fontweight="bold", pad=8, color=NORD["snow2"])
    plt.tight_layout()
    return fig


# ── Full bracket SVG ──────────────────────────────────────────────────────────

def _bracket_slot_ys(bh: int = 38, pad: int = 22) -> list[float]:
    """Y positions (top edge) for all 16 R64 slots."""
    ys, y = [], float(pad)
    for i in range(16):
        ys.append(y)
        if i % 2 == 0:          # first of a pair
            y += bh + 1
        else:                    # end of a pair — bigger gap between matchups
            gi = i // 2
            y += bh + (20 if gi == 3 else 10)   # group gap at the halfway point
    return ys


def _all_round_ys(slot_ys: list[float], bh: int = 38) -> list[list[float]]:
    """Derive slot-y lists for R32, S16, E8 from the R64 positions."""
    all_ys = [slot_ys]
    for _ in range(3):
        prev = all_ys[-1]
        nxt = [
            (prev[2 * i] + bh / 2 + prev[2 * i + 1] + bh / 2) / 2 - bh / 2
            for i in range(len(prev) // 2)
        ]
        all_ys.append(nxt)
    return all_ys   # lengths: 16, 8, 4, 2


def _project_bracket(seed_map: dict, win_prob_fn) -> list[list[dict]]:
    """Project all 4 rounds for a 16-team region. Returns list of rounds."""
    teams = [
        {"s": s, "tid": seed_map[s][0], "name": seed_map[s][1], "rating": seed_map[s][2]}
        for s in BRACKET_SLOT_ORDER if s in seed_map
    ]
    rounds, current = [], teams
    while len(current) > 1:
        games, nxt = [], []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            p = win_prob_fn(a["tid"], b["tid"])
            w = a if p >= 0.5 else b
            games.append({"a": a, "b": b, "p_a": p, "winner": w})
            nxt.append(w)
        rounds.append(games)
        current = nxt
    return rounds   # [R64×8, R32×4, S16×2, E8×1]


def region_bracket_svg(
    seed_map: dict, win_prob_fn, color: str, mirror: bool = False
) -> str:
    """Generate an SVG string for a full 16-team single-region bracket."""
    BH, BW, HGAP, PAD = 38, 182, 54, 22
    slot_y   = _bracket_slot_ys(BH, PAD)
    all_ys   = _all_round_ys(slot_y, BH)
    rounds   = _project_bracket(seed_map, win_prob_fn)
    RX       = [PAD + r * (BW + HGAP) for r in range(4)]
    if mirror:
        RX = list(reversed(RX))
    RND_LBL  = ["First Round", "Round of 32", "Sweet 16", "Elite Eight"]
    SVG_H    = int(slot_y[-1] + BH + PAD)
    SVG_W    = int(max(RX) + BW + PAD)
    FONT     = "ui-sans-serif,system-ui,Arial,sans-serif"
    LINE_CLR = NORD["bg3"]   # #4C566A — Nord subtle border

    p: list[str] = []
    p.append(f'<rect width="{SVG_W}" height="{SVG_H}" fill="{NORD["bg"]}"/>')

    # Column header labels
    for rx, lbl in zip(RX, RND_LBL):
        cx = rx + BW / 2
        p.append(
            f'<text x="{cx:.0f}" y="14" text-anchor="middle" '
            f'font-size="8" font-weight="700" fill="{NORD["frost2"]}" '
            f'letter-spacing="0.06em" font-family="{FONT}">'
            f'{lbl.upper()}</text>'
        )

    def esc(s: str) -> str:
        return _html.escape(str(s))

    def draw_box(rx: float, y: float, team: dict, p_win: float, is_winner: bool) -> str:
        name   = esc(team["name"][:24])
        seed   = team["s"]
        rating = team["rating"]
        p_str  = f"{p_win:.0%}"
        if is_winner:
            bg, tc, sc = color, NORD["bg"], color
            fw, sub_opacity = "600", "0.85"
        else:
            bg, tc, sc = NORD["bg1"], NORD["bg3"], NORD["bg2"]
            fw, sub_opacity = "400", "1"
        return "\n".join([
            f'<rect x="{rx:.1f}" y="{y:.1f}" width="{BW}" height="{BH}" rx="3" '
            f'fill="{bg}" stroke="{sc}" stroke-width="0.8"/>',
            # seed + name
            f'<text x="{rx+7:.1f}" y="{y+15:.1f}" font-size="10.5" font-weight="{fw}" '
            f'fill="{tc}" font-family="{FONT}">'
            f'<tspan font-size="8.5" opacity="0.7">#{seed} </tspan>{name}</text>',
            # elo (left) and win% (right)
            f'<text x="{rx+7:.1f}" y="{y+29:.1f}" font-size="8.5" opacity="{sub_opacity}" '
            f'fill="{tc}" font-family="{FONT}">{rating:.0f} Elo</text>',
            f'<text x="{rx+BW-5:.1f}" y="{y+29:.1f}" text-anchor="end" '
            f'font-size="9" font-weight="600" opacity="{sub_opacity}" '
            f'fill="{tc}" font-family="{FONT}">{p_str}</text>',
        ])

    def draw_line(x1: float, y1: float, x2: float, y2: float) -> str:
        # Route connectors toward the next-round column, left or right.
        mx = x1 + HGAP / 2 if x2 >= x1 else x1 - HGAP / 2
        return (
            f'<polyline points="{x1:.1f},{y1:.1f} {mx:.1f},{y1:.1f} '
            f'{mx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}" '
            f'fill="none" stroke="{LINE_CLR}" stroke-width="1"/>'
        )

    for r, (rx, games) in enumerate(zip(RX, rounds)):
        ys = all_ys[r]
        for gi, game in enumerate(games):
            a, b   = game["a"], game["b"]
            p_a    = game["p_a"]
            is_a   = game["winner"]["tid"] == a["tid"]
            ya, yb = ys[2 * gi], ys[2 * gi + 1]

            p.append(draw_box(rx, ya, a, p_a,     is_a))
            p.append(draw_box(rx, yb, b, 1 - p_a, not is_a))

            # Elbow connector from winner to next-round slot.
            # In mirrored regions, rounds flow right->left, so anchor on the
            # left edge of current box and right edge of next box.
            if r < 3:
                wy     = ya if is_a else yb
                w_cy   = wy + BH / 2
                next_cy = all_ys[r + 1][gi] + BH / 2
                next_rx = RX[r + 1]
                going_right = next_rx >= rx
                x1 = rx + BW if going_right else rx
                x2 = next_rx if going_right else next_rx + BW
                p.append(draw_line(x1, w_cy, x2, next_cy))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{SVG_W}" height="{SVG_H}">\n'
        + "\n".join(p)
        + "\n</svg>"
    )


# ── Combined bracket HTML (single-canvas, classic layout) ────────────────────

def _draw_region_svg(
    seed_map: dict,
    region_name: str,
    win_prob_fn,
    color: str,
    side: str,
    x_rounds: list[int],
    y_offset: float,
    bh: int,
    bw: int,
    hgap: int,
) -> tuple[list[str], dict, tuple[float, float]]:
    """Render one region on a shared canvas and return champion anchor."""
    line_clr = NORD["bg3"]
    text_clr = NORD["snow1"]
    muted = NORD["snow0"]
    panel = NORD["bg1"]
    font = "ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif"

    slot_y = [y_offset + y for y in _bracket_slot_ys(bh=bh, pad=0)]
    all_ys = _all_round_ys(slot_y, bh=bh)
    rounds = _project_bracket(seed_map, win_prob_fn)
    going_right = side == "left"

    p: list[str] = []
    p.append(
        f'<text x="{x_rounds[2] + bw/2:.1f}" y="{y_offset - 18:.1f}" text-anchor="middle" '
        f'font-size="18" font-weight="800" fill="{NORD["snow2"]}" letter-spacing="0.07em" font-family="{font}">'
        f'{_html.escape(region_name.upper())}</text>'
    )

    def draw_line(x1: float, y1: float, x2: float, y2: float, stroke: str = line_clr, width: float = 1.2) -> str:
        mx = x1 + hgap / 2 if going_right else x1 - hgap / 2
        return (
            f'<polyline points="{x1:.1f},{y1:.1f} {mx:.1f},{y1:.1f} '
            f'{mx:.1f},{y2:.1f} {x2:.1f},{y2:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{width:.1f}"/>'
        )

    def draw_box(rx: float, y: float, team: dict, p_win: float, is_winner: bool) -> str:
        name = _html.escape(str(team["name"])[:22])
        fill = color if is_winner else panel
        tclr = NORD["bg"] if is_winner else text_clr
        stroke = color if is_winner else line_clr
        return "\n".join([
            f'<rect x="{rx:.1f}" y="{y:.1f}" width="{bw}" height="{bh}" rx="2.5" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>',
            f'<text x="{rx+7:.1f}" y="{y+13:.1f}" font-size="9.3" font-weight="700" fill="{tclr}" font-family="{font}">'
            f'{team["s"]}</text>',
            f'<text x="{rx+21:.1f}" y="{y+13:.1f}" font-size="9.3" font-weight="500" fill="{tclr}" font-family="{font}">'
            f'{name}</text>',
            f'<text x="{rx+bw-6:.1f}" y="{y+13:.1f}" text-anchor="end" font-size="8.2" fill="{muted}" font-family="{font}">'
            f'{p_win:.0%}</text>',
        ])

    for r, (rx, games) in enumerate(zip(x_rounds, rounds)):
        ys = all_ys[r]
        for gi, game in enumerate(games):
            a, b = game["a"], game["b"]
            ya, yb = ys[2 * gi], ys[2 * gi + 1]
            is_a = game["winner"]["tid"] == a["tid"]
            a_cy, b_cy = ya + bh / 2, yb + bh / 2

            p.append(draw_box(rx, ya, a, game["p_a"], is_a))
            p.append(draw_box(rx, yb, b, 1 - game["p_a"], not is_a))

            # Draw an explicit matchup junction so head-to-head pairings are clear.
            join_x = (rx + bw + 10) if going_right else (rx - 10)
            edge_x = (rx + bw) if going_right else rx
            p.append(
                f'<polyline points="{edge_x:.1f},{a_cy:.1f} {join_x:.1f},{a_cy:.1f} '
                f'{join_x:.1f},{b_cy:.1f} {edge_x:.1f},{b_cy:.1f}" '
                f'fill="none" stroke="{line_clr}" stroke-width="1.1" opacity="0.95"/>'
            )
            p.append(
                f'<circle cx="{join_x:.1f}" cy="{(a_cy+b_cy)/2:.1f}" r="2.1" fill="{NORD["snow0"]}" opacity="0.8"/>'
            )

            if r < 3:
                wy = ya if is_a else yb
                w_cy = wy + bh / 2
                next_cy = all_ys[r + 1][gi] + bh / 2
                next_rx = x_rounds[r + 1]
                x1 = join_x
                x2 = next_rx if going_right else next_rx + bw
                p.append(draw_line(x1, w_cy, x2, next_cy, stroke=color, width=1.4))

    champion = rounds[-1][0]["winner"]
    champ_y = all_ys[3][0] + bh / 2
    champ_x = x_rounds[3] + (bw if going_right else 0)
    return p, champion, (champ_x, champ_y)


def combined_bracket_html(
    regions: dict, win_prob_fn, color: str, division_label: str, adv_odds: dict
) -> str:
    """Single-canvas bracket: South TL, Midwest BL, East TR, West BR.
    Center shows Final Four matchups with title odds, then Championship.
    """
    region_order = ["South", "Midwest", "East", "West"]
    available = set(regions.keys())
    if not set(region_order).issubset(available):
        region_order = list(REGIONS)

    left_top, left_bottom, right_top, right_bottom = region_order
    # left_top=South (TL), left_bottom=Midwest (BL)
    # right_top=East (TR), right_bottom=West (BR)

    W, H   = 2500, 1180
    BH, BW, HGAP = 22, 190, 34
    LX     = [28, 252, 476, 700]           # South/Midwest flow left → right
    RX     = [2282, 2058, 1834, 1610]      # East/West flow right → left
    top_y, bottom_y = 105, 640

    font     = "ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif"
    line_clr = NORD["bg3"]
    text_clr = NORD["snow2"]

    svg_parts: list[str] = [
        f'<rect width="{W}" height="{H}" fill="{NORD["bg"]}"/>',
    ]

    lt_parts, lt_champ, lt_anchor = _draw_region_svg(
        regions[left_top],    left_top,    win_prob_fn, color, "left",  LX, top_y,    BH, BW, HGAP
    )
    lb_parts, lb_champ, lb_anchor = _draw_region_svg(
        regions[left_bottom], left_bottom, win_prob_fn, color, "left",  LX, bottom_y, BH, BW, HGAP
    )
    rt_parts, rt_champ, rt_anchor = _draw_region_svg(
        regions[right_top],   right_top,   win_prob_fn, color, "right", RX, top_y,    BH, BW, HGAP
    )
    rb_parts, rb_champ, rb_anchor = _draw_region_svg(
        regions[right_bottom],right_bottom,win_prob_fn, color, "right", RX, bottom_y, BH, BW, HGAP
    )
    svg_parts.extend(lt_parts + lb_parts + rt_parts + rb_parts)

    # ── Final Four center section ──────────────────────────────────────────────
    # Layout (left to right):
    #   E8(890) → lff boxes (915–1100) → join(1122) → champ(1145–1355) ← join(1377) ← rff boxes(1400–1585) ← E8(1610)
    FF_W, FF_H, FF_GAP = 185, 32, 14
    CY = H / 2      # 590 — vertical center of canvas

    lff_x     = 915           # left FF team boxes x start
    rff_x     = 1400          # right FF team boxes x start
    lff_right = lff_x + FF_W  # 1100
    rff_right = rff_x + FF_W  # 1585

    lff_top_y = CY - FF_H - FF_GAP / 2   # 551
    lff_bot_y = CY + FF_GAP / 2          # 597
    rff_top_y, rff_bot_y = lff_top_y, lff_bot_y

    lff_top_cy = lff_top_y + FF_H / 2    # 567
    lff_bot_cy = lff_bot_y + FF_H / 2    # 613
    rff_top_cy, rff_bot_cy = lff_top_cy, lff_bot_cy
    ff_mid_cy  = (lff_top_cy + lff_bot_cy) / 2   # 590

    # Championship box
    CHAMP_W, CHAMP_H = 210, 52
    champ_x      = (lff_right + rff_x) / 2 - CHAMP_W / 2   # 1145
    champ_y      = CY - CHAMP_H / 2                         # 564
    champ_right  = champ_x + CHAMP_W                        # 1355
    champ_mid_cy = champ_y + CHAMP_H / 2                    # 590

    # Bracket join x positions (midpoints between FF boxes and championship)
    join_lx = (lff_right + champ_x) / 2   # ~1122
    join_rx = (champ_right + rff_x) / 2   # ~1377

    # ── Matchup probabilities ──────────────────────────────────────────────────
    left_p   = win_prob_fn(lt_champ["tid"], lb_champ["tid"])
    left_winner  = lt_champ if left_p >= 0.5 else lb_champ
    right_p  = win_prob_fn(rt_champ["tid"], rb_champ["tid"])
    right_winner = rt_champ if right_p >= 0.5 else rb_champ
    title_p  = win_prob_fn(left_winner["tid"], right_winner["tid"])
    champion = left_winner if title_p >= 0.5 else right_winner

    def title_odds(tid: str) -> float:
        return adv_odds.get(tid, {}).get(6, 0.0)

    def ff_team_box(rx: float, y: float, team: dict, p_ff: float, is_winner: bool) -> str:
        fill   = color        if is_winner else NORD["bg1"]
        tclr   = NORD["bg"]  if is_winner else NORD["snow1"]
        stroke = color        if is_winner else line_clr
        name   = _html.escape(str(team["name"])[:20])
        seed   = team.get("s", "?")
        return "\n".join([
            f'<rect x="{rx:.1f}" y="{y:.1f}" width="{FF_W}" height="{FF_H}" rx="3" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.1"/>',
            f'<text x="{rx+8:.1f}" y="{y+14:.1f}" font-size="9.5" font-weight="700" '
            f'fill="{tclr}" font-family="{font}">'
            f'<tspan font-size="8" opacity="0.7">#{seed} </tspan>{name}</text>',
            f'<text x="{rx+8:.1f}" y="{y+27:.1f}" font-size="8.2" fill="{tclr}" '
            f'opacity="0.88" font-family="{font}">'
            f'FF: {p_ff:.0%}  |  Title: {title_odds(team["tid"]):.1%}</text>',
        ])

    # Draw the 4 Final Four team boxes
    svg_parts.append(ff_team_box(lff_x, lff_top_y, lt_champ, left_p,      lt_champ["tid"] == left_winner["tid"]))
    svg_parts.append(ff_team_box(lff_x, lff_bot_y, lb_champ, 1 - left_p,  lb_champ["tid"] == left_winner["tid"]))
    svg_parts.append(ff_team_box(rff_x, rff_top_y, rt_champ, right_p,     rt_champ["tid"] == right_winner["tid"]))
    svg_parts.append(ff_team_box(rff_x, rff_bot_y, rb_champ, 1 - right_p, rb_champ["tid"] == right_winner["tid"]))

    # "FINAL FOUR" labels above each pair
    for lbl_x in (lff_x + FF_W / 2, rff_x + FF_W / 2):
        svg_parts.append(
            f'<text x="{lbl_x:.1f}" y="{lff_top_y - 10:.1f}" text-anchor="middle" '
            f'font-size="9" font-weight="700" fill="{NORD["snow0"]}" letter-spacing=".07em" '
            f'font-family="{font}">FINAL FOUR</text>'
        )

    # ── Elbow connectors: E8 anchors → FF team boxes ───────────────────────────
    def elbow(ax: float, ay: float, bx: float, by: float) -> str:
        mx = (ax + bx) / 2
        return (
            f'<polyline points="{ax:.1f},{ay:.1f} {mx:.1f},{ay:.1f} '
            f'{mx:.1f},{by:.1f} {bx:.1f},{by:.1f}" '
            f'fill="none" stroke="{line_clr}" stroke-width="1.4"/>'
        )

    # Left E8 exits right (anchor x = 890), connects to left FF boxes left edge
    svg_parts.append(elbow(lt_anchor[0], lt_anchor[1], lff_x,     lff_top_cy))
    svg_parts.append(elbow(lb_anchor[0], lb_anchor[1], lff_x,     lff_bot_cy))
    # Right E8 exits left (anchor x = 1610), connects to right FF boxes right edge
    svg_parts.append(elbow(rt_anchor[0], rt_anchor[1], rff_right, rff_top_cy))
    svg_parts.append(elbow(rb_anchor[0], rb_anchor[1], rff_right, rff_bot_cy))

    # ── Bracket connectors: FF pairs → Championship ────────────────────────────
    for (jx, box_left, box_right, toward_champ) in (
        (join_lx, lff_right, lff_right, champ_x),      # left pair  → champ left
        (join_rx, rff_x,     rff_x,     champ_right),  # right pair → champ right
    ):
        top_cy = lff_top_cy
        bot_cy = lff_bot_cy
        edge_x = box_left if jx > box_left else box_right
        # Bracket brace: top → join → bottom
        svg_parts.append(
            f'<polyline points="{edge_x:.1f},{top_cy:.1f} {jx:.1f},{top_cy:.1f} '
            f'{jx:.1f},{bot_cy:.1f} {edge_x:.1f},{bot_cy:.1f}" '
            f'fill="none" stroke="{line_clr}" stroke-width="1.2"/>'
        )
        # Dot at midpoint
        svg_parts.append(
            f'<circle cx="{jx:.1f}" cy="{ff_mid_cy:.1f}" r="2.5" '
            f'fill="{NORD["snow0"]}" opacity="0.6"/>'
        )
        # Horizontal line from join to championship edge
        svg_parts.append(
            f'<line x1="{jx:.1f}" y1="{ff_mid_cy:.1f}" '
            f'x2="{toward_champ:.1f}" y2="{champ_mid_cy:.1f}" '
            f'stroke="{line_clr}" stroke-width="1.4"/>'
        )

    # ── Championship box ────────────────────────────────────────────────────────
    svg_parts.append(
        f'<rect x="{champ_x:.1f}" y="{champ_y:.1f}" width="{CHAMP_W}" height="{CHAMP_H}" rx="4" '
        f'fill="{NORD["bg2"]}" stroke="{color}" stroke-width="1.7"/>'
    )
    svg_parts.append(
        f'<text x="{champ_x + CHAMP_W/2:.1f}" y="{champ_y - 11:.1f}" text-anchor="middle" '
        f'font-size="10" font-weight="800" fill="{NORD["snow0"]}" letter-spacing=".06em" '
        f'font-family="{font}">NATIONAL CHAMPIONSHIP</text>'
    )
    svg_parts.append(
        f'<text x="{champ_x + CHAMP_W/2:.1f}" y="{champ_y + 17:.1f}" text-anchor="middle" '
        f'font-size="8.5" fill="{NORD["bg3"]}" font-family="{font}">'
        f'{_html.escape(left_winner["name"][:17])} vs {_html.escape(right_winner["name"][:17])}</text>'
    )
    svg_parts.append(
        f'<text x="{champ_x + CHAMP_W/2:.1f}" y="{champ_y + 33:.1f}" text-anchor="middle" '
        f'font-size="11" font-weight="800" fill="{text_clr}" font-family="{font}">'
        f'\U0001f3c6 {_html.escape(champion["name"][:22])}</text>'
    )
    svg_parts.append(
        f'<text x="{champ_x + CHAMP_W/2:.1f}" y="{champ_y + 47:.1f}" text-anchor="middle" '
        f'font-size="8.5" fill="{NORD["snow0"]}" opacity="0.72" font-family="{font}">'
        f'Title odds: {title_odds(champion["tid"]):.1%}</text>'
    )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'style="display:block;background:{NORD["bg"]}">'
        + "".join(svg_parts)
        + "\n</svg>"
    )

    return f"""
<style>
#full-wrap{{overflow:auto;background:{NORD["bg"]};border:1px solid {NORD["bg3"]};border-radius:10px;padding:10px;width:100%;min-height:780px}}
#full-stage{{width:{W}px;height:{H}px;transform:scale(.56);transform-origin:top left;display:block}}
.full-btn{{background:{NORD["bg1"]};border:1px solid {NORD["bg3"]};color:{NORD["snow1"]};padding:4px 14px;
  border-radius:5px;cursor:pointer;font-size:14px;font-family:{font}}}
.full-btn:hover{{background:{NORD["bg2"]}}}
</style>
<div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">
  <button class="full-btn" onclick="adjZ(-.08)">&#8722;</button>
  <button class="full-btn" onclick="adjZ(.08)">+</button>
  <button class="full-btn" onclick="setZ(.56)">Reset</button>
  <span id="full-lbl" style="color:{NORD["snow0"]};font-size:12px;font-family:{font}">56%</span>
  <span style="color:{NORD["bg3"]};font-size:11px;font-family:{font};margin-left:8px">Scroll to pan</span>
</div>
<div id="full-wrap"><div id="full-stage">{svg}</div></div>
<script>
var _z=.56;
function setZ(s){{
  _z=Math.max(.35,Math.min(1.3,s));
  var st=document.getElementById('full-stage');
  st.style.transform='scale('+_z+')';
  document.getElementById('full-wrap').style.height=Math.ceil({H}*_z+24)+'px';
  document.getElementById('full-lbl').textContent=Math.round(_z*100)+'%';
}}
function adjZ(d){{setZ(_z+d);}}
setZ(.56);
</script>
"""


# ── Region advancement table ───────────────────────────────────────────────────

def region_table(seed_map: dict, adv_odds: dict, champ_odds: dict) -> pd.DataFrame:
    rows = []
    for seed in BRACKET_SLOT_ORDER:
        if seed not in seed_map:
            continue
        tid, name, rating = seed_map[seed]
        row = {
            "Seed": seed,
            "Team": name,
            "Elo": round(rating, 1),
        }
        for rnd, lbl in ROUND_LABELS.items():
            p = adv_odds.get(tid, {}).get(rnd, 0.0)
            row[lbl] = p
        row["Title"] = champ_odds.get(tid, 0.0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Seed")
    return df


def style_region_table(df: pd.DataFrame, color: str):
    pct_cols = [c for c in df.columns if c not in ("Team", "Elo")]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.0%}" if v >= 0.005 else "-"
        return v

    styled = (
        df.style
        .format({c: fmt for c in pct_cols})
        .format({"Elo": "{:.1f}"})
        .background_gradient(subset=pct_cols, cmap="Blues", vmin=0, vmax=1)
        .set_properties(**{"font-size": "12px"})
        .set_table_styles([
            {"selector": "th", "props": [("font-size", "12px"),
                                          ("text-align", "center"),
                                          ("white-space", "nowrap")]},
            {"selector": "td", "props": [("text-align", "center")]},
            {"selector": "td:nth-child(2)", "props": [("text-align", "left")]},
        ])
    )
    return styled


# ── Page layout ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChalkIQ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS — Nord dark theme
st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        padding: 0 18px;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
    }
    div[data-testid="metric-container"] {
        background: #3B4252;
        border: 1px solid #4C566A;
        border-radius: 8px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header + division toggle ───────────────────────────────────────────────────

hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown("## 🎯 ChalkIQ")
    st.markdown("*the favorites win*")
    st.caption(f"Elo power ratings · Monte Carlo simulation · 2025–26 season through {SEASON_END}")
with hcol2:
    division = st.radio(
        "Division",
        options=["mens", "womens"],
        format_func=lambda d: f"{DIVISION_CONFIG[d]['emoji']} {DIVISION_CONFIG[d]['label']}",
        horizontal=True,
        key="division",
    )

cfg   = DIVISION_CONFIG[division]
color = cfg["color"]
light = cfg["light"]

st.markdown("---")

# ── Load data ─────────────────────────────────────────────────────────────────

engine                     = load_engine(division)
rankings                   = engine.rankings()
regions, adv_odds, champ_odds = load_bracket_data(division)
metrics                    = evaluate(engine.history)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_rank, tab_bracket, tab_live, tab_eval, tab_matchup, tab_math = st.tabs([
    "📊  Power Rankings",
    "🏆  Bracket",
    "🔴  Live",
    "📈  Model Evaluation",
    "⚔️  Matchup",
    "📐  Math",
])


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — Power Rankings
# ════════════════════════════════════════════════════════════════════════════ #

with tab_rank:
    st.subheader(f"Top 64 | {cfg['label']} Division")
    st.caption("The 64 teams most likely to receive an at-large or automatic bid on Selection Sunday, ranked by Elo.")

    all_rows = []
    for rank, (tid, name, rating) in enumerate(rankings[:64], 1):
        all_rows.append({
            "Rank":  rank,
            "Team":  name,
            "Elo":   round(rating, 1),
            "R32":   adv_odds.get(tid, {}).get(2, 0),
            "S16":   adv_odds.get(tid, {}).get(3, 0),
            "E8":    adv_odds.get(tid, {}).get(4, 0),
            "FF":    adv_odds.get(tid, {}).get(5, 0),
            "Title": champ_odds.get(tid, 0),
        })

    df_all  = pd.DataFrame(all_rows).set_index("Rank")
    pct_cols = ["R32", "S16", "E8", "FF", "Title"]

    def _fmt_pct(v):
        if isinstance(v, float) and v > 0:
            return f"{v:.1%}"
        return "-"

    styled_all = (
        df_all.style
        .format({c: _fmt_pct for c in pct_cols})
        .format({"Elo": "{:.1f}"})
        .background_gradient(subset=pct_cols, cmap="Blues", vmin=0, vmax=0.5)
        .bar(subset=["Elo"], color=[light, color])
        .set_properties(**{"font-size": "13px"})
    )

    col_tbl, col_chart = st.columns([3, 2])
    with col_tbl:
        st.dataframe(styled_all, height=1800, width="stretch")

    with col_chart:
        top20_names = [name for _, name, _ in rankings[:20]]
        top20_odds  = [champ_odds.get(tid, 0) for tid, _, _ in rankings[:20]]
        fig_bar = plotly_odds_bar(top20_names[::-1], top20_odds[::-1], color)
        st.plotly_chart(fig_bar, width="stretch")
        st.caption("Top 20 championship odds from 10,000 Monte Carlo bracket simulations.")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Bracket
# ════════════════════════════════════════════════════════════════════════════ #

with tab_bracket:
    st.subheader(f"Tournament Bracket | {cfg['label']} Division")
    st.caption(
        "Bracket seeded by current Elo (S-curve). "
        "South TL | Midwest BL | East TR | West BR. "
        "Win% and title odds shown on each Final Four team. "
        "Actual Selection Sunday seeding may differ."
    )

    # Final Four visual at the top
    fig_ff = draw_final_four(regions, adv_odds, engine.names, color)
    st.pyplot(fig_ff, width="stretch")
    plt.close()

    st.markdown("---")

    # Combined bracket — all 4 regions + Final Four + Championship on one canvas
    bracket_html = combined_bracket_html(regions, engine.win_prob, color, cfg["label"], adv_odds)
    components.html(bracket_html, height=980, scrolling=True)
    st.caption(
        "Highlighted box = projected winner.  "
        "Win% shown bottom-right of each team.  "
        "Elo shown bottom-left.  Neutral court assumption."
    )

    # Advancement probability tables (per region, collapsible)
    with st.expander("Show advancement probability tables"):
        reg_sub_tabs = st.tabs(REGIONS)
        for rsub, region_name in zip(reg_sub_tabs, REGIONS):
            with rsub:
                df_reg = region_table(regions[region_name], adv_odds, champ_odds)
                styled = style_region_table(df_reg, color)
                st.dataframe(styled, width="stretch", height=420)
        st.caption(
            "**R64** = survives first round  ·  **R32** = Round of 32  ·  "
            "**S16** = Sweet 16  ·  **E8** = Elite 8  ·  "
            "**FF** = Final Four  ·  **Title** = Championship"
        )


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Live Games
# ════════════════════════════════════════════════════════════════════════════ #

with tab_live:
    st.subheader(f"Live Games | {cfg['label']} Division")

    # ── Refresh controls ──────────────────────────────────────────────────────
    ctrl_col, ts_col = st.columns([1, 3])
    with ctrl_col:
        if st.button("🔄 Refresh", key="live_refresh"):
            st.cache_data.clear()
            st.rerun()
    with ts_col:
        import datetime as _dt
        st.caption(f"Last checked: {_dt.datetime.now().strftime('%I:%M:%S %p')}")

    live_games = load_live_games(division)

    if not live_games:
        st.info(
            "No games are live right now.\n\n"
            "During the tournament (March), this tab shows real-time "
            "win probabilities, upset alerts, and bracket impact for every "
            "game in progress."
        )
    else:
        # Annotate each game with live win probability and pregame baseline
        ranked_ids = {tid for tid, _, _ in rankings}
        for g in live_games:
            p_pre = engine.win_prob(g["home_id"], g["away_id"], neutral=g["neutral"])
            score_diff = g["home_score"] - g["away_score"]
            p_live = live_win_prob(score_diff, g["minutes_remaining"], p_pre)
            g["pregame_prob_home"] = p_pre
            g["live_prob_home"]    = p_live
            g["swing"]             = prob_swing(p_live, p_pre)
            g["upset"]             = upset_alert(p_live, p_pre)
            g["home_ranked"]       = g["home_id"] in ranked_ids
            g["away_ranked"]       = g["away_id"] in ranked_ids

        # Sort: upsets first, then by swing magnitude
        live_games.sort(key=lambda g: (not g["upset"], -abs(g["swing"])))

        # ── Game cards ────────────────────────────────────────────────────────
        cols = st.columns(2)
        for i, g in enumerate(live_games):
            col = cols[i % 2]
            with col:
                p_h  = g["live_prob_home"]
                p_a  = 1.0 - p_h
                pp_h = g["pregame_prob_home"]
                sw   = g["swing"]

                # Card border color: upset = orange, normal = bg3
                border = NORD["orange"] if g["upset"] else NORD["bg3"]
                badge  = (
                    f'<span style="background:{NORD["orange"]};color:{NORD["bg"]};'
                    f'font-size:9px;font-weight:700;padding:2px 6px;border-radius:3px;'
                    f'letter-spacing:.05em">UPSET ALERT</span>'
                    if g["upset"] else ""
                )

                swing_color = NORD["green"] if sw > 0 else NORD["red"]
                swing_str   = f"{sw:+.0%}"

                # Clock display
                if g["status"] == "halftime":
                    clock_str = "HALFTIME"
                else:
                    h = int(g["clock_mins"])
                    s = int((g["clock_mins"] - h) * 60)
                    half_lbl = (
                        "1st Half" if g["period"] == 1 else
                        "2nd Half" if g["period"] == 2 else
                        f"OT{g['period']-2}"
                    )
                    clock_str = f"{h}:{s:02d} — {half_lbl}"

                card_html = f"""
<div style="border:1px solid {border};border-radius:8px;padding:14px 16px;
            margin-bottom:12px;background:{NORD["bg1"]}">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <span style="font-size:11px;color:{NORD["bg3"]};font-family:ui-sans-serif,sans-serif">
      {_html.escape(g["status_detail"] or clock_str)}
    </span>
    {badge}
  </div>

  <table style="width:100%;border-collapse:collapse;font-family:ui-sans-serif,sans-serif">
    <tr>
      <td style="font-size:13px;font-weight:700;color:{NORD["snow2"]};padding:2px 0">
        {_html.escape(g["home_name"][:26])}
      </td>
      <td style="font-size:20px;font-weight:800;color:{NORD["snow2"]};text-align:right;padding:2px 0">
        {g["home_score"]}
      </td>
    </tr>
    <tr>
      <td style="font-size:13px;font-weight:700;color:{NORD["snow2"]};padding:2px 0">
        {_html.escape(g["away_name"][:26])}
      </td>
      <td style="font-size:20px;font-weight:800;color:{NORD["snow2"]};text-align:right;padding:2px 0">
        {g["away_score"]}
      </td>
    </tr>
  </table>

  <div style="margin:10px 0 4px 0">
    <div style="display:flex;justify-content:space-between;
                font-size:10px;color:{NORD["bg3"]};margin-bottom:3px;
                font-family:ui-sans-serif,sans-serif">
      <span>{_html.escape(g["home_name"][:20])}</span>
      <span>{_html.escape(g["away_name"][:20])}</span>
    </div>
    <div style="background:{NORD["bg3"]};border-radius:4px;height:8px;overflow:hidden">
      <div style="background:{color};height:100%;width:{p_h*100:.1f}%;
                  border-radius:4px;transition:width .3s"></div>
    </div>
    <div style="display:flex;justify-content:space-between;
                font-size:11px;font-weight:700;margin-top:3px;
                font-family:ui-sans-serif,sans-serif">
      <span style="color:{color}">{p_h:.0%}</span>
      <span style="color:{NORD["bg3"]};font-size:10px">
        Pregame: {pp_h:.0%} / {1-pp_h:.0%}
      </span>
      <span style="color:{color}">{p_a:.0%}</span>
    </div>
  </div>

  <div style="font-size:10px;color:{swing_color};font-family:ui-sans-serif,sans-serif">
    Home swing since tip-off: <strong>{swing_str}</strong>
  </div>
</div>
"""
                st.markdown(card_html, unsafe_allow_html=True)

        # ── Bracket odds impact (Milestone 4) ─────────────────────────────────
        st.markdown("---")
        st.markdown("### Bracket Odds Impact")
        st.caption(
            "How would today's results shift each team's projected championship odds? "
            "Simulated by applying the hypothetical game result to current Elo ratings "
            "and rerunning 5,000 bracket simulations."
        )

        # Only show impact for games where at least one team is in our top-64
        impact_games = [g for g in live_games if g["home_ranked"] or g["away_ranked"]]

        if not impact_games:
            st.info("No live games involve top-64 ranked teams right now.")
        else:
            for g in impact_games:
                hname = g["home_name"]
                aname = g["away_name"]

                with st.expander(f"{hname} vs {aname} — impact if result holds", expanded=True):
                    # Base title odds for these two teams
                    base_h = champ_odds.get(g["home_id"], 0.0)
                    base_a = champ_odds.get(g["away_id"], 0.0)

                    # Simulate both outcomes
                    with st.spinner("Simulating bracket impact…"):
                        odds_h_wins = live_bracket_impact(division, g["home_id"], g["away_id"], True)
                        odds_a_wins = live_bracket_impact(division, g["home_id"], g["away_id"], False)

                    new_h_if_h = odds_h_wins.get(g["home_id"], 0.0)
                    new_h_if_a = odds_a_wins.get(g["home_id"], 0.0)
                    new_a_if_h = odds_h_wins.get(g["away_id"], 0.0)
                    new_a_if_a = odds_a_wins.get(g["away_id"], 0.0)

                    ic1, ic2 = st.columns(2)
                    with ic1:
                        st.markdown(f"**If {hname[:22]} wins**")
                        dh = new_h_if_h - base_h
                        da = new_a_if_h - base_a
                        st.metric(
                            f"{hname[:20]} title odds",
                            f"{new_h_if_h:.2%}",
                            delta=f"{dh:+.2%}",
                            delta_color="normal",
                        )
                        st.metric(
                            f"{aname[:20]} title odds",
                            f"{new_a_if_h:.2%}",
                            delta=f"{da:+.2%}",
                            delta_color="normal",
                        )
                    with ic2:
                        st.markdown(f"**If {aname[:22]} wins (upset)**")
                        dh2 = new_h_if_a - base_h
                        da2 = new_a_if_a - base_a
                        st.metric(
                            f"{hname[:20]} title odds",
                            f"{new_h_if_a:.2%}",
                            delta=f"{dh2:+.2%}",
                            delta_color="normal",
                        )
                        st.metric(
                            f"{aname[:20]} title odds",
                            f"{new_a_if_a:.2%}",
                            delta=f"{da2:+.2%}",
                            delta_color="normal",
                        )
                    st.caption(
                        f"Baseline title odds: {hname[:18]} {base_h:.2%} | "
                        f"{aname[:18]} {base_a:.2%}"
                    )


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 4 — Model Evaluation
# ════════════════════════════════════════════════════════════════════════════ #

with tab_eval:
    st.subheader(f"Model Evaluation | {cfg['label']} Division")

    ll   = metrics.get("log_loss",            0)
    bs   = metrics.get("brier_score",         0)
    ll_b = metrics.get("baseline_log_loss",   0)
    bs_b = metrics.get("baseline_brier_score",0)
    n    = metrics.get("n_games",             0)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Games evaluated", f"{n:,}")
    m2.metric(
        "Log Loss: Elo", f"{ll:.4f}",
        help="−[y·log(p) + (1−y)·log(1−p)] per game. Lower = better. "
             "Coin-flip baseline ≈ 0.693.",
    )
    m3.metric(
        "Log Loss: Baseline", f"{ll_b:.4f}",
        delta=f"{ll - ll_b:+.4f}", delta_color="inverse",
        help="Negative delta = Elo beats the 50/50 baseline.",
    )
    m4.metric(
        "Brier Score: Elo", f"{bs:.4f}",
        help="(p − y)² per game. Lower = better. Baseline = 0.25.",
    )
    m5.metric(
        "Brier Score: Baseline", f"{bs_b:.4f}",
        delta=f"{bs - bs_b:+.4f}", delta_color="inverse",
    )

    st.markdown("---")

    cal_col, txt_col = st.columns([2, 1])
    with cal_col:
        st.markdown("**Calibration: predicted vs observed win rate**")
        fig_cal = plotly_calibration(metrics.get("calibration", []), color)
        st.plotly_chart(fig_cal, width="stretch")

    with txt_col:
        st.markdown("**What is calibration?**")
        st.markdown(
            "A model is *well-calibrated* if events predicted at 70% "
            "actually happen about 70% of the time.\n\n"
            "Points on the dashed line = perfect calibration.\n\n"
            "Points **above** the line → the model is underconfident.\n\n"
            "Points **below** the line → overconfident.\n\n"
            "Early in the season, Elo starts all teams at 1500 "
            "so predictions are near 50/50 and calibration improves "
            "as ratings converge."
        )
        st.markdown("**Why two metrics?**")
        st.markdown(
            "**Log loss** punishes overconfident wrong predictions "
            "very hard (logarithmic penalty). "
            "**Brier score** is the squared error, softer and easier to interpret. "
            "Using both gives a rounder picture of forecast quality."
        )


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 5 — Matchup Calculator
# ════════════════════════════════════════════════════════════════════════════ #

with tab_matchup:
    st.subheader(f"Head-to-Head Matchup | {cfg['label']} Division")

    all_teams = [(tid, name) for tid, name, _ in rankings]
    team_options = {name: tid for tid, name in all_teams}
    names_list   = [name for _, name in all_teams]

    col_a, col_b = st.columns(2)
    with col_a:
        name_a = st.selectbox(
            "Team A",
            names_list,
            index=0,
            help="First team (neutral court assumed).",
        )
    with col_b:
        remaining = [n for n in names_list if n != name_a]
        name_b = st.selectbox(
            "Team B",
            remaining,
            index=min(4, len(remaining) - 1),
            help="Second team. Win probability = 1 - P(Team A).",
        )

    tid_a = team_options[name_a]
    tid_b = team_options[name_b]
    m = matchup_prob(engine, tid_a, tid_b)

    st.markdown("---")

    res_col, chart_col = st.columns([1, 2])

    with res_col:
        st.markdown(f"### {name_a} vs {name_b}")
        st.markdown(
            f"| | {name_a[:22]} | {name_b[:22]} |\n"
            f"|---|---|---|\n"
            f"| **Elo** | {m['rating_a']} | {m['rating_b']} |\n"
            f"| **Win prob** | **{m['prob_a']:.1%}** | **{m['prob_b']:.1%}** |\n"
            f"| **Rating diff** | {m['rating_diff']:+.1f} | - |"
        )
        st.markdown(
            f"**Favorite:** {m['favorite']}  \n"
            f"Formula: $p = \\dfrac{{1}}{{1 + 10^{{(R_B - R_A)/400}}}}$"
        )

    with chart_col:
        fig_mu = go.Figure(go.Bar(
            x=[m["prob_a"], m["prob_b"]],
            y=[name_a[:30], name_b[:30]],
            orientation="h",
            marker_color=[color, NORD["bg3"]],
            text=[f"{m['prob_a']:.1%}", f"{m['prob_b']:.1%}"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(color="white", size=15),
        ))
        fig_mu.add_vline(x=0.5, line_dash="dash", line_color=NORD["bg3"], line_width=1)
        fig_mu.update_layout(
            xaxis=dict(tickformat=".0%", range=[0, 1], title="Win probability"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=160, r=20, t=10, b=40),
            height=160,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_mu, width="stretch")
        st.caption("Neutral court assumption. Elo home-court adjustment not applied here.")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 6 — Math
# ════════════════════════════════════════════════════════════════════════════ #

with tab_math:
    st.subheader("The Math Behind This Dashboard")
    st.markdown(
        "Everything shown (rankings, win probabilities, live game estimates, bracket odds) "
        "comes from a small set of clean mathematical ideas. "
        "This page walks through each one with the formula and a plain-English explanation "
        "of what it means and why it works."
    )
    st.markdown("---")

    # ── 1. Game outcome ──────────────────────────────────────────────────────
    with st.expander("1 · Game outcome as a random variable", expanded=True):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Formula**")
            st.latex(r"""
Y_{ij} = \begin{cases} 1 & \text{team } i \text{ wins} \\ 0 & \text{team } i \text{ loses} \end{cases}
""")
            st.latex(r"Y_{ij} \sim \text{Bernoulli}(p_{ij})")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "A basketball game has exactly two outcomes for team $i$: win or lose. "
                "We model that as a **Bernoulli random variable**, a coin flip with a "
                "weighted coin. The weight $p_{ij}$ is the probability that team $i$ beats "
                "team $j$. The model's entire job is to figure out what $p_{ij}$ should be "
                "for every possible matchup."
            )

    # ── 2. Elo win probability ────────────────────────────────────────────────
    with st.expander("2 · Elo win probability", expanded=True):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Formula**")
            st.latex(r"p_{ij} = \frac{1}{1 + 10^{\,(R_j - R_i)\,/\,400}}")
            st.markdown("where $R_i$ and $R_j$ are the current Elo ratings of each team.")
            st.markdown("**Worked example:** Duke (1692) vs Houston (1580):")
            st.latex(r"p = \frac{1}{1 + 10^{(1580 - 1692)/400}} = \frac{1}{1 + 10^{-0.28}} \approx 0.63")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "This is a **logistic curve** scaled to base-10 notation. "
                "A few things to notice:\n\n"
                "- When $R_i = R_j$ (equal teams), $p = 0.5$ exactly (a coin flip).\n"
                "- A **400-point gap** gives the stronger team about a **91%** win probability.\n"
                "- A **100-point gap** ≈ 64% for the stronger team.\n"
                "- The formula is symmetric: $p_{ji} = 1 - p_{ij}$ always.\n\n"
                "The '400' and 'base 10' are historical convention from chess Elo. "
                "They set the *sensitivity* of the scale: how much a rating difference matters."
            )

    # ── 3. Why base 10 / 400? ────────────────────────────────────────────────
    with st.expander("3 · Why base 10 and why divide by 400?"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**The logistic connection**")
            st.latex(r"p_{ij} = \frac{1}{1 + e^{-(\ln 10 / 400)\,\Delta R}}")
            st.markdown("where $\\Delta R = R_i - R_j$.")
            st.markdown("**Reference points**")
            st.markdown(
                "| Rating diff | Win probability |\n"
                "|---|---|\n"
                "| 0 | 50.0% |\n"
                "| 100 | 64.0% |\n"
                "| 200 | 76.0% |\n"
                "| 400 | 90.9% |\n"
                "| 800 | 99.0% |"
            )
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Using $10^x = e^{x \\ln 10}$, the Elo formula is just a logistic sigmoid with slope "
                "$\\ln(10)/400 \\approx 0.00576$. So Elo is actually a **logistic regression model** "
                "with one input: the rating difference.\n\n"
                "- **Base 10** is a historical choice from chess. You could use base $e$ with "
                "a different divisor and get the same behavior.\n"
                "- **Dividing by 400** sets the scale. Smaller values make ratings more sensitive "
                "(a small difference matters more). Larger values flatten the curve "
                "(you need a huge gap to get a very high win probability).\n\n"
                "This project uses the standard chess constants because they are well-studied "
                "and give sensible probabilities for college basketball rating gaps."
            )

    # ── 4. Elo rating update ─────────────────────────────────────────────────
    with st.expander("4 · Elo rating update rule", expanded=True):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**After each game:**")
            st.latex(r"R_i' = R_i + K\,(S_i - E_i)")
            st.latex(r"R_j' = R_j + K\,(S_j - E_j)")
            st.markdown(
                "where:\n"
                "- $S_i \\in \\{0, 1\\}$ = actual result\n"
                "- $E_i = p_{ij}$ = predicted win probability\n"
                "- $K = 24$ = update step size (this project)\n"
            )
            st.markdown("**Worked example:** Duke expected to win (80%), but loses:")
            st.latex(r"R_{\text{Duke}}' = R_{\text{Duke}} + 24\,(0 - 0.80) = R_{\text{Duke}} - 19.2")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "The update rule is beautifully simple. It says:\n\n"
                "> *Move your rating up or down in proportion to how surprised you should be.*\n\n"
                "- $(S_i - E_i)$ is the **prediction error**, the gap between what happened and "
                "what the model expected.\n"
                "- If the favorite wins as expected, the surprise is small, so the rating barely moves.\n"
                "- If the underdog wins, the surprise is large, so the rating shifts significantly.\n"
                "- $K$ controls the **learning rate**. Higher $K$ = faster updates but more "
                "volatile ratings. $K=24$ is a common basketball choice (original chess Elo used $K=10$).\n\n"
                "This is essentially **stochastic gradient descent** on the prediction error, "
                "the same core idea used in neural network training."
            )

    # ── 5. Home court advantage ───────────────────────────────────────────────
    with st.expander("5 · Home court advantage"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Adjusted win probability for non-neutral games:**")
            st.latex(r"p_{\text{home}} = \frac{1}{1 + 10^{\,(R_{\text{away}} - R_{\text{home}} - H)\,/\,400}}")
            st.markdown("where $H = 100$ Elo points in this model.")
            st.markdown("**Effect of H = 100:**")
            st.latex(r"p = \frac{1}{1 + 10^{-100/400}} = \frac{1}{1 + 10^{-0.25}} \approx 0.64")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Playing at home is worth 100 Elo points, which translates to roughly "
                "a **64% win probability** in an otherwise even matchup.\n\n"
                "This is added as a temporary boost to the home team's effective rating "
                "during the win probability calculation; it does **not** change the stored Elo. "
                "The stored rating is home-adjusted away after each game update, so ratings "
                "reflect true team strength rather than schedule luck.\n\n"
                "Tournament games are played on **neutral courts**, so $H = 0$ applies. "
                "All bracket simulation probabilities assume neutral site."
            )

    # ── 6. Log loss ──────────────────────────────────────────────────────────
    with st.expander("6 · Log loss (proper scoring rule)"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Per game:**")
            st.latex(r"\ell_{\log}(p, y) = -\bigl[y \log p + (1-y)\log(1-p)\bigr]")
            st.markdown("**Average over $n$ games:**")
            st.latex(r"\text{Log Loss} = \frac{1}{n}\sum_{k=1}^{n} \ell_{\log}(p_k, y_k)")
            st.markdown("**50/50 baseline** (no model, always predict 50%):")
            st.latex(r"\text{Log Loss}_{\text{baseline}} = \log 2 \approx 0.693")
            st.markdown(f"**This model:** {ll:.4f}  ·  **Baseline:** {ll_b:.4f}  ·  **Δ** {ll - ll_b:+.4f}")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Imagine every time you make a prediction, you have to bet money on it. "
                "If you say *'I'm 99% sure Duke wins'* and Duke loses, you lose a TON of money. "
                "If you say *'I'm 60% sure'* and get it wrong, you only lose a little.\n\n"
                "Log loss is just the average amount of money you lose per game. "
                "**Lower is better.** A model that says 50/50 every time scores 0.693. "
                f"Our model scores **{ll:.3f}**, which means it's meaningfully better than "
                "just shrugging and saying 'I dunno, coin flip' for every game.\n\n"
                "The key rule: **never be extremely confident unless you're sure.** "
                "Being wrong with 99% confidence is catastrophically punished. "
                "This forces the model to be honest about uncertainty."
            )

    # ── 7. Brier score ───────────────────────────────────────────────────────
    with st.expander("7 · Brier score (proper scoring rule)"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Per game:**")
            st.latex(r"\ell_{\text{Brier}}(p, y) = (p - y)^2")
            st.markdown("**Average over $n$ games:**")
            st.latex(r"\text{Brier} = \frac{1}{n}\sum_{k=1}^{n}(p_k - y_k)^2")
            st.markdown("**50/50 baseline:**")
            st.latex(r"\text{Brier}_{\text{baseline}} = (0.5 - 1)^2 = (0.5 - 0)^2 = 0.25")
            st.markdown(f"**This model:** {bs:.4f}  ·  **Baseline:** {bs_b:.4f}  ·  **Δ** {bs - bs_b:+.4f}")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Think of this like measuring how far off your guess was on a number line "
                "from 0 to 1, then squaring it.\n\n"
                "- You say **90%**, team wins: you were 10% off, $(0.1)^2 = 0.01$ (great).\n"
                "- You say **90%**, team loses: you were 90% off, $(0.9)^2 = 0.81$ (bad).\n"
                "- You say **50%**, team wins: you were 50% off, $(0.5)^2 = 0.25$ (that's the baseline).\n\n"
                "**Why use both log loss and Brier?** They're grading you differently. "
                "Log loss is the strict teacher who goes nuclear if you're confidently wrong. "
                "Brier score is the lenient teacher who just measures how far off you were. "
                "A good model passes both."
            )

    # ── 8. Calibration ───────────────────────────────────────────────────────
    with st.expander("8 · Calibration"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Perfectly calibrated model:**")
            st.latex(r"P(Y=1 \mid \hat{p} = p) = p \quad \forall\, p \in [0,1]")
            st.markdown("**How we check it:**")
            st.markdown(
                "Group all game predictions into bins by predicted probability "
                "(e.g. 50–60%, 60–70%, …). For each bin, compare:\n\n"
                "$$\\bar{p}_{\\text{bin}} \\approx \\bar{y}_{\\text{bin}}$$\n\n"
                "average predicted prob ≈ observed win rate."
            )
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Calibration checks one simple thing: **does 70% actually mean 70%?**\n\n"
                "Imagine you looked at every game where our model said the favorite had "
                "exactly a 70% chance to win. If the model is well-calibrated, that team "
                "should actually win about 70 out of every 100 such games.\n\n"
                "If they only win 55 out of 100, the model is **overconfident**: it keeps "
                "saying 70% but reality is closer to 55%.\n\n"
                "If they win 85 out of 100, the model is **underconfident**: it's better "
                "than it thinks it is.\n\n"
                "The chart in the Model Evaluation tab shows this visually. "
                "Every dot sitting on the dashed diagonal line = perfect calibration."
            )

    # ── 9. Monte Carlo simulation ─────────────────────────────────────────────
    with st.expander("9 · Monte Carlo bracket simulation", expanded=True):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Championship probability estimate:**")
            st.latex(
                r"\widehat{P}(\text{team } i \text{ wins title}) = "
                r"\frac{1}{N}\sum_{s=1}^{N} \mathbf{1}\{\text{team }i\text{ wins sim }s\}"
            )
            st.markdown(
                "where $N = 100{,}000$ simulations and $\\mathbf{1}\\{\\cdot\\}$ is "
                "the indicator function (1 if true, 0 if false)."
            )
            st.markdown("**Each simulation:**")
            st.markdown(
                "1. Draw 64 teams into the bracket.\n"
                "2. For each game, flip a weighted coin using $p_{ij}$.\n"
                "3. Advance winner; repeat until one team remains.\n"
                "4. Record the champion."
            )
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "The full bracket has over **9 quintillion** possible outcomes. "
                "There is no way to calculate every one exactly; it would take longer "
                "than the age of the universe.\n\n"
                "So instead we just **play the whole tournament 100,000 times** on the "
                "computer, each time flipping weighted coins for every game. Then we count: "
                "*how many times did Duke win the whole thing?*\n\n"
                "If Duke won 9,200 out of 100,000 simulated tournaments, "
                "we report their title odds as **9.2%**.\n\n"
                "It's the same idea as: if you want to know how often you roll a 6 with "
                "a fair die, just roll it 100,000 times and count. You'll get very close "
                "to the right answer (16.7%) without ever doing the math."
            )

    # ── 10. Round advancement ────────────────────────────────────────────────
    with st.expander("10 · Round advancement probabilities"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Probability of reaching round $r$:**")
            st.latex(
                r"\widehat{P}(\text{team }i\text{ reaches round }r) = "
                r"\frac{1}{N}\sum_{s=1}^{N} \mathbf{1}\{w_i^{(s)} \geq r\}"
            )
            st.markdown(
                "where $w_i^{(s)}$ = number of wins for team $i$ in simulation $s$."
            )
            st.markdown("**Round labels used here:**")
            st.markdown(
                "| Wins | Round | Label |\n"
                "|---|---|---|\n"
                "| ≥ 1 | Round of 32 | R32 |\n"
                "| ≥ 2 | Sweet 16 | S16 |\n"
                "| ≥ 3 | Elite 8 | E8 |\n"
                "| ≥ 4 | Final Four | FF |\n"
                "| ≥ 5 | Championship game | (runner-up) |\n"
                "| ≥ 6 | Champion | Title |"
            )
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Same idea as section 9, but instead of just tracking the champion, "
                "we keep a scorecard for **every team in every simulation**.\n\n"
                "After 100,000 simulated tournaments, Duke might have:\n"
                "- Made the Sweet 16 in 34,000 of them → **34% Sweet 16 odds**\n"
                "- Made the Elite 8 in 21,000 → **21% Elite 8 odds**\n"
                "- Made the Final Four in 13,000 → **13% Final Four odds**\n"
                "- Won the title in 9,000 → **9% title odds**\n\n"
                "Notice the numbers get smaller each round, which makes sense, "
                "because to make the Final Four you first have to make the Elite 8. "
                "Each round is a harder hurdle. The formula $w_i^{(s)} \\geq r$ just "
                "means *'did team i win at least r games in simulation s?'*"
            )

    # ── 11. S-curve seeding ──────────────────────────────────────────────────
    with st.expander("11 · NCAA S-curve bracket seeding"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**S-curve assignment:**")
            st.markdown(
                "Teams ranked 1–64 by Elo are assigned to regions using the **snake pattern**:\n\n"
                "| Overall seeds | Regional seed | Region order |\n"
                "|---|---|---|\n"
                "| 1–4 | 1 | East, West, South, Midwest |\n"
                "| 5–8 | 2 | Midwest, South, West, East |\n"
                "| 9–12 | 3 | East, West, South, Midwest |\n"
                "| … | … | alternating snake |\n"
            )
            st.markdown("**First-round matchups in each region:**")
            st.markdown(
                "1 vs 16, 8 vs 9, 5 vs 12, 4 vs 13, 6 vs 11, 3 vs 14, 7 vs 10, 2 vs 15"
            )
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "The NCAA uses the S-curve to **balance strength across the four regions**. "
                "No region should have three #1-caliber teams while another has none.\n\n"
                "The snake pattern ensures each region gets exactly one team from every group of "
                "four consecutive overall seeds. So the East gets the #1 overall team, the Midwest "
                "gets the #4 overall team, but in the next group the Midwest gets #5 and the East "
                "gets #8, balancing out.\n\n"
                "In this dashboard, we assign seeds by **current Elo rating**, not the actual "
                "Selection Committee's rankings. The real bracket will differ based on subjective "
                "factors, conference tournaments, and the committee's own metrics."
            )

    # ── 12. Live win probability model ────────────────────────────────────────
    with st.expander("12 · Live win probability: the random-walk model", expanded=True):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**The model:**")
            st.latex(
                r"P(\text{win} \mid d,\, t,\, p_0)"
                r"= \sigma\!\left(\frac{d}{\sigma_s \sqrt{t}} + z_{\text{prior}}\right)"
            )
            st.markdown(
                "where:\n"
                r"- $d$ = current score differential (positive = team leading)" "\n"
                r"- $t$ = minutes remaining in the game" "\n"
                r"- $\sigma_s = 2.0$ pts$/$√min (scoring diffusion constant)" "\n"
                r"- $z_{\text{prior}}$ = Elo prior in $z$-score space (see section 13)" "\n"
                r"- $\sigma(z) = \tfrac{1}{1+e^{-z}}$ is the logistic function"
            )
            st.markdown("**Worked examples:**")
            st.markdown(
                "| Scenario | Live prob |\n"
                "|---|---|\n"
                "| Tied at tip-off, 60% Elo fav | 60.0% |\n"
                "| Up 7, 8.5 min left, 60% fav | 93.0% |\n"
                "| Up 10 at halftime, equal teams | 88.4% |\n"
                "| Down 5, 3 min left, 65% fav | 11.9% |\n"
                "| Up 1, 1 min left, equal teams | 71.2% |"
            )
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Think of the score difference as a **drunk walk on a number line**. "
                "At every possession, the score ticks up for one team or the other "
                "somewhat randomly. Over time, this walk wanders up and down.\n\n"
                "The key insight is **time remaining**. With 35 minutes left, a "
                "5-point lead is nearly meaningless — the walk has dozens of steps "
                "left to erase it. With 1 minute left, a 5-point lead is almost "
                "insurmountable — there simply aren't enough possessions to overcome it.\n\n"
                "Mathematically, the uncertainty in the final score grows like "
                r"$\sigma_s \sqrt{t}$: the longer the game, the more the score can wander. "
                "Dividing the current lead by $\\sigma_s \\sqrt{t}$ tells us how many "
                "'standard deviations' ahead this team is, accounting for how much time "
                "is left to flip the result.\n\n"
                "At tip-off ($d=0$, $t=40$), the model gives exactly the Elo pregame "
                "probability — no game information yet, so only the prior matters."
            )

    # ── 13. Logit-probit bridge ───────────────────────────────────────────────
    with st.expander("13 · Connecting the Elo prior to the live model"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**The approximation:**")
            st.latex(
                r"\text{logit}(p) = \ln\!\frac{p}{1-p} \approx \frac{\pi}{\sqrt{3}}\,\Phi^{-1}(p)"
            )
            st.markdown("**Converting the Elo prior to a z-score:**")
            st.latex(
                r"z_{\text{prior}} = \text{logit}(p_0) \cdot \frac{\sqrt{3}}{\pi}"
            )
            st.markdown("**Combined live z-score:**")
            st.latex(
                r"z_{\text{total}} = \underbrace{\frac{d}{\sigma_s\sqrt{t}}}_{\text{game evidence}}"
                r"+ \underbrace{\text{logit}(p_0)\cdot\frac{\sqrt{3}}{\pi}}_{\text{Elo prior}}"
            )
            st.markdown("**Example:** $p_0 = 0.60$")
            st.latex(
                r"z_{\text{prior}} = \ln\!\tfrac{0.6}{0.4} \cdot \tfrac{\sqrt{3}}{\pi}"
                r"= 0.405 \times 0.551 = 0.223"
            )
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "The random-walk model uses **normal distribution** units (z-scores). "
                "The Elo model outputs a **logistic probability**. These live in slightly "
                "different mathematical spaces, so we need a bridge between them.\n\n"
                "The key fact: the logit function ($\\ln(p/(1-p))$) and the probit "
                "function ($\\Phi^{-1}(p)$, the inverse normal CDF) are very close to "
                "proportional — differing by the constant $\\pi/\\sqrt{3} \\approx 1.81$. "
                "This is the standard **logit-probit approximation**.\n\n"
                "In plain English: we're asking *'if the Elo model says 60%, where does "
                "that sit on the normal bell curve?'* The answer is about 0.22 standard "
                "deviations above center. We then add the score-differential z-score on "
                "top of that prior.\n\n"
                "This means:\n"
                "- If the game is tied ($d=0$), the live probability equals the Elo prior exactly.\n"
                "- A lead *adds* to the prior; a deficit *subtracts* from it.\n"
                "- As time runs out, the score evidence overwhelms the prior completely."
            )

    # ── 14. Upset detection ───────────────────────────────────────────────────
    with st.expander("14 · Upset detection: when the underdog flips"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Formal definition:**")
            st.latex(r"\text{UPSET ALERT if:} \quad p_0 < 0.5 \;\text{ and }\; p_{\text{live}} \geq 0.5")
            st.markdown(
                "where $p_0$ is the pregame Elo probability for team $A$ "
                "and $p_{\\text{live}}$ is the current in-game probability."
            )
            st.markdown("**Probability swing:**")
            st.latex(r"\Delta p = p_{\text{live}} - p_0")
            st.markdown(
                "Positive $\\Delta p$ means team $A$ has gained probability since tip-off. "
                "The upset alert fires at the exact moment $\\Delta p$ crosses the 50% threshold."
            )
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "An upset alert is exactly what it sounds like: the team that was supposed "
                "to lose is now more likely to win.\n\n"
                "The pregame model said the underdog had, say, a **38% chance**. "
                "But they went on a 12-0 run and now lead by 7 with 10 minutes left. "
                "The live model recalculates and says **61%** — the underdog has flipped "
                "to become the live favourite. That's the moment the alert fires.\n\n"
                "Why does this matter for the bracket? Because in a tournament, every "
                "team's path to the title depends on who comes out of each game. "
                "If the #12 seed is live-favourite over the #5 seed, the entire "
                "right side of that region's bracket just changed its expected shape.\n\n"
                "The probability swing $\\Delta p$ also tells you **how dramatic** the "
                "reversal is. A +35% swing (from 25% to 60%) is a much bigger moment "
                "than a +10% swing (from 45% to 55%)."
            )

    # ── 15. Live bracket impact ────────────────────────────────────────────────
    with st.expander("15 · Live bracket impact: upset propagation"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Process:**")
            st.markdown(
                "1. **Clone** the current Elo engine (copy all ratings).\n"
                "2. **Apply** the hypothetical game result (team A wins or team B wins).\n"
                "3. **Update** both teams' ratings via the Elo rule: $R' = R + K(S - E)$.\n"
                "4. **Re-seed** the 64-team bracket from the updated ratings.\n"
                "5. **Simulate** 5,000 Monte Carlo tournaments.\n"
                "6. **Compare** new title odds $\\hat{p}'$ to baseline $\\hat{p}$.\n"
                "7. **Report** $\\Delta = \\hat{p}' - \\hat{p}$ for each team."
            )
            st.latex(r"\Delta_i = \hat{p}'_i(\text{if upset}) - \hat{p}_i(\text{baseline})")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "The live bracket impact answers: *'if this upset actually holds, "
                "what happens to everybody's championship odds?'*\n\n"
                "It's a **counterfactual simulation**. We freeze the current state of "
                "all other teams, hypothetically record the current live game's outcome, "
                "and re-run the entire tournament 5,000 times with that one change.\n\n"
                "The upsets that matter most are ones where:\n"
                "- The losing team was a **deep title contender** (their odds drop sharply).\n"
                "- The winning team now has an **easier projected path** to the Final Four.\n"
                "- The winner's **next opponent** shifts because the bracket re-seeds.\n\n"
                "This is the same logic as financial scenario analysis: *'if this event "
                "occurs, how does the entire portfolio of outcomes shift?'* "
                "The answer is never just about the two teams in the game."
            )

    # ── 16. Parameters used ──────────────────────────────────────────────────
    with st.expander("16 · Model parameters used in this dashboard"):
        st.markdown(
            "**Elo / bracket model**\n\n"
            "| Parameter | Value | What it controls |\n"
            "|---|---|---|\n"
            "| $K$ (update factor) | 24 | How fast ratings respond to results |\n"
            "| Initial rating | 1500 | Starting Elo for all new teams |\n"
            "| Home advantage | 100 pts | Temporary boost for home team |\n"
            "| Elo scale | 400 | Sensitivity of win probability to rating gap |\n"
            "| Simulations (bracket) | 100,000 | Monte Carlo runs for bracket odds |\n"
            "| Simulations (impact) | 5,000 | Quick re-sim for live bracket impact |\n"
            "| Season | Nov 4, 2025 – Feb 23, 2026 | Data range for ratings |\n"
            "| Top $N$ seeded | 64 | Teams included in bracket simulation |\n\n"
            "**Live win probability model**\n\n"
            "| Parameter | Value | What it controls |\n"
            "|---|---|---|\n"
            r"| $\sigma_s$ (diffusion constant) | 2.0 pts/√min | Scoring uncertainty per unit time |"
            "\n"
            "| Prior bridge | $\\sqrt{3}/\\pi$ | Logit-to-probit conversion factor |\n"
            "| Live data TTL | 120 s | How often live games are re-fetched |"
        )
        st.markdown(
            "**Why K = 24?** Chess originally used K = 10 for experienced players. "
            "Basketball has more variance per game and a shorter season relative to chess, "
            "so a larger K is appropriate: ratings need to move faster to stay meaningful. "
            "K = 24 is a widely-used starting point for college basketball Elo models.\n\n"
            "**Why 1500?** It's arbitrary: only rating *differences* matter, not absolute values. "
            "1500 is the chess convention; you could start at 0 or 1000 and the win probabilities "
            "would be identical as long as all teams start at the same value.\n\n"
            "**Why 100 points for home advantage?** Empirical studies of college basketball "
            "suggest home teams win about 60–64% of games, corresponding to roughly 80–100 Elo points."
        )
