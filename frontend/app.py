"""
ChalkIQ — the favorites win
==============================
Run from the project root:
    streamlit run frontend/app.py
"""

import math as _math
import sys
from datetime import date, datetime as _datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
from src.live.model import live_win_prob, upset_alert, prob_swing
from src.ratings.elo import EloEngine
from src.utils.data import fetch_season
from src.utils.metrics import evaluate

# ── Constants ─────────────────────────────────────────────────────────────────

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

# Power conference team names as they appear in ESPN data.
# Used to filter CLV records to high-confidence matchups.
POWER_CONF_TEAMS: set[str] = {
    # Big Ten
    "Oregon Ducks","Washington Huskies","USC Trojans","UCLA Bruins",
    "Ohio State Buckeyes","Michigan Wolverines","Michigan State Spartans",
    "Penn State Nittany Lions","Indiana Hoosiers","Purdue Boilermakers",
    "Illinois Fighting Illini","Iowa Hawkeyes","Wisconsin Badgers",
    "Minnesota Golden Gophers","Northwestern Wildcats","Rutgers Scarlet Knights",
    "Nebraska Cornhuskers","Maryland Terrapins",
    # Big 12
    "Kansas Jayhawks","Kansas State Wildcats","Baylor Bears",
    "Texas Tech Red Raiders","Oklahoma State Cowboys","TCU Horned Frogs",
    "Iowa State Cyclones","West Virginia Mountaineers","Cincinnati Bearcats",
    "UCF Knights","Houston Cougars","BYU Cougars","Colorado Buffaloes",
    "Utah Utes","Arizona Wildcats","Arizona State Sun Devils",
    # SEC
    "Alabama Crimson Tide","Auburn Tigers","Florida Gators","Georgia Bulldogs",
    "Kentucky Wildcats","LSU Tigers","Mississippi State Bulldogs",
    "Ole Miss Rebels","South Carolina Gamecocks","Tennessee Volunteers",
    "Vanderbilt Commodores","Arkansas Razorbacks","Missouri Tigers",
    "Texas Longhorns","Texas A&M Aggies","Oklahoma Sooners",
    # ACC
    "Duke Blue Devils","North Carolina Tar Heels","NC State Wolfpack",
    "Virginia Cavaliers","Virginia Tech Hokies","Clemson Tigers",
    "Syracuse Orange","Pittsburgh Panthers","Miami Hurricanes",
    "Wake Forest Demon Deacons","Louisville Cardinals","Georgia Tech Yellow Jackets",
    "Boston College Eagles","Notre Dame Fighting Irish","Stanford Cardinal",
    "California Golden Bears","SMU Mustangs","Florida State Seminoles",
    # Big East
    "UConn Huskies","Georgetown Hoyas","St. John's Red Storm",
    "Seton Hall Pirates","Providence Friars","Xavier Musketeers",
    "Marquette Golden Eagles","DePaul Blue Demons","Butler Bulldogs",
    "Creighton Bluejays","Villanova Wildcats",
}

DIVISION_CONFIG = {
    "mens": {
        "label":        "Men's CBB",
        "cache_dir":    str(ROOT / "data" / "raw" / "mens"),
        "emoji":        "🏀",
        "color":        NORD["frost1"],
        "light":        NORD["bg1"],
        "season_start": date(2025, 11, 4),
        "season_end":   date.today(),
        "is_nba":       False,
        "avg_total":    140.0,   # avg combined points per game
    },
    "womens": {
        "label":        "Women's CBB",
        "cache_dir":    str(ROOT / "data" / "raw" / "womens"),
        "emoji":        "🏀",
        "color":        NORD["purple"],
        "light":        NORD["bg1"],
        "season_start": date(2025, 11, 4),
        "season_end":   date.today(),
        "is_nba":       False,
        "avg_total":    130.0,
    },
}

N_SIMS = 100_000

# ── Timezone options ───────────────────────────────────────────────────────────
_TZ_OPTIONS = {
    "Eastern (ET)":  "America/New_York",
    "Central (CT)":  "America/Chicago",
    "Mountain (MT)": "America/Denver",
    "Pacific (PT)":  "America/Los_Angeles",
    "UTC":           "UTC",
}

# ── Gambling / prediction helpers ──────────────────────────────────────────────

def _predicted_score(p_home: float, avg_total: float = 140.0) -> tuple[int, int]:
    """Predicted home and away points from home win probability."""
    p = max(1e-7, min(1 - 1e-7, p_home))
    logit_p  = _math.log(p / (1 - p))
    probit_p = logit_p * _math.sqrt(3) / _math.pi
    margin   = probit_p * _math.sqrt(40.0) * 2.0   # SIGMA_game ≈ 12.65 pts
    return round((avg_total + margin) / 2), round((avg_total - margin) / 2)


def _prob_to_american(p: float, vig: float = 0.05) -> int:
    """Convert probability to American moneyline with standard vig."""
    p = max(0.01, min(0.99, p))
    p_vig = p + (1 - p) * vig / 2          # inflate by half the juice
    if p_vig >= 0.5:
        return -round(p_vig / (1 - p_vig) * 100)
    return round((1 - p_vig) / p_vig * 100)


def _american_to_decimal(ml: int) -> float:
    """American moneyline → decimal odds."""
    return (ml / 100 + 1.0) if ml > 0 else (100 / abs(ml) + 1.0)


def _fmt_ml(ml: int) -> str:
    return f"+{ml}" if ml > 0 else str(ml)


# ── Data loading (cached per division) ────────────────────────────────────────

@st.cache_resource(ttl=3600, show_spinner="Loading game data…")
def load_engine(division: str) -> EloEngine:
    cfg = DIVISION_CONFIG[division]
    games = fetch_season(
        cfg["season_start"], cfg["season_end"],
        cache_dir=cfg["cache_dir"],
        division=division,
        verbose=False,
    )
    engine = EloEngine(k=24.0, home_advantage=100.0)
    engine.process_games(games)
    return engine


@st.cache_data(show_spinner="Simulating bracket…")
def load_bracket_data(division: str, _injury_override_key: tuple = ()):
    """
    _injury_override_key: tuple of (team_id, delta) pairs used only to bust
    the Streamlit cache when injury adjustments change. The actual overrides
    are reconstructed inside the function from the key.
    """
    base_engine = load_engine(division)
    if _injury_override_key:
        sim_engine = base_engine.adjusted_copy(dict(_injury_override_key))
    else:
        sim_engine = base_engine

    rankings = sim_engine.rankings()
    regions  = assign_seeds(rankings)

    # Build bracket in proper seed order: East slots 0-15, West 16-31,
    # South 32-47, Midwest 48-63 — so #1 seed plays #16, not #2.
    bracket_order: list[str] = []
    for region_name in REGIONS:
        bracket_order.extend(region_bracket_order(regions[region_name]))

    adv_odds = round_advancement_odds(
        seeded_teams=bracket_order,
        win_prob_fn=sim_engine.win_prob,
        n_sims=N_SIMS,
        seed=42,
    )
    # Round 6 = championship win probability
    champ_odds = {tid: adv_odds[tid].get(6, 0.0) for tid in bracket_order}

    return regions, adv_odds, champ_odds


@st.cache_data(ttl=300, show_spinner=False)
def load_future_games(division: str, for_date: date) -> list[dict]:
    """Fetch scheduled games for a date, cached 5 min."""
    from src.live.feed import fetch_other_games
    return fetch_other_games(
        division=division,
        for_date=for_date,
        status_filter={"STATUS_SCHEDULED", "STATUS_PREGAME"},
    )




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

engine  = load_engine(division)
metrics = evaluate(engine.history)

# ── Injury adjustments ───────────────────────────────────────────────────────
# Load alerts from the signals store and build Elo overrides for currently-OUT
# players. These adjustments ripple through rankings, bracket, odds edge, and
# NCAA 361. The base engine is never mutated.

def _build_injury_overrides(eng: "EloEngine") -> dict[str, float]:
    """
    Read injury alerts, collapse to latest status per player, and return
    {team_id: total_elo_delta} for players currently Out / Doubtful / IR.
    """
    from src.signals.injuries import load_alerts as _load_inj, estimate_impact, _OUT_STATUSES
    _raw = _load_inj()
    if not _raw:
        return {}
    # Latest alert per player (last entry wins)
    _latest: dict[str, dict] = {}
    for _a in _raw:
        _pid = _a.get("player_id", "")
        if _pid:
            _latest[_pid] = _a
    overrides: dict[str, float] = {}
    for _a in _latest.values():
        if _a.get("new_status") not in _OUT_STATUSES:
            continue
        _tid = _a.get("team_id", "")
        if not _tid:
            continue
        _delta = estimate_impact(
            player_name=_a.get("player_name", ""),
            position=_a.get("position", ""),
            team_elo=eng.rating(_tid),
            status=_a.get("new_status", ""),
        )
        if _delta != 0.0:
            overrides[_tid] = overrides.get(_tid, 0.0) + _delta
    return overrides

_raw_inj_overrides = _build_injury_overrides(engine)

# Sidebar toggle — only shown when there are active injury adjustments
_apply_injuries = True
if _raw_inj_overrides:
    with st.sidebar:
        st.markdown("---")
        _n_teams = len(_raw_inj_overrides)
        _apply_injuries = st.toggle(
            f"Injury adjustments ({_n_teams} team{'s' if _n_teams > 1 else ''})",
            value=True,
            key="apply_injuries",
            help="Applies Elo penalties for currently-out star players to rankings, "
                 "bracket odds, and edge calculations. Turn off to see unadjusted model.",
        )
        if _apply_injuries:
            for _tid, _delta in sorted(_raw_inj_overrides.items(), key=lambda x: x[1]):
                _tname = engine.names.get(_tid, _tid)
                st.caption(f"⚠ {_tname}: {_delta:+.0f} Elo pts")

_inj_overrides = _raw_inj_overrides if _apply_injuries else {}
_inj_override_key = tuple(sorted(_inj_overrides.items()))

# adj_engine: ratings-adjusted copy used for all win-probability computations
adj_engine = engine.adjusted_copy(_inj_overrides) if _inj_overrides else engine

rankings                   = adj_engine.rankings()
regions, adv_odds, champ_odds = load_bracket_data(division, _inj_override_key)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_rank, tab_bracket, tab_eval, tab_math, tab_ncaa361, tab_signals, tab_backtest, tab_players, tab_sources = st.tabs([
    "📊  Power Rankings",
    "🏆  Bracket",
    "📈  Model Evaluation",
    "📐  Math",
    "📈  NCAA 361",
    "🔍  Signals",
    "📉  Backtest",
    "🏀  Players",
    "📚  Sources",
])


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 1 — Power Rankings
# ════════════════════════════════════════════════════════════════════════════ #

with tab_rank:
    st.subheader(f"Power Rankings | {cfg['label']} Division")
    st.caption(f"All Division I teams ranked by Elo. {len(rankings)} teams tracked through {date.today().strftime('%b %d, %Y')}.")

    # Build per-team GP, Win%, and SoS from game history
    # SoS = average Elo of all opponents faced
    _rating_map = {tid: rating for tid, _, rating in rankings}
    _rank_rec: dict[str, dict] = {}
    for _rg in engine.history:
        for _rtid, _ropp_id, _rfor, _ragainst in [
            (_rg["home_id"], _rg["away_id"], _rg["home_score"], _rg["away_score"]),
            (_rg["away_id"], _rg["home_id"], _rg["away_score"], _rg["home_score"]),
        ]:
            if _rtid not in _rank_rec:
                _rank_rec[_rtid] = {"gp": 0, "wins": 0, "opp_elos": []}
            _rank_rec[_rtid]["gp"] += 1
            if _rfor > _ragainst:
                _rank_rec[_rtid]["wins"] += 1
            if _ropp_id in _rating_map:
                _rank_rec[_rtid]["opp_elos"].append(_rating_map[_ropp_id])

    # Compute adjusted efficiency (KenPom-style iterative adjustment)
    from src.utils.efficiency import compute_efficiency
    _eff = compute_efficiency(engine.history)

    all_rows = []
    for rank, (tid, name, rating) in enumerate(rankings, 1):
        _rec  = _rank_rec.get(tid, {"gp": 0, "wins": 0, "opp_elos": []})
        _gp   = _rec["gp"]
        _wp   = _rec["wins"] / _gp if _gp else 0.0
        _elos = _rec["opp_elos"]
        _sos  = round(sum(_elos) / len(_elos), 1) if _elos else 0.0
        _e    = _eff.get(tid, {})
        all_rows.append({
            "Rank":    rank,
            "Team":    name,
            "Elo":     round(rating, 1),
            "GP":      _gp,
            "Win%":    _wp,
            "SoS":     _sos,
            "Pace":    _e.get("pace"),
            "Adj Off": _e.get("adj_off"),
            "Adj Def": _e.get("adj_def"),
            "Net":     _e.get("net_adj"),
            "R32":     adv_odds.get(tid, {}).get(2, 0),
            "S16":     adv_odds.get(tid, {}).get(3, 0),
            "E8":      adv_odds.get(tid, {}).get(4, 0),
            "FF":      adv_odds.get(tid, {}).get(5, 0),
            "Title":   champ_odds.get(tid, 0),
        })

    df_all   = pd.DataFrame(all_rows).set_index("Rank")
    pct_cols = ["R32", "S16", "E8", "FF", "Title"]

    def _fmt_pct(v):
        if isinstance(v, float) and v > 0:
            return f"{v:.3%}"
        return "-"

    def _fmt_eff(v):
        return f"{v:.1f}" if v is not None else "-"

    styled_all = (
        df_all.style
        .format({c: _fmt_pct for c in pct_cols})
        .format({"Win%": "{:.1%}", "Elo": "{:.1f}", "GP": "{:.0f}", "SoS": "{:.1f}"})
        .format({"Adj Off": _fmt_eff, "Adj Def": _fmt_eff, "Net": _fmt_eff, "Pace": _fmt_eff})
        .background_gradient(subset=pct_cols, cmap="Blues", vmin=0, vmax=0.5)
        .background_gradient(subset=["SoS"], cmap="Oranges", vmin=1450, vmax=1600)
        .background_gradient(subset=["Pace"], cmap="Purples", vmin=120, vmax=165)
        .background_gradient(subset=["Adj Off"], cmap="Greens", vmin=60, vmax=95)
        .background_gradient(subset=["Adj Def"], cmap="Reds_r", vmin=60, vmax=95)
        .bar(subset=["Elo"], color=[light, color])
        .set_properties(**{"font-size": "13px"})
    )

    st.caption(
        "**Elo** — team strength rating; 1500 = average, updates after every game via win/loss + margin of victory. "
        "**Pace** — avg total points per game (proxy for possessions per 40 min). "
        "**SoS** — avg Elo of opponents faced. "
        "**R32 / S16 / E8 / FF / Title** — tournament advancement odds from 100,000 Monte Carlo simulations."
    )
    col_tbl, col_chart = st.columns([3, 2])
    with col_tbl:
        st.dataframe(styled_all, height=1800, width="stretch")

    with col_chart:
        top20_names = [name for _, name, _ in rankings[:20]]
        top20_odds  = [champ_odds.get(tid, 0) for tid, _, _ in rankings[:20]]
        fig_bar = plotly_odds_bar(top20_names[::-1], top20_odds[::-1], color)
        st.plotly_chart(fig_bar, width="stretch")
        st.caption(f"Top 20 championship odds from {N_SIMS:,} Monte Carlo bracket simulations.")

        # Efficiency scatter: Adj Def (x) vs Adj Off (y)
        _eff_teams   = [(tid, name, _eff[tid]) for tid, name, _ in rankings if tid in _eff]
        _scatter_top = _eff_teams[:120]  # top 120 by Elo for readability
        if _scatter_top:
            _sx   = [e["adj_def"] for _, _, e in _scatter_top]
            _sy   = [e["adj_off"] for _, _, e in _scatter_top]
            _slbl = [n for _, n, _ in _scatter_top]
            _snet = [e["net_adj"] for _, _, e in _scatter_top]
            _scdata = [[net, i + 1] for i, net in enumerate(_snet)]

            _fig_eff = go.Figure(go.Scatter(
                x=_sx, y=_sy,
                mode="markers+text",
                text=_slbl,
                textposition="top center",
                textfont=dict(size=7),
                customdata=_scdata,
                marker=dict(
                    size=8,
                    color=_snet,
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Net Adj"),
                    line=dict(width=0.5, color=NORD["bg3"]),
                ),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Adj Off: %{y:.1f}<br>"
                    "Adj Def: %{x:.1f}<br>"
                    "Net Adj: %{customdata[0]:+.1f}<br>"
                    "Rank: #%{customdata[1]}<br>"
                    "<extra></extra>"
                ),
            ))
            # Quadrant lines at league medians
            _med_def = sorted(_sx)[len(_sx) // 2]
            _med_off = sorted(_sy)[len(_sy) // 2]
            _fig_eff.add_vline(x=_med_def, line_color=NORD["bg3"], line_dash="dot")
            _fig_eff.add_hline(y=_med_off, line_color=NORD["bg3"], line_dash="dot")
            _fig_eff.update_layout(
                xaxis=dict(title="Adj Def (lower = better defense)", autorange="reversed"),
                yaxis=dict(title="Adj Off (higher = better offense)"),
                height=480,
                margin=dict(l=60, r=20, t=30, b=50),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.markdown("**Offense vs Defense (Top 120)**")
            st.caption(
                "Net Adj = Adj Off - Adj Def. "
                "Adj Off = Raw Off x (league avg / opp Adj Def), iterated 20x with GP-weighted opponent averages, renormalized each pass. "
                "Adj Def = Raw Def x (league avg / opp Adj Off), iterated 20x with GP-weighted opponent averages, renormalized each pass."
            )
            st.plotly_chart(_fig_eff, width="stretch")


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

    st.markdown("---")


# TAB 9 — Math
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

    # ── 12. Adjusted efficiency ───────────────────────────────────────────────
    with st.expander("12 · Adjusted offensive and defensive efficiency"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Raw efficiency:**")
            st.latex(r"\text{Raw Off}_i = \frac{1}{G_i}\sum_{g \in G_i} \text{pts scored}_g")
            st.latex(r"\text{Raw Def}_i = \frac{1}{G_i}\sum_{g \in G_i} \text{pts allowed}_g")
            st.markdown("**Iterative opponent adjustment (20 passes):**")
            st.latex(
                r"\text{Adj Off}_i^{(k+1)} = \text{Raw Off}_i \cdot "
                r"\frac{\bar{\mu}}{\sum_j w_j \cdot \text{Adj Def}_j^{(k)} / \sum_j w_j}"
            )
            st.latex(
                r"\text{Adj Def}_i^{(k+1)} = \text{Raw Def}_i \cdot "
                r"\frac{\bar{\mu}}{\sum_j w_j \cdot \text{Adj Off}_j^{(k)} / \sum_j w_j}"
            )
            st.markdown(
                "where $w_j = G_j$ (games played by opponent $j$) and "
                "$\\bar{\\mu}$ is the league average. "
                "Both renormalized to $\\bar{\\mu}$ after each pass."
            )
            st.markdown("**Net rating:**")
            st.latex(r"\text{Net}_i = \text{Adj Off}_i - \text{Adj Def}_i")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Raw scoring averages are misleading: a team that scores 85 ppg against "
                "elite defenses is far more impressive than one that scores 85 against "
                "bottom-feeders.\n\n"
                "Adjusted efficiency corrects for that. Each team's offense is scaled up "
                "if they played tough defenses, and scaled down if their schedule was weak. "
                "Same for defense.\n\n"
                "**Weighting by games played** means opponents with more data pull more "
                "weight in the adjustment — a team with 30 games is more reliable evidence "
                "than a team with 6.\n\n"
                "After 20 iterative passes the numbers converge. The result is a KenPom-style "
                "efficiency rating built entirely from ESPN score data — no possession "
                "tracking required.\n\n"
                "**Net rating** is the single best summary: positive = better offense than defense, "
                "which is what elite teams look like."
            )

    # ── 13. Closing line value ────────────────────────────────────────────────
    with st.expander("13 · Closing Line Value (CLV)"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Implied probability from American moneyline:**")
            st.latex(
                r"p_{\text{impl}} = \begin{cases}"
                r"\dfrac{|ML|}{|ML|+100} & ML < 0 \text{ (favorite)} \\[8pt]"
                r"\dfrac{100}{ML+100} & ML > 0 \text{ (underdog)}"
                r"\end{cases}"
            )
            st.markdown("**Vig removal (two-sided market):**")
            st.latex(
                r"p_{\text{fair}} = \frac{p_{\text{impl,home}}}{p_{\text{impl,home}} + p_{\text{impl,away}}}"
            )
            st.markdown("**CLV vs closing line:**")
            st.latex(r"\text{CLV} = p_{\text{model}} - p_{\text{fair,close}}")
            st.markdown("**CLV vs opening line:**")
            st.latex(r"\text{CLV}_{\text{open}} = p_{\text{model}} - p_{\text{fair,open}}")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "The closing line is the sharpest price signal in sports betting. "
                "By the time a game starts, the market has absorbed all publicly "
                "available information — sharp bettors, injury reports, weather, everything.\n\n"
                "**CLV** measures whether our model was smarter than the closing market. "
                "Positive CLV means our model assigned a higher probability than the "
                "book's final implied probability — we would have had an edge.\n\n"
                "**Why it matters:** A model that consistently beats closing lines "
                "has genuine predictive signal. Win rate alone can be misleading "
                "(you can win 55% betting only huge favorites). CLV is harder to fake.\n\n"
                "**Vig removal** strips out the bookmaker's margin (the juice) so we're "
                "comparing fair probabilities, not inflated ones."
            )

    # ── 14. Line movement awareness ───────────────────────────────────────────
    with st.expander("14 · Line Movement Awareness (LMA)"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Line move size:**")
            st.latex(r"\Delta p = p_{\text{fair}}(t_2) - p_{\text{fair}}(t_1)")
            st.markdown("**Sharp classification threshold:**")
            st.latex(r"|\Delta p| \geq 0.04 \implies \text{likely sharp money}")
            st.markdown("**Reverse line movement:**")
            st.markdown(
                "Public betting % favors team A, but the line moves *against* team A "
                "(toward team B). This suggests sharp bettors disagree with the public."
            )
            st.latex(
                r"\text{RLM} = \mathbf{1}\{\text{public favors A}\} \cap \mathbf{1}\{\Delta p_A < 0\}"
            )
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Sportsbooks move lines for two reasons: **public money** (casual bettors "
                "all piling on one side) or **sharp money** (professional bettors with "
                "genuine edge).\n\n"
                "A line move of 4%+ in implied probability is large enough that it's "
                "unlikely to be casual public action — it's more consistent with a "
                "sharp bet that forced the book to reprice.\n\n"
                "**Reverse line movement** is the strongest signal: the public is "
                "overwhelmingly on one team, but the line moves the other way. "
                "That means the book is intentionally moving toward the sharp side, "
                "accepting liability on the public side because they trust the sharps more.\n\n"
                "LMA doesn't tell you who will win. It tells you where the informed "
                "money is — which is useful context when our model also agrees."
            )

    # ── 15. Backtesting and edge ──────────────────────────────────────────────
    with st.expander("15 · Backtesting and edge over breakeven"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Breakeven win rate at -110 juice:**")
            st.latex(r"p_{\text{break}} = \frac{100}{110 + 100} \approx 52.4\%")
            st.markdown("**Model edge on a bet:**")
            st.latex(r"\text{edge} = p_{\text{model}} - p_{\text{break}}")
            st.markdown("**P&L per bet (flat $100 stake at -110):**")
            st.latex(
                r"\text{P\&L} = \begin{cases} +\$90.91 & \text{if win} \\ -\$100 & \text{if loss} \end{cases}"
            )
            st.markdown("**ROI over } $n$ bets:**")
            st.latex(r"\text{ROI} = \frac{\sum \text{P\&L}}{n \times \text{stake}}")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "At standard -110 juice (bet $110 to win $100), you need to win "
                "**52.4%** of bets just to break even. Anything above that is profit.\n\n"
                "The backtest walks through every historical game in order — "
                "ratings build from scratch exactly as they would in real time, "
                "no future data leaks in. Whenever the model sees an edge above "
                "the threshold, it simulates placing a flat $100 bet.\n\n"
                "**Edge bucket analysis** groups bets by how much edge the model "
                "claimed. If the model is well-calibrated, higher-edge buckets "
                "should show higher win rates — that's the key validation check.\n\n"
                "**Important caveat:** the backtest assumes -110 is always available. "
                "Real books move lines, may not offer the price you want, and will "
                "limit winning accounts. The backtest measures signal quality, "
                "not guaranteed real-world returns."
            )

    # ── 16. Projected score ───────────────────────────────────────────────────
    with st.expander("16 · Projected score from win probability"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**The chain of formulas**")
            st.markdown("**Step 1.** Start with the Elo win probability $p$ for the home team.")
            st.latex(r"p = \frac{1}{1 + 10^{(R_B - R_A)/400}}")
            st.markdown(
                "**Step 2.** Convert $p$ to a z-score (standard deviations from zero) "
                "using the logit-to-probit bridge from section 13."
            )
            st.latex(
                r"z = \frac{\sqrt{3}}{\pi}\,\ln\!\left(\frac{p}{1-p}\right)"
            )
            st.markdown(
                "**Step 3.** Scale by the game-level scoring volatility $\\sigma_{\\text{game}}$."
            )
            st.latex(
                r"\sigma_{\text{game}} = \sigma_s \cdot \sqrt{T}"
                r"\quad \sigma_s = 2.0\;\text{pts/}\sqrt{\text{min}},\;"
                r"T = 40\;\text{min}"
            )
            st.latex(r"\sigma_{\text{game}} \approx 12.65\;\text{pts}")
            st.latex(r"\text{Expected margin} = z \cdot \sigma_{\text{game}}")
            st.markdown(
                "**Step 4.** Split the expected margin around the average total score."
            )
            st.latex(
                r"\hat{S}_{\text{home}} = \frac{\bar{T} + \text{margin}}{2}"
                r",\quad"
                r"\hat{S}_{\text{away}} = \frac{\bar{T} - \text{margin}}{2}"
            )
            st.latex(r"\bar{T} = 140\;\text{pts (college basketball average total)}")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "The win probability tells us *who* is more likely to win, but not *by how much*. "
                "To get a score, we reverse-engineer the implied margin.\n\n"
                "The key idea is that score differences in basketball behave like a random walk: "
                "each possession adds a small, roughly-equal amount of uncertainty. "
                "The further the game goes, the wider the possible range of final margins. "
                "This is captured by $\\sigma_{\\text{game}}$, the total scoring volatility "
                "over 40 minutes.\n\n"
                "The logit-to-probit conversion (the $\\sqrt{3}/\\pi$ bridge) translates the "
                "win probability — which is in 'logistic space' — into 'normal space' "
                "so we can multiply it by $\\sigma_{\\text{game}}$ directly.\n\n"
                "**Worked example:** Suppose team A has a **72%** win probability.\n\n"
                "- $z = (\\sqrt{3}/\\pi)\\,\\ln(0.72/0.28) \\approx 0.52$\n"
                "- margin $= 0.52 \\times 12.65 \\approx 6.6$ pts\n"
                "- scores $= (140 + 6.6)/2 = 73$ and $(140 - 6.6)/2 = 67$\n\n"
                "**Caveats:** This is a *probabilistic mean*, not a guarantee. "
                "The average total of 140 is a long-run constant; any individual game "
                "can run far higher or lower. Treat the predicted score as a rough "
                "central estimate, not a precise forecast."
            )
            st.markdown("**What each parameter controls**")
            st.markdown(
                "| Parameter | Value | Effect |\n"
                "|---|---|---|\n"
                "| $\\sigma_s$ | 2.0 pts/√min | Wider → higher-scoring games assumed |\n"
                "| $T$ | 40 min | Regulation game length |\n"
                "| $\\bar{T}$ | 140 pts | Average total; shifts both scores equally |\n"
                "| $z$ | varies | Higher → bigger predicted margin |"
            )

    # ── 17. Model parameters ─────────────────────────────────────────────────
    with st.expander("17 · Model parameters used in this dashboard"):
        st.markdown(
            "**Elo / bracket engine**\n\n"
            "| Parameter | Value | What it controls |\n"
            "|---|---|---|\n"
            "| $K$ (update factor) | 24 | How fast ratings respond to results |\n"
            "| Initial rating | 1500 | Starting Elo for all new teams |\n"
            "| Home advantage | 100 pts | Temporary boost for home team |\n"
            "| Elo scale | 400 | Sensitivity of win probability to rating gap |\n"
            "| Simulations (bracket) | 100,000 | Monte Carlo runs for bracket odds |\n"
            "| Teams seeded | 64 | Teams included in bracket simulation |\n\n"
            "**Adjusted efficiency engine**\n\n"
            "| Parameter | Value | What it controls |\n"
            "|---|---|---|\n"
            "| Iterations | 20 | Convergence passes for opponent adjustment |\n"
            "| Min games | 5 | Teams excluded below this threshold |\n"
            "| Opponent weight | games played | More data = more pull in the adjustment |\n\n"
            "**Signals engine**\n\n"
            "| Parameter | Value | What it controls |\n"
            "|---|---|---|\n"
            "| Edge threshold | 5% | Min model vs line gap to flag as EDGE |\n"
            "| Sharp move threshold | 4% | Min implied prob shift to classify as sharp |\n"
            "| Breakeven (-110) | 52.4% | Win rate needed to profit at standard juice |\n"
            "| Backtest stake | $100 flat | Per-bet size in walk-forward simulation |\n"
        )
        st.markdown(
            "**Why K = 24?** Basketball has more variance per game and a shorter season "
            "than chess, so ratings need to move faster. K = 24 is a widely-used starting "
            "point for college basketball Elo models.\n\n"
            "**Why 1500?** Only rating *differences* matter. You could start at 0 or 1000 "
            "and win probabilities would be identical.\n\n"
            "**Why 100 pts home advantage?** Empirical college basketball data suggests "
            "home teams win ~60-64% of games, corresponding to ~80-100 Elo points.\n\n"
            "**Why 5% edge threshold?** Below 5%, the signal is too close to noise "
            "given typical model error. At 5%+, the expected value is meaningfully positive "
            "even accounting for model uncertainty."
        )


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 8 — Players
# ════════════════════════════════════════════════════════════════════════════ #

with tab_players:
    st.subheader("Players | CBB Player Ratings, Trajectories, Prop Edge, Injury Impact")
    st.caption(
        "Player ratings use the same Elo framework as teams. "
        "Each game a player's Game Score (Hollinger) is normalized to a [0,1] outcome "
        "and their rating updates: R' = R + K*(outcome - expected). "
        "K=16 (lower than team K=24 because individual games are noisier). "
        "Ratings update after every game in chronological order."
    )

    _box_dir = ROOT / "data" / "raw" / division / "boxscores"
    _box_count = len(list(_box_dir.glob("*.json"))) if _box_dir.exists() else 0

    @st.cache_resource(show_spinner="Building player ratings...")
    def _build_player_engine(box_dir: str, _n_files: int) -> "PlayerEloEngine | None":
        from src.players.engine import PlayerEloEngine, load_boxscores
        games = load_boxscores(box_dir)
        if not games:
            return None
        engine = PlayerEloEngine()
        engine.process_games(games)
        return engine

    _p_engine = _build_player_engine(str(_box_dir), _box_count)

    if _p_engine is None or not _p_engine.ratings:
        st.info(
            "No player data yet. Run the box score fetcher first:\n\n"
            "```\npython scripts/fetch_boxscores.py --seasons 2026\n```"
        )
    else:
        _p_tab_top, _p_tab_traj, _p_tab_inj, _p_tab_prop = st.tabs([
            "Top Players", "Season Trajectory", "Injury Impact", "Prop Edge"
        ])

        # ── Top Players ───────────────────────────────────────────────────
        with _p_tab_top:
            st.caption(
                "**Rating** — Elo rating (1500 = average). "
                "**GmSc** — Hollinger Game Score: PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) "
                "+ 0.7*OREB + 0.3*DREB + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TO. "
                "Elite game = 25+, average = ~8. "
                "**Trend** — rating change over last 5 games (positive = improving form)."
            )
            _gc = _p_engine.player_game_counts()

            _p_col1, _p_col2, _p_col3 = st.columns([2, 2, 2])
            with _p_col1:
                _p_pos_filter = st.selectbox("Position", ["All", "G", "F", "C"], key="p_pos")
            with _p_col2:
                _p_min_gp = st.slider("Min games played", 1, 20, 5, key="p_mingp")
            with _p_col3:
                _p_top_n = st.slider("Show top N", 25, 200, 100, step=25, key="p_topn")

            _pos_arg = None if _p_pos_filter == "All" else _p_pos_filter
            _top = _p_engine.top_players(n=_p_top_n, min_games=_p_min_gp, position=_pos_arg)

            _p_rows = []
            for rank, (pid, name, rating, tid, pos) in enumerate(_top, 1):
                _hist = _p_engine.player_history(pid)
                _avg_gmsc = sum(r["game_score"] for r in _hist) / len(_hist) if _hist else 0
                _avg_off  = sum(r.get("off_score", 0) for r in _hist) / len(_hist) if _hist else 0
                _avg_pts  = sum(r["pts"] for r in _hist) / len(_hist) if _hist else 0
                _avg_reb  = sum(r["reb"] for r in _hist) / len(_hist) if _hist else 0
                _avg_ast  = sum(r["ast"] for r in _hist) / len(_hist) if _hist else 0
                # Trend: rating change over last 5 games
                _recent5 = _hist[-5:]
                _trend = (_recent5[-1]["rating_post"] - _recent5[0]["rating_pre"]) if len(_recent5) >= 2 else 0
                _team_name = engine.names.get(tid, tid)
                _p_rows.append({
                    "#":        rank,
                    "Player":   name,
                    "Team":     _team_name,
                    "Pos":      pos,
                    "GP":       _gc.get(pid, 0),
                    "Rating":   round(rating, 1),
                    "Avg GmSc": round(_avg_gmsc, 1),
                    "Avg OFF":  round(_avg_off, 1),
                    "Avg PTS":  round(_avg_pts, 1),
                    "Avg REB":  round(_avg_reb, 1),
                    "Avg AST":  round(_avg_ast, 1),
                    "Trend":    round(_trend, 1),
                })

            _df_players = pd.DataFrame(_p_rows)

            def _color_trend(val):
                if isinstance(val, float) and val > 3:
                    return f"color: {NORD['green']}; font-weight: bold"
                if isinstance(val, float) and val < -3:
                    return f"color: {NORD['red']}"
                return ""

            st.dataframe(
                _df_players.style
                    .map(_color_trend, subset=["Trend"])
                    .background_gradient(subset=["Rating"], cmap="Blues", vmin=1400, vmax=1700)
                    .background_gradient(subset=["Avg GmSc"], cmap="Greens", vmin=0, vmax=25),
                width="stretch",
                hide_index=True,
            )

            # ── Player Off vs Defense scatter (top 200) ───────────────────
            _p_scatter_all = _p_engine.top_players(n=200, min_games=_p_min_gp, position=_pos_arg)
            if len(_p_scatter_all) >= 5:
                _psx, _psy, _pslbl, _psrating, _pscdata = [], [], [], [], []
                for _rank_i, (_pid, _pname, _prating, _ptid, _ppos) in enumerate(_p_scatter_all, 1):
                    _phist = _p_engine.player_history(_pid)
                    if not _phist:
                        continue
                    _pn = len(_phist)
                    _pavg_pts = sum(r["pts"] for r in _phist) / _pn
                    _pavg_ast = sum(r["ast"] for r in _phist) / _pn
                    _pavg_stl = sum(r["stl"] for r in _phist) / _pn
                    _pavg_blk = sum(r["blk"] for r in _phist) / _pn
                    _pavg_reb = sum(r["reb"] for r in _phist) / _pn
                    _p_off = round(_pavg_pts + 0.7 * _pavg_ast, 2)
                    _p_def = round(_pavg_stl + 0.7 * _pavg_blk + 0.3 * _pavg_reb, 2)
                    _psx.append(_p_def)
                    _psy.append(_p_off)
                    _pslbl.append(_pname)
                    _psrating.append(_prating)
                    _pscdata.append([round(_prating, 1), _rank_i])

                _fig_peff = go.Figure(go.Scatter(
                    x=_psx, y=_psy,
                    mode="markers+text",
                    text=_pslbl,
                    textposition="top center",
                    textfont=dict(size=7),
                    customdata=_pscdata,
                    marker=dict(
                        size=8,
                        color=_psrating,
                        colorscale="RdYlGn",
                        showscale=True,
                        colorbar=dict(title="Elo Rating"),
                        cmin=1450,
                        cmax=1700,
                        line=dict(width=0.5, color=NORD["bg3"]),
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Off Score: %{y:.1f}<br>"
                        "Def Score: %{x:.1f}<br>"
                        "Elo Rating: %{customdata[0]:.1f}<br>"
                        "Rank: #%{customdata[1]}<br>"
                        "<extra></extra>"
                    ),
                ))
                _pmed_def = sorted(_psx)[len(_psx) // 2]
                _pmed_off = sorted(_psy)[len(_psy) // 2]
                _fig_peff.add_vline(x=_pmed_def, line_color=NORD["bg3"], line_dash="dot")
                _fig_peff.add_hline(y=_pmed_off, line_color=NORD["bg3"], line_dash="dot")
                _fig_peff.update_layout(
                    xaxis=dict(title="Def Score (STL + 0.7*BLK + 0.3*REB per game, higher = better)"),
                    yaxis=dict(title="Off Score (PTS + 0.7*AST per game)"),
                    height=520,
                    margin=dict(l=60, r=20, t=30, b=50),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.markdown("**Offense vs Defense (Top 200)**")
                st.caption(
                    "Off Score = PTS + 0.7 x AST per game (scoring + playmaking). "
                    "Def Score = STL + 0.7 x BLK + 0.3 x REB per game (defensive contribution). "
                    "Upper-right = two-way impact player. Color = Elo Rating. "
                    "Hover for Elo Rating + rank."
                )
                st.plotly_chart(_fig_peff, width="stretch")

        # ── Season Trajectory ─────────────────────────────────────────────
        with _p_tab_traj:
            st.caption(
                "Select a player to see their Game Score and Elo rating trend over the season. "
                "Rolling 5-game average smooths out single-game variance. "
                "Rising rating = consistent above-average performances."
            )
            _all_players = _p_engine.top_players(n=500, min_games=1)
            _player_options = {f"{name} ({_p_engine.teams.get(pid,'')})": pid
                               for pid, name, *_ in _all_players}
            _selected_label = st.selectbox("Select player", list(_player_options.keys()), key="p_traj_sel")
            _selected_pid   = _player_options.get(_selected_label, "")

            if _selected_pid:
                _traj = _p_engine.season_trajectory(_selected_pid)
                if _traj:
                    _t_dates = [t["date"] for t in _traj]
                    _t_gmsc  = [t["game_score"] for t in _traj]
                    _t_roll  = [t["rolling_avg"] for t in _traj]
                    _t_rat   = [t["rating"] for t in _traj]

                    _fig_traj = go.Figure()
                    _fig_traj.add_trace(go.Bar(
                        x=_t_dates, y=_t_gmsc,
                        name="Game Score",
                        marker_color="rgba(136,192,208,0.6)",
                        hovertemplate="%{x}<br>GmSc: %{y:.1f}<extra></extra>",
                    ))
                    _fig_traj.add_trace(go.Scatter(
                        x=_t_dates, y=_t_roll,
                        name="5-game avg",
                        line=dict(color=NORD["green"], width=2.5),
                        hovertemplate="%{x}<br>5-game avg: %{y:.1f}<extra></extra>",
                    ))
                    _fig_traj.add_hline(
                        y=8.0, line_dash="dash", line_color=NORD["bg3"],
                        annotation_text="League avg (8.0)",
                        annotation_position="bottom right",
                    )
                    _fig_traj.update_layout(
                        title=f"{_selected_label} | Season Game Score",
                        xaxis_title="Date",
                        yaxis_title="Game Score",
                        height=350,
                        margin=dict(l=60, r=20, t=50, b=50),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", y=1.1),
                    )
                    st.plotly_chart(_fig_traj, width="stretch")

                    # Rating trend on secondary chart
                    _fig_rat = go.Figure()
                    _fig_rat.add_trace(go.Scatter(
                        x=_t_dates, y=_t_rat,
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=5),
                        name="Elo rating",
                        hovertemplate="%{x}<br>Rating: %{y:.1f}<extra></extra>",
                    ))
                    _fig_rat.add_hline(y=1500, line_dash="dash", line_color=NORD["bg3"])
                    _fig_rat.update_layout(
                        title=f"{_selected_label} | Elo Rating Over Season",
                        xaxis_title="Date",
                        yaxis_title="Elo Rating",
                        height=260,
                        margin=dict(l=60, r=20, t=50, b=50),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(_fig_rat, width="stretch")

                    # Stat breakdown
                    _last5 = _traj[-5:]
                    _s1, _s2, _s3, _s4 = st.columns(4)
                    _s1.metric("Last 5 avg GmSc", f"{sum(t['game_score'] for t in _last5)/len(_last5):.1f}")
                    _s2.metric("Last 5 avg PTS",  f"{sum(t['pts'] for t in _last5)/len(_last5):.1f}")
                    _s3.metric("Last 5 avg REB",  f"{sum(t['reb'] for t in _last5)/len(_last5):.1f}")
                    _s4.metric("Last 5 avg AST",  f"{sum(t['ast'] for t in _last5)/len(_last5):.1f}")

        # ── Injury Impact ─────────────────────────────────────────────────
        with _p_tab_inj:
            st.caption(
                "**Elo Impact** — how many Elo points the team loses if this player misses the game. "
                "Formula: (player_rating - 1500) x minutes_share x position_weight. "
                "**Adj Team Elo** — the team's current Elo minus this player's contribution. "
                "Use this to reprice win probabilities when injury news breaks."
            )
            # Team selector — only teams with player data
            _teams_with_data = sorted(set(
                _p_engine.teams[pid]
                for pid in _p_engine.ratings
                if _p_engine.teams.get(pid)
            ))
            _team_names_map = {tid: engine.names.get(tid, tid) for tid in _teams_with_data}
            _team_options = {v: k for k, v in _team_names_map.items() if v}
            _sel_team_name = st.selectbox(
                "Select team", sorted(_team_options.keys()), key="p_inj_team"
            )
            _sel_team_id = _team_options.get(_sel_team_name, "")

            if _sel_team_id:
                _team_elo = engine.rating(_sel_team_id)
                _inj_report = _p_engine.team_injury_report(_sel_team_id, _team_elo, min_games=3)

                if not _inj_report:
                    st.info("Not enough player data for this team yet.")
                else:
                    _i1, _i2, _i3 = st.columns(3)
                    _i1.metric("Team Elo", f"{_team_elo:.1f}")
                    _top_impact = _inj_report[0] if _inj_report else {}
                    _i2.metric("Highest impact player", _top_impact.get("name", "-"))
                    _i3.metric("Their Elo impact", f"{_top_impact.get('elo_impact', 0):+.1f} pts")

                    st.markdown("---")
                    _inj_rows = [{
                        "Player":       r["name"],
                        "Pos":          r["position"],
                        "Rating":       r["rating"],
                        "Avg MIN":      r["avg_min"],
                        "GP":           r["games"],
                        "Elo Impact":   r["elo_impact"],
                        "Adj Team Elo": r["adj_team_elo"],
                    } for r in _inj_report]

                    def _color_impact(val):
                        if isinstance(val, (int, float)) and val > 20:
                            return f"color: {NORD['red']}; font-weight: bold"
                        if isinstance(val, (int, float)) and val > 10:
                            return f"color: {NORD['orange']}"
                        return ""

                    st.dataframe(
                        pd.DataFrame(_inj_rows).style.map(_color_impact, subset=["Elo Impact"]),
                        width="stretch",
                        hide_index=True,
                    )
                    st.caption(
                        "Position weights: C=1.1, F=1.0, G=0.95. "
                        "Minutes share = avg minutes / 200 (5 players x 40 min). "
                        "Only players with 3+ games shown."
                    )

        # ── Prop Edge ─────────────────────────────────────────────────────
        with _p_tab_prop:
            st.caption(
                "Model projects a player's stat distribution from recent games (normal approximation). "
                "**Edge** = P(exceeds line) - 52.38% breakeven at -110. "
                "Positive edge = model thinks the over has value. "
                "Use as a signal, not a guarantee — small samples have high variance."
            )
            _prop_all = _p_engine.top_players(n=500, min_games=3)
            _prop_options = {f"{name} ({_p_engine.teams.get(pid,'')})": pid
                             for pid, name, *_ in _prop_all}
            _prop_sel_label = st.selectbox("Select player", list(_prop_options.keys()), key="p_prop_sel")
            _prop_pid = _prop_options.get(_prop_sel_label, "")

            if _prop_pid:
                _pc1, _pc2, _pc3 = st.columns(3)
                with _pc1:
                    _prop_stat = st.selectbox("Stat", ["pts", "reb", "ast", "stl", "blk"], key="p_prop_stat")
                with _pc2:
                    _prop_line = st.number_input("Prop line", min_value=0.0, value=14.5, step=0.5, key="p_prop_line")
                with _pc3:
                    _prop_n = st.slider("Last N games", 3, 20, 10, key="p_prop_n")

                _edge_result = _p_engine.prop_edge(_prop_pid, _prop_stat, _prop_line, last_n=_prop_n)

                if "error" not in _edge_result:
                    _e1, _e2, _e3, _e4 = st.columns(4)
                    _e1.metric("Model projection", f"{_edge_result['mean']:.1f}")
                    _e2.metric("Std dev",          f"± {_edge_result['std_dev']:.1f}")
                    _e3.metric("P(over)",           f"{_edge_result['p_over']:.1%}")
                    _edge_val = _edge_result['edge']
                    _e4.metric("Edge vs -110",
                               f"{_edge_val:+.1%}",
                               delta=f"{'OVER value' if _edge_val > 0 else 'UNDER value'}",
                               delta_color="normal")

                    st.markdown("---")
                    st.caption(
                        f"Based on last {_edge_result['n_games']} games: "
                        f"{_edge_result['last_values']}. "
                        f"Normal approximation with continuity correction."
                    )

# ════════════════════════════════════════════════════════════════════════════ #
# TAB 9 — Sources
# ════════════════════════════════════════════════════════════════════════════ #

with tab_sources:
    st.subheader("Sources and Acknowledgements")
    st.markdown(
        "ChalkIQ is built on a stack of well-established academic results and open data. "
        "Below are the people and work this dashboard draws from directly."
    )
    st.markdown("---")

    # ── Mathematical foundations ───────────────────────────────────────────────
    st.markdown("### Mathematical Foundations")

    src_col1, src_col2 = st.columns(2)

    with src_col1:
        st.markdown("#### Elo Rating System")
        st.markdown(
            "**Arpad Elo** (1903–1992)\n\n"
            "Hungarian-American physics professor who invented the Elo rating system "
            "for chess in the 1960s. Originally published as:\n\n"
            "*The Rating of Chess Players, Past and Present* (1978), Arco Publishing.\n\n"
            "The core formula — expected score as a logistic function of rating difference, "
            "with iterative updates proportional to prediction error — is used verbatim "
            "in this project's ratings engine."
        )
        st.divider()

        st.markdown("#### Brier Score")
        st.markdown(
            "**Glenn W. Brier** (1950)\n\n"
            "\"Verification of Forecasts Expressed in Terms of Probability\"\n"
            "*Monthly Weather Review*, 78(1), 1–3.\n\n"
            "Introduced the mean squared error of probabilistic forecasts. "
            "Originally developed to evaluate weather forecasts; now a standard "
            "tool in any domain that produces probability estimates."
        )
        st.divider()

        st.markdown("#### In-Game Win Probability (Brownian Motion)")
        st.markdown(
            "**Hal S. Stern** (1994)\n\n"
            "\"A Brownian Motion Model for the Progress of Sports Scores\"\n"
            "*Journal of the American Statistical Association*, 89(427), 1128–1134.\n\n"
            "Modeled the score differential in a sports game as a Brownian motion "
            "process, giving a closed-form formula for win probability as a function "
            "of current lead and time remaining. The theoretical foundation for "
            "ChalkIQ's live win probability model."
        )

    with src_col2:
        st.markdown("#### Diffusion Constant for Basketball")
        st.markdown(
            "**Aaron Clauset, Martin Kogan, Sidney Redner** (2015)\n\n"
            "\"Safe Leads and Lead Changes in Competitive Team Sports\"\n"
            "*Physical Review E*, 91(6), 062815.\n\n"
            "Measured the empirical scoring diffusion constant for multiple sports "
            "using play-by-play data. Their college basketball estimates inform the "
            "$\\sigma_s = 2.0$ pts/√min parameter used in ChalkIQ's live model."
        )
        st.divider()

        st.markdown("#### Log Loss / Cross-Entropy")
        st.markdown(
            "**Claude Shannon** (1948)\n\n"
            "\"A Mathematical Theory of Communication\"\n"
            "*Bell System Technical Journal*, 27, 379–423.\n\n"
            "Log loss is the negative log-likelihood of a Bernoulli model, rooted "
            "in Shannon's information theory. As a **proper scoring rule**, it "
            "incentivizes honest probability estimates — a forecaster minimises "
            "expected log loss only by reporting their true beliefs."
        )
        st.divider()

        st.markdown("#### Logit-Probit Approximation")
        st.markdown(
            "**Standard result in statistics**\n\n"
            "The approximation $\\text{logit}(p) \\approx (\\pi/\\sqrt{3})\\,\\Phi^{-1}(p)$ "
            "is a classical result connecting the logistic and normal distributions. "
            "It is widely used in biostatistics and econometrics to convert between "
            "logistic regression outputs and probit/normal-distribution-based models. "
            "ChalkIQ uses it to translate the Elo logit prior into z-score space "
            "for the random-walk live model."
        )

    st.markdown("---")

    # ── Data sources ───────────────────────────────────────────────────────────
    st.markdown("### Data Sources")

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### ESPN Scoreboard API")
        st.markdown(
            "**ESPN / Disney** (unofficial public API)\n\n"
            "`site.api.espn.com/apis/site/v2/sports/basketball/`\n\n"
            "ChalkIQ fetches all regular-season game results and live game states "
            "from ESPN's publicly accessible scoreboard endpoint. "
            "This API is not officially documented or supported by ESPN; "
            "it is used here for educational and research purposes only."
        )
    with d2:
        st.markdown("#### NCAA Basketball")
        st.markdown(
            "**National Collegiate Athletic Association (NCAA)**\n\n"
            "All team names, game results, and tournament structure reflect the "
            "NCAA Division I Men's and Women's basketball seasons. "
            "Bracket seeding in this dashboard is projected from Elo ratings "
            "and does not reflect official Selection Committee decisions."
        )

    st.markdown("---")

    # ── Inspirations ───────────────────────────────────────────────────────────
    st.markdown("### Inspirations")

    i1, i2 = st.columns(2)
    with i1:
        st.markdown("#### KenPom")
        st.markdown(
            "**Ken Pomeroy** — kenpom.com\n\n"
            "The gold standard for college basketball analytics. Pomeroy's adjusted "
            "efficiency metrics and tempo-free statistics popularised the idea that "
            "rigorous quantitative models can outperform conventional basketball wisdom. "
            "A direct inspiration for applying Elo and probabilistic forecasting "
            "to the college game."
        )
    with i2:
        st.markdown("#### FiveThirtyEight")
        st.markdown(
            "**Nate Silver et al.** — FiveThirtyEight (2008–2023)\n\n"
            "FiveThirtyEight's Elo-based sports ratings — first for MLB, then NFL, "
            "NBA, and March Madness — demonstrated that Elo could be adapted beyond "
            "chess to produce well-calibrated, publicly explainable forecasts. "
            "Their open methodology articles directly shaped the design choices "
            "in ChalkIQ's ratings and simulation engines."
        )

    st.markdown("---")

    # ── Stack ──────────────────────────────────────────────────────────────────
    st.markdown("### Built With")
    st.markdown(
        "| Library | Authors | Used for |\n"
        "|---|---|---|\n"
        "| [Streamlit](https://streamlit.io) | Streamlit Inc. | Web app framework |\n"
        "| [Plotly](https://plotly.com) | Plotly Technologies | Interactive charts |\n"
        "| [Matplotlib](https://matplotlib.org) | Hunter et al. (2007) | Final Four figure |\n"
        "| [pandas](https://pandas.pydata.org) | Wes McKinney et al. | Data tables |\n"
        "| [requests](https://requests.readthedocs.io) | Kenneth Reitz | ESPN API calls |"
    )
    st.caption(
        "Hunter, J.D. (2007). Matplotlib: A 2D graphics environment. "
        "*Computing in Science and Engineering*, 9(3), 90–95."
    )


# ════════════════════════════════════════════════════════════════════════════ #
# CSV EXPORT UTILITIES
# ════════════════════════════════════════════════════════════════════════════ #

_SHEETS_ROOT = ROOT / "spreadsheets"


def _ensure_dirs():
    (_SHEETS_ROOT / "analysis").mkdir(parents=True, exist_ok=True)
    (_SHEETS_ROOT / "backtests").mkdir(parents=True, exist_ok=True)
    (_SHEETS_ROOT / "players").mkdir(parents=True, exist_ok=True)
    (_SHEETS_ROOT / "injuries").mkdir(parents=True, exist_ok=True)


def export_analysis_csvs(snapshots: list[dict], clv_recs: list[dict], alerts: list[dict]) -> list[str]:
    """Export odds/CLV/alert data to spreadsheets/analysis/. Returns list of paths written."""
    _ensure_dirs()
    written = []

    if snapshots:
        p = _SHEETS_ROOT / "analysis" / "odds_snapshots.csv"
        pd.DataFrame(snapshots).to_csv(p, index=False)
        written.append(str(p))

    if clv_recs:
        p = _SHEETS_ROOT / "analysis" / "clv_records.csv"
        pd.DataFrame(clv_recs).to_csv(p, index=False)
        written.append(str(p))

    lma = [a for a in alerts if a.get("type") == "line_move"]
    if lma:
        p = _SHEETS_ROOT / "analysis" / "line_movement.csv"
        pd.DataFrame(lma).to_csv(p, index=False)
        written.append(str(p))

    inj = [a for a in alerts if a.get("type") == "injury"]
    if inj:
        p = _SHEETS_ROOT / "injuries" / "injuries.csv"
        pd.DataFrame(inj).to_csv(p, index=False)
        written.append(str(p))

    return written


def export_backtest_csvs(
    bets_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    buckets_df: pd.DataFrame,
    label: str,
) -> list[str]:
    """Export backtest results to spreadsheets/backtests/. Returns list of paths written."""
    _ensure_dirs()
    written = []
    safe = label.replace(" ", "_").replace("/", "-")

    for df, suffix in [(bets_df, "bets"), (monthly_df, "monthly"), (buckets_df, "edge_buckets")]:
        if not df.empty:
            p = _SHEETS_ROOT / "backtests" / f"{safe}_{suffix}.csv"
            df.to_csv(p, index=False)
            written.append(str(p))

    return written


# ════════════════════════════════════════════════════════════════════════════ #
# TAB — NCAA 361 Exchange
# ════════════════════════════════════════════════════════════════════════════ #

with tab_ncaa361:
    from src.utils.rating_history import build_rating_histories, market_movers, ncaa361_spread

    st.subheader("NCAA 361 Exchange")
    st.caption(
        "Elo rating charted as a stock price. "
        "Every win is a price increase, every loss a drop. "
        "Upsets move the line more than expected results. "
        "All 361 D-I teams tracked from opening day."
    )
    st.markdown("---")

    @st.cache_data(show_spinner=False)
    def _load_histories(division: str):
        _eng = load_engine(division)
        _hists = build_rating_histories(_eng.history)
        _movers_g, _movers_l = market_movers(_hists, _eng.names, window_days=30)
        _spread = ncaa361_spread(_hists)
        return _hists, _eng.names, _movers_g, _movers_l, _spread

    _hists, _enames, _gainers, _losers, _spread = _load_histories(division)

    # ── Market summary metrics ────────────────────────────────────────────────
    _all_current = {tid: pts[-1][1] for tid, pts in _hists.items() if pts}
    # Apply injury adjustments to current ratings (same as adj_engine)
    _all_current_adj = {
        tid: elo + _inj_overrides.get(tid, 0.0)
        for tid, elo in _all_current.items()
    }
    _avg_elo  = sum(_all_current_adj.values()) / len(_all_current_adj) if _all_current_adj else 1500
    _spread_now = _spread[-1][1] if _spread else 0
    _top_team_id = max(_all_current_adj, key=_all_current_adj.get) if _all_current_adj else None
    _top_team_nm = _enames.get(_top_team_id, "") if _top_team_id else ""
    _top_team_el = _all_current_adj.get(_top_team_id, 0) if _top_team_id else 0

    _mc1, _mc2, _mc3, _mc4 = st.columns(4)
    _mc1.metric("Teams tracked", f"{len(_all_current_adj):,}")
    _mc2.metric("Index avg (Elo)", f"{_avg_elo:.0f}",
                help="Injury-adjusted" if _inj_overrides else None)
    _mc3.metric("Rating spread (stdev)", f"{_spread_now:.1f}",
                help="Higher spread = more separation between elite and bottom teams.")
    _top_inj_delta = _inj_overrides.get(_top_team_id, 0.0) if _top_team_id else 0.0
    _mc4.metric("Top rated", f"{_top_team_nm}",
                delta=f"{_top_team_el:.0f} Elo" + (" ⚠inj adj" if _top_inj_delta else ""))

    st.markdown("---")

    # ── Team chart ────────────────────────────────────────────────────────────
    _all_team_names = sorted(_enames.values())
    _name_to_id = {v: k for k, v in _enames.items()}

    _ch_col, _ctrl_col = st.columns([3, 1])
    with _ctrl_col:
        _selected_name = st.selectbox(
            "Select team", _all_team_names,
            index=_all_team_names.index("Duke Blue Devils") if "Duke Blue Devils" in _all_team_names else 0,
            key="ncaa361_team",
        )
        _normalize = st.toggle("Normalize to 100", value=False, key="ncaa361_norm",
                               help="Index all teams to 100 at season start for % comparison.")
        _compare_names = st.multiselect(
            "Compare with (up to 4)", [n for n in _all_team_names if n != _selected_name],
            max_selections=4,
            key="ncaa361_compare",
        )

    _selected_id = _name_to_id.get(_selected_name)

    def _team_series(tid: str):
        pts = _hists.get(tid, [])
        if not pts:
            return [], []
        dates = [p[0] for p in pts]
        ratings = [p[1] for p in pts]
        if _normalize and ratings:
            base = ratings[0]
            ratings = [r / base * 100 for r in ratings]
        return dates, ratings

    with _ch_col:
        if _selected_id:
            _fig_price = go.Figure()

            # Main team
            _dx, _dy = _team_series(_selected_id)
            _line_color = NORD["green"] if (len(_dy) > 1 and _dy[-1] >= _dy[0]) else NORD["red"]
            _fig_price.add_trace(go.Scatter(
                x=_dx, y=_dy,
                mode="lines",
                name=_selected_name,
                line=dict(color=_line_color, width=2.5),
                hovertemplate="%{x}<br>Elo: %{y:.1f}<extra>" + _selected_name + "</extra>",
            ))

            # Comparison teams
            _comp_colors = [NORD["frost1"], NORD["yellow"], NORD["purple"], NORD["orange"]]
            for _ci, _cn in enumerate(_compare_names):
                _cid = _name_to_id.get(_cn)
                if _cid:
                    _cx, _cy = _team_series(_cid)
                    _fig_price.add_trace(go.Scatter(
                        x=_cx, y=_cy,
                        mode="lines",
                        name=_cn,
                        line=dict(color=_comp_colors[_ci % len(_comp_colors)], width=1.5, dash="dot"),
                        hovertemplate="%{x}<br>Elo: %{y:.1f}<extra>" + _cn + "</extra>",
                    ))

            _y_label = "Indexed (base 100)" if _normalize else "Elo Rating"
            _fig_price.add_hline(
                y=100 if _normalize else 1500,
                line_color=NORD["bg3"], line_dash="dash", line_width=1,
            )
            _fig_price.update_layout(
                xaxis_title="Date",
                yaxis_title=_y_label,
                height=380,
                margin=dict(l=60, r=20, t=20, b=50),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified",
            )
            st.plotly_chart(_fig_price, width="stretch")

            # Stats row for selected team
            _s_pts = _hists.get(_selected_id, [])
            if len(_s_pts) > 1:
                _s_start = _s_pts[0][1]
                _s_end   = _s_pts[-1][1]
                _s_chg   = _s_end - _s_start
                _s_peak  = max(p[1] for p in _s_pts)
                _s_trough = min(p[1] for p in _s_pts)
                _s1, _s2, _s3, _s4 = st.columns(4)
                _s1.metric("Season open",  f"{_s_start:.0f}")
                _s2.metric("Current",      f"{_s_end:.0f}", delta=f"{_s_chg:+.1f}")
                _s3.metric("Season high",  f"{_s_peak:.0f}")
                _s4.metric("Season low",   f"{_s_trough:.0f}")

    st.markdown("---")

    # ── Rating spread (volatility index) ─────────────────────────────────────
    if _spread:
        with st.expander("NCAA 361 Spread Index (field separation over time)"):
            st.caption(
                "Standard deviation of all team Elo ratings over the season. "
                "Rises as elite teams pull away from the bottom. "
                "Analogous to a market volatility index."
            )
            _sp_dates = [s[0] for s in _spread]
            _sp_vals  = [s[1] for s in _spread]
            _fig_spread = go.Figure(go.Scatter(
                x=_sp_dates, y=_sp_vals,
                mode="lines",
                fill="tozeroy",
                line=dict(color=NORD["frost1"], width=2),
                fillcolor="rgba(136,192,208,0.15)",
                hovertemplate="%{x}<br>Spread: %{y:.1f}<extra></extra>",
            ))
            _fig_spread.update_layout(
                xaxis_title="Date",
                yaxis_title="Rating std dev",
                height=260,
                margin=dict(l=60, r=20, t=10, b=50),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(_fig_spread, width="stretch")

    # ── Market movers ─────────────────────────────────────────────────────────
    st.markdown("### Market Movers (last 30 days)")
    _mv1, _mv2 = st.columns(2)

    with _mv1:
        st.markdown(f"**Top Gainers**")
        _g_rows = [{"Team": r["name"], "Current": r["current"], "+/- Elo": f"+{r['change']:.1f}"} for r in _gainers]
        _df_g = pd.DataFrame(_g_rows)
        st.dataframe(_df_g.style.map(lambda _: f"color: {NORD['green']}", subset=["+/- Elo"]),
                     hide_index=True, width="stretch")

    with _mv2:
        st.markdown(f"**Top Losers**")
        _l_rows = [{"Team": r["name"], "Current": r["current"], "+/- Elo": f"{r['change']:.1f}"} for r in _losers]
        _df_l = pd.DataFrame(_l_rows)
        st.dataframe(_df_l.style.map(lambda _: f"color: {NORD['red']}", subset=["+/- Elo"]),
                     hide_index=True, width="stretch")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB — Signals
# ════════════════════════════════════════════════════════════════════════════ #

with tab_signals:
    st.subheader("Signals | Odds, Line Movement, CLV, Injuries")
    st.caption(
        "Data is written by `scripts/poll_odds.py` and `scripts/poll_injuries.py`. "
        "Run those scripts to populate this tab."
    )

    # Load all data upfront
    _odds_dir = ROOT / "data" / "odds"
    try:
        from src.odds.store import load_snapshots, load_clv_records, load_alerts
        _snapshots = load_snapshots(_odds_dir)
        _live_clv  = load_clv_records(_odds_dir)
        _alerts    = load_alerts(_odds_dir)
        for _r in _live_clv:
            _r.setdefault("source", "live")
        _clv_recs  = _live_clv
    except Exception as _e:
        st.warning(f"Could not load odds data: {_e}")
        _snapshots, _clv_recs, _alerts = [], [], []

    # Historical CLV — loaded separately so it never breaks live signals
    try:
        from src.odds.store import load_historical_clv_records as _load_hist
        _hist_clv = _load_hist(_odds_dir)
        for _r in _hist_clv:
            _r.setdefault("source", "historical")
        _clv_recs = _hist_clv + _clv_recs
    except Exception:
        pass  # historical file or function missing — silently skip

    _lma_alerts = [a for a in _alerts if a.get("type") == "line_move"]
    _inj_alerts = [a for a in _alerts if a.get("type") == "injury"]

    # Also load from the signals store (written by poll_injuries.py)
    # and merge so the tab is populated regardless of which path wrote them.
    try:
        from src.signals.injuries import load_alerts as _load_sig_inj
        _sig_inj_raw = _load_sig_inj()
        _sig_inj_ids = {
            (a.get("player_id",""), a.get("detected_at",""))
            for a in _inj_alerts
        }
        for _sa in _sig_inj_raw:
            _key = (_sa.get("player_id",""), _sa.get("detected_at",""))
            if _key not in _sig_inj_ids:
                _inj_alerts.append({
                    **_sa,
                    "type":    "injury",
                    "player":  _sa.get("player_name", ""),
                    "team":    _sa.get("team_id", ""),
                    "status":  _sa.get("new_status", ""),
                    "elo_impact": _sa.get("elo_impact", 0),
                })
    except Exception:
        pass

    # Export button (top level)
    if st.button("Export all Signals to CSV", key="export_signals"):
        _paths = export_analysis_csvs(_snapshots, _clv_recs, _alerts)
        if _paths:
            st.success(f"Exported {len(_paths)} files to spreadsheets/analysis/")
            for _p in _paths:
                st.caption(_p)
        else:
            st.info("No data to export yet.")

    st.markdown("---")

    sig_tab_odds, sig_tab_lma, sig_tab_clv, sig_tab_inj = st.tabs([
        f"Odds / Edge ({len(set((s.get('game_id','') for s in _snapshots))):,} games)",
        f"Line Movement ({len(_lma_alerts):,} moves)",
        f"CLV ({len(_clv_recs):,} records)",
        f"Injuries ({len(_inj_alerts):,} alerts)",
    ])

    # ── Odds / Edge ──────────────────────────────────────────────────────────
    with sig_tab_odds:
        st.caption(
            "**Home ML / Away ML** — current moneyline from the book. "
            "+150 means bet $100 to win $150. -110 means bet $110 to win $100. "
            "| **Line Prob** — the book's implied win probability for the home team (vig removed). "
            "| **Model Prob** — our Elo model's win probability for the home team. "
            "| **Edge (home)** — Model Prob minus Line Prob. Positive = our model likes the home team "
            "more than the book does. Negative = our model likes the away team. "
            "| **Signal** — flagged EDGE HOME or EDGE AWAY when the gap is 5%+ (meaningful disagreement)."
        )
        if not _snapshots:
            st.info("No odds data. Run: `python scripts/poll_odds.py --once`")
        else:
            # Latest snapshot per game+bookmaker
            _latest: dict[tuple, dict] = {}
            for _s in _snapshots:
                _k = (_s.get("game_id"), _s.get("bookmaker"))
                _latest[_k] = _s

            _rows = []
            for _s in _latest.values():
                _home_espn = _s.get("home_espn_id", "")
                _away_espn = _s.get("away_espn_id", "")
                # Recompute model prob live using injury-adjusted engine when IDs available
                if _home_espn and _away_espn and _inj_overrides:
                    _adj_prob = adj_engine.win_prob(_home_espn, _away_espn, neutral=False)
                else:
                    _adj_prob = _s.get("model_prob_home", 0.5)
                _gap = (_adj_prob - _s.get("home_prob", 0.5))
                _inj_flag = "⚠" if (
                    _inj_overrides and (
                        _home_espn in _inj_overrides or _away_espn in _inj_overrides
                    )
                ) else ""
                _rows.append({
                    "Matchup":       f"{_s.get('away_team','?')} @ {_s.get('home_team','?')}",
                    "Book":          _s.get("bookmaker", ""),
                    "Home ML":       _fmt_ml(int(_s["home_ml"])) if _s.get("home_ml") else "-",
                    "Away ML":       _fmt_ml(int(_s["away_ml"])) if _s.get("away_ml") else "-",
                    "Line Prob":     f"{_s.get('home_prob', 0):.1%}",
                    "Model Prob":    f"{_adj_prob:.1%}{_inj_flag}",
                    "Edge (home)":   f"{_gap:+.1%}",
                    "Signal":        "EDGE HOME" if _gap >= 0.05 else ("EDGE AWAY" if _gap <= -0.05 else "-"),
                    "Fetched":       str(_s.get("fetched_at", _s.get("timestamp", "")))[:16],
                })

            _df_odds = pd.DataFrame(_rows).sort_values("Signal", ascending=False)

            def _color_signal(val):
                if "EDGE" in str(val):
                    return f"color: {NORD['green']}; font-weight: bold"
                return ""

            st.dataframe(
                _df_odds.style.map(_color_signal, subset=["Signal"]),
                width="stretch",
                hide_index=True,
            )

    # ── Line Movement ─────────────────────────────────────────────────────────
    with sig_tab_lma:
        st.caption(
            "**From ML / To ML** — the moneyline before and after the move was detected. "
            "| **Move** — change in implied probability. Positive = line moved toward home team. "
            "Negative = line moved toward away team. "
            "| **Sharp** — YES if the move was 4%+ in implied probability. "
            "A move that large is unlikely to be casual public money — it suggests a professional "
            "bettor (sharp) forced the book to reprice. "
            "| **Sharp + Edge** — YES (green) when the sharp move direction matches our model's edge signal. "
            "This is the highest-confidence signal: sharps and our model independently agree on the same side."
        )
        if not _lma_alerts:
            st.info("No line movement detected yet. Populate by running poll_odds.py over multiple polls.")
        else:
            # Build latest snapshot lookup: (game_id, bookmaker) -> model edge
            _snap_lookup: dict[tuple, float] = {}
            for _s in _snapshots:
                _sk = (_s.get("game_id"), _s.get("bookmaker"))
                _h_id = _s.get("home_espn_id", "")
                _a_id = _s.get("away_espn_id", "")
                if _h_id and _a_id and _inj_overrides:
                    _mp = adj_engine.win_prob(_h_id, _a_id, neutral=False)
                else:
                    _mp = _s.get("model_prob_home", 0.5)
                _snap_lookup[_sk] = _mp - _s.get("home_prob", 0.5)

            _lma_rows = []
            for _a in reversed(_lma_alerts):
                _move = _a.get("move_size", 0)
                _sharp = _a.get("sharp", False)
                # Sharp direction: positive move = toward home, negative = toward away
                _sharp_dir = "HOME" if _move > 0 else "AWAY"
                # Our edge direction from latest snapshot
                _edge_gap = _snap_lookup.get((_a.get("game_id"), _a.get("bookmaker")))
                if _edge_gap is not None and abs(_edge_gap) >= 0.05:
                    _edge_dir = "HOME" if _edge_gap > 0 else "AWAY"
                    _agrees = _sharp and (_sharp_dir == _edge_dir)
                    _agree_str = "YES" if _agrees else ("no" if _sharp else "-")
                else:
                    _agree_str = "-"  # no edge signal on this game

                _lma_rows.append({
                    "Matchup":       f"{_a.get('away_team','?')} @ {_a.get('home_team','?')}",
                    "Book":          _a.get("bookmaker", ""),
                    "From ML":       _fmt_ml(int(_a["from_ml"])) if _a.get("from_ml") else "-",
                    "To ML":        _fmt_ml(int(_a["to_ml"])) if _a.get("to_ml") else "-",
                    "Move":          f"{_move:+.1%}",
                    "Sharp":         "YES" if _sharp else "no",
                    "Sharp + Edge":  _agree_str,
                    "Detected":      str(_a.get("detected_at", ""))[:16],
                })
            _df_lma = pd.DataFrame(_lma_rows)

            def _color_sharp(val):
                if val == "YES":
                    return f"color: {NORD['yellow']}; font-weight: bold"
                return ""

            def _color_agree(val):
                if val == "YES":
                    return f"color: {NORD['green']}; font-weight: bold"
                return ""

            st.dataframe(
                _df_lma.style
                    .map(_color_sharp, subset=["Sharp"])
                    .map(_color_agree, subset=["Sharp + Edge"]),
                width="stretch",
                hide_index=True,
            )

    # ── CLV ───────────────────────────────────────────────────────────────────
    with sig_tab_clv:
        st.caption(
            "**CLV (Closing Line Value)** — measures whether our model was smarter than the market. "
            "| **Open ML / Close ML** — the line when we first saw it vs. when the game started. "
            "| **CLV vs Open** — our model prob minus the opening implied prob. "
            "| **CLV vs Close** — our model prob minus the closing implied prob. "
            "Closing line is the most important: the market is sharpest right before tip-off. "
            "Positive CLV vs close means our model had an edge over the final market price. "
            "| **Beat Close** — YES if CLV vs close is positive. "
            "Consistently beating the closing line is the strongest evidence of real model edge."
        )
        if not _clv_recs:
            st.info("No CLV records yet. CLV is computed when a game closes (poll_odds.py checks completed games).")
        else:
            _clv_power_only = st.toggle(
                "Power conferences only (ACC / Big Ten / Big 12 / SEC / Big East)",
                value=False,
                key="clv_power_filter",
                help="Filters to matchups where at least one team is a power conference program. "
                     "Elo calibration is better for high-volume programs; "
                     "small-conference CLV gaps are often noise.",
            )

            _clv_filtered = _clv_recs
            if _clv_power_only:
                _clv_filtered = [
                    _r for _r in _clv_recs
                    if _r.get("home_team") in POWER_CONF_TEAMS
                    or _r.get("away_team") in POWER_CONF_TEAMS
                ]

            _clv_rows = []
            for _r in reversed(_clv_filtered):
                _beat = (_r.get("clv_vs_closing", 0) or 0) > 0
                _clv_rows.append({
                    "Matchup":       f"{_r.get('away_team','?')} @ {_r.get('home_team','?')}",
                    "Date":          _r.get("game_date", _r.get("recorded_at", ""))[:10],
                    "Book":          _r.get("bookmaker", ""),
                    "Source":        _r.get("source", "live"),
                    "Model Prob":    f"{_r.get('model_prob_home', 0):.1%}",
                    "Open ML":       _fmt_ml(int(_r["opening_home_ml"])) if _r.get("opening_home_ml") else "-",
                    "Close ML":      _fmt_ml(int(_r["closing_home_ml"])) if _r.get("closing_home_ml") else "-",
                    "CLV vs Open":   f"{(_r.get('clv_vs_opening') or 0):+.2%}",
                    "CLV vs Close":  f"{(_r.get('clv_vs_closing') or 0):+.2%}",
                    "Beat Close":    "YES" if _beat else "no",
                    "Home Won":      "W" if _r.get("home_won") else ("L" if _r.get("home_won") is False else "?"),
                })
            _df_clv = pd.DataFrame(_clv_rows)

            # Summary metrics for filtered set
            _n_clv   = len(_clv_filtered)
            _beat_n  = sum(1 for _r in _clv_filtered if (_r.get("clv_vs_closing") or 0) > 0)
            _avg_clv = sum(_r.get("clv_vs_closing") or 0 for _r in _clv_filtered) / _n_clv if _n_clv else 0
            _n_hist  = sum(1 for _r in _clv_filtered if _r.get("source") == "historical")
            _n_live  = _n_clv - _n_hist
            _n_total = len(_clv_recs)

            _c1, _c2, _c3, _c4 = st.columns(4)
            _c1.metric("Games shown", f"{_n_clv} / {_n_total}",
                       help=f"{_n_hist} historical  |  {_n_live} live")
            _c2.metric("Beat closing line", f"{_beat_n}/{_n_clv} ({_beat_n/_n_clv:.0%})" if _n_clv else "-")
            _c3.metric("Avg CLV vs close", f"{_avg_clv:+.2%}")
            _c4.metric("Target", "> 55% beat rate")
            st.markdown("---")

            def _color_beat(val):
                if val == "YES":
                    return f"color: {NORD['green']}; font-weight: bold"
                return ""

            if _clv_rows:
                st.dataframe(
                    _df_clv.style.map(_color_beat, subset=["Beat Close"]),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No records match the current filter.")

    # ── Injuries ──────────────────────────────────────────────────────────────
    with sig_tab_inj:
        st.caption(
            "**Status** — the injury designation from ESPN (Out, Questionable, Doubtful, etc.). "
            "| **Est. Impact** — estimated Elo point penalty to the team if this player misses the game. "
            "Calculated from position and team strength: a star PG on a top-10 team "
            "has more impact than a backup on a mid-major. "
            "Use this to spot games where the book may not have fully priced in an injury yet — "
            "that's where the edge window is."
        )
        if not _inj_alerts:
            st.info("No injury alerts. Run: `python scripts/poll_injuries.py --once`")
        else:
            _inj_rows = []
            for _a in reversed(_inj_alerts):
                _inj_rows.append({
                    "Player":    _a.get("player", "?"),
                    "Team":      _a.get("team", "?"),
                    "Status":    _a.get("status", "?"),
                    "Position":  _a.get("position", "?"),
                    "Est. Impact": f"{_a.get('elo_impact', 0):+.0f} Elo pts",
                    "Detected":  str(_a.get("detected_at", ""))[:16],
                })
            st.dataframe(pd.DataFrame(_inj_rows), width="stretch", hide_index=True)


# ════════════════════════════════════════════════════════════════════════════ #
# TAB — Backtest
# ════════════════════════════════════════════════════════════════════════════ #

with tab_backtest:
    st.subheader("Backtest | Historical Model Performance")
    st.markdown(
        "Walk-forward simulation: ratings build from scratch each season, "
        "then flat-stake bets at -110 juice wherever the model has edge. "
        "Breakeven win rate at -110 is **52.4%**."
    )

    st.info(
        "**How to read P&L:** Each bet is a flat **$100** stake. "
        "A win at -110 returns +$90.91. A loss is -$100.\n\n"
        "**Accuracy disclaimer:** This backtest assumes every game is available at exactly "
        "-110 on both sides, which is not realistic. Real book lines move — a game our model "
        "likes at 60% might have already closed at -150 (63%), making it a losing bet. "
        "The backtest does not compare against actual historical closing lines because we "
        "don't have them. It answers: *does the model find signal above random?* "
        "It does **not** answer: *would this make money on a real book?* "
        "The CLV records in the Signals tab (built from live poll_odds.py data) are a "
        "forward-test against real closing lines and are more meaningful for that question."
    )
    st.markdown("---")

    _bt_col1, _bt_col2, _bt_col3, _bt_col4, _bt_col5 = st.columns([2, 2, 2, 2, 2])
    with _bt_col1:
        _bt_div = st.selectbox("Division", ["mens", "womens"], key="bt_div")
    with _bt_col2:
        _bt_seasons = st.multiselect(
            "Seasons", [2023, 2024, 2025, 2026],
            default=[2026],
            key="bt_seasons",
        )
    with _bt_col3:
        _bt_edge = st.slider(
            "Min edge (%)", 1, 20, 3, key="bt_edge",
            help="Only bet when the model's win probability exceeds the 52.4% breakeven by at least this much. "
                 "3% = model must say 55.4%+ to bet. Higher = fewer bets, theoretically higher quality.",
        )
    with _bt_col4:
        _bt_warmup = st.number_input(
            "Warmup games", 10, 300, 50, step=10, key="bt_warmup",
            help="Skip betting for the first N games of the season. "
                 "Early in the season all teams start at 1500 Elo so ratings are unreliable — "
                 "the model needs games to separate good teams from bad ones before its "
                 "predictions are trustworthy enough to bet on.",
        )
    with _bt_col5:
        _bt_decay = st.slider(
            "Decay half-life (days)", 0, 365, 0, step=15, key="bt_decay",
            help="Recency decay: games N days old have K scaled by 0.5^(N/half_life). "
                 "0 = no decay (all games weighted equally). "
                 "90 = games 90 days ago have half the K factor of today's games. "
                 "Shorter half-life = ratings reflect recent form more than season-long history.",
        )

    _bt_use_injuries = st.toggle(
        "Apply injury adjustments to backtest",
        value=False,
        key="bt_injuries",
        help="Uses timestamped injury alerts from data/signals/injury_alerts.jsonl "
             "to apply Elo penalties on dates when players were known to be out. "
             "Only affects games in the current season (2025-26) where alerts have been collected.",
    )
    _run_bt = st.button("Run Backtest", type="primary", key="run_bt")

    _SEASON_RANGES = {
        2023: (date(2022, 11, 7),  date(2023, 4, 3)),
        2024: (date(2023, 11, 6),  date(2024, 4, 8)),
        2025: (date(2024, 11, 4),  date(2025, 4, 7)),
        2026: (date(2025, 11, 4),  date.today()),
    }

    # ── Injury timeline for backtest (current season only) ──────────────────
    def _build_bt_injury_timeline(bt_games: list[dict]) -> dict:
        """
        Build {game_date: {team_id: elo_delta}} for use in injury-adjusted backtesting.

        Primary source: boxscore DNPs — players who averaged 15+ min over the prior
        5 games but are absent from a boxscore. Covers all seasons with cached boxscores.

        Fallback: timestamped injury alerts from data/signals/injury_alerts.jsonl.
        Only covers dates since polling started (current season only).

        The two sources are merged; boxscore DNPs take precedence since they are
        observed facts rather than status-change estimates.
        """
        from src.players.engine import build_dnp_timeline, load_boxscores

        # ── Primary: DNP timeline from boxscores ──────────────────────────────
        _box_dir = ROOT / "data" / "raw" / division / "boxscores"
        _boxscores = load_boxscores(str(_box_dir))
        timeline: dict = {}
        if _boxscores:
            timeline = build_dnp_timeline(
                _boxscores,
                team_elo_fn=engine.rating,
                min_avg_min=15.0,
                min_prior_games=5,
            )

        # ── Fallback: alert-based timeline for dates not covered by boxscores ─
        from src.signals.injuries import load_alerts as _load_inj, estimate_impact, _OUT_STATUSES
        _raw = _load_inj()
        if _raw:
            from datetime import date as _d, timedelta as _td
            _player_events: dict = {}
            for _a in _raw:
                _pid = _a.get("player_id", "")
                if not _pid:
                    continue
                _dt = _a.get("detected_at", "")[:10]
                _tid = _a.get("team_id", "")
                _status = _a.get("new_status", "")
                _imp = estimate_impact(
                    player_name=_a.get("player_name", ""),
                    position=_a.get("position", ""),
                    team_elo=engine.rating(_tid),
                    status=_status,
                )
                _player_events.setdefault(_pid, []).append((_dt, _status, _tid, _imp))

            today_str = str(_d.today())
            for _pid, _events in _player_events.items():
                _events.sort(key=lambda x: x[0])
                for _i, (_dt, _status, _tid, _imp) in enumerate(_events):
                    if _status not in _OUT_STATUSES:
                        continue
                    _end = _events[_i + 1][0] if _i + 1 < len(_events) else today_str
                    _cur = _d.fromisoformat(_dt)
                    _end_d = _d.fromisoformat(_end)
                    while _cur <= _end_d:
                        _ds = str(_cur)
                        # Only add if boxscore DNPs didn't already cover this date+team
                        if _ds not in timeline or _tid not in timeline[_ds]:
                            if _ds not in timeline:
                                timeline[_ds] = {}
                            timeline[_ds][_tid] = timeline[_ds].get(_tid, 0.0) + _imp
                        _cur += _td(days=1)

        return timeline

    @st.cache_data(show_spinner="Running backtest…")
    def _run_backtest(division: str, seasons: tuple[int, ...], min_edge: float, warmup: int, decay: float, use_injuries: bool = False):
        from src.backtest.engine import Backtester
        from src.utils.data import fetch_season

        _cache_dirs = {
            "mens":   {2023: "data/raw/mens/2023", 2024: "data/raw/mens/2024",
                       2025: "data/raw/mens/2025", 2026: "data/raw/mens"},
            "womens": {2023: "data/raw/womens/2023", 2024: "data/raw/womens/2024",
                       2025: "data/raw/womens/2025", 2026: "data/raw/womens"},
        }

        all_games: list[dict] = []
        for _s in sorted(seasons):
            _start, _end = _SEASON_RANGES[_s]
            _cache = _cache_dirs.get(division, _cache_dirs["mens"]).get(_s, f"data/raw/{division}/{_s}")
            _games = fetch_season(_start, _end, cache_dir=_cache, division=division, verbose=False)
            all_games.extend(_games)

        # injury_timeline is computed outside the cache boundary and passed in
        # via use_injuries flag (busts cache when toggled)
        bt = Backtester(k=24.0, home_advantage=100.0, decay_half_life=float(decay))
        return bt.run(all_games, min_edge=min_edge, stake=100.0, warmup_games=warmup)

    @st.cache_data(show_spinner="Loading game data for backtest…")
    def _fetch_bt_games(division: str, seasons: tuple[int, ...]) -> list[dict]:
        """Load and cache game data for all selected seasons. Shared by both backtest paths."""
        from src.utils.data import fetch_season as _fs
        _cache_dirs = {
            "mens":   {2023: "data/raw/mens/2023", 2024: "data/raw/mens/2024",
                       2025: "data/raw/mens/2025", 2026: "data/raw/mens"},
            "womens": {2023: "data/raw/womens/2023", 2024: "data/raw/womens/2024",
                       2025: "data/raw/womens/2025", 2026: "data/raw/womens"},
        }
        _games: list[dict] = []
        for _s in sorted(seasons):
            _st, _en = _SEASON_RANGES[_s]
            _cd = _cache_dirs.get(division, _cache_dirs["mens"]).get(_s, f"data/raw/{division}/{_s}")
            _games.extend(_fs(_st, _en, cache_dir=_cd, division=division, verbose=False))
        return _games

    if _run_bt and _bt_seasons:
        with st.spinner("Running backtest…"):
            if _bt_use_injuries:
                _bt_games = _fetch_bt_games(_bt_div, tuple(sorted(_bt_seasons)))
                _timeline = _build_bt_injury_timeline(_bt_games)
                from src.backtest.engine import Backtester as _BTAdj
                _bt_adj = _BTAdj(k=24.0, home_advantage=100.0, decay_half_life=float(_bt_decay))
                _results = _bt_adj.run(
                    _bt_games, min_edge=_bt_edge / 100, stake=100.0,
                    warmup_games=_bt_warmup, injury_timeline=_timeline or None,
                )
                if _timeline:
                    _n_dnp_dates = len(_timeline)
                    _box_count_bt = len(list((ROOT / "data" / "raw" / _bt_div / "boxscores").glob("*.json")))
                    st.caption(
                        f"Injury adjustments applied across {_n_dnp_dates} game dates "
                        f"({_box_count_bt:,} boxscores used for DNP detection). "
                        "Fetch historical boxscores to extend coverage to prior seasons: "
                        "`python scripts/fetch_boxscores.py --seasons 2023 2024 2025`"
                    )
                else:
                    st.warning(
                        "No injury data found. Fetch boxscores to enable DNP-based adjustments: "
                        "`python scripts/fetch_boxscores.py --seasons 2023 2024 2025 2026`"
                    )
            else:
                _results = _run_backtest(
                    _bt_div,
                    tuple(sorted(_bt_seasons)),
                    _bt_edge / 100,
                    _bt_warmup,
                    _bt_decay,
                )

        if not _results.bets_placed:
            st.warning("No bets placed — try lowering the min edge or adding more seasons.")
        else:
            # ── Summary metrics ───────────────────────────────────────────────
            _m1, _m2, _m3, _m4, _m5, _m6 = st.columns(6)
            _m1.metric("Bets placed",  f"{_results.bets_placed:,}")
            _m2.metric("Win rate",     f"{_results.win_rate:.1%}",
                       delta=f"{_results.win_rate - 0.524:+.1%} vs breakeven",
                       delta_color="normal")
            _m3.metric("Total P&L",    f"${_results.total_pnl:,.0f}")
            _m4.metric("ROI",          f"{_results.roi:+.1%}")
            _m5.metric("Avg edge",     f"{_results.avg_edge:+.1%}")
            _m6.metric("Log loss",     f"{_results.log_loss:.4f}")

            st.markdown("---")

            # ── Cumulative P&L chart ──────────────────────────────────────────
            from src.backtest.engine import Backtester as _BT
            _curve = _BT.cumulative_pnl(_results)
            if _curve:
                _dates  = [c[0] for c in _curve]
                _cumPnl = [c[1] for c in _curve]
                _color_line = NORD["green"] if _cumPnl[-1] >= 0 else NORD["red"]
                _fill_rgba = "rgba(163,190,140,0.15)" if _cumPnl[-1] >= 0 else "rgba(191,97,106,0.15)"

                _fig_bt = go.Figure()
                _fig_bt.add_trace(go.Scatter(
                    x=_dates, y=_cumPnl,
                    mode="lines",
                    fill="tozeroy",
                    line=dict(color=_color_line, width=2),
                    fillcolor=_fill_rgba,
                    name="Cumulative P&L",
                    hovertemplate="%{x}<br>P&L: $%{y:,.0f}<extra></extra>",
                ))
                _fig_bt.add_hline(y=0, line_color=NORD["bg3"], line_dash="dash")
                _fig_bt.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Cumulative P&L ($)",
                    height=320,
                    margin=dict(l=60, r=20, t=20, b=50),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(_fig_bt, width="stretch")

            st.markdown("---")

            _bt_tab_month, _bt_tab_buckets, _bt_tab_bets, _bt_tab_clv = st.tabs([
                "Monthly Breakdown", "Edge Buckets", "All Bets", "vs Closing Line"
            ])

            # ── Monthly breakdown ─────────────────────────────────────────────
            with _bt_tab_month:
                _monthly_rows = []
                for _mo, _md in sorted(_results.monthly.items()):
                    _wr  = _md["wins"] / _md["bets"] if _md["bets"] else 0
                    _roi = _md["pnl"] / (100 * _md["bets"]) if _md["bets"] else 0
                    _monthly_rows.append({
                        "Month":  _mo,
                        "Bets":   _md["bets"],
                        "Wins":   _md["wins"],
                        "Win%":   f"{_wr:.1%}",
                        "P&L":    f"${_md['pnl']:+,.0f}",
                        "ROI":    f"{_roi:+.1%}",
                    })
                _df_monthly = pd.DataFrame(_monthly_rows)
                st.dataframe(_df_monthly, width="stretch", hide_index=True)

            # ── Edge buckets ──────────────────────────────────────────────────
            with _bt_tab_buckets:
                st.caption(
                    "Edge bucket = how far above 52.4% breakeven the model was at bet time. "
                    "Higher edge should mean higher win rate if the model is calibrated."
                )
                _buckets = _BT.edge_buckets(_results)
                _bk_rows = []
                for _bk, _bv in _buckets.items():
                    _bk_rows.append({
                        "Edge bucket": _bk,
                        "N bets":      _bv["n"],
                        "Win%":        f"{_bv['win_rate']:.1%}",
                        "Avg edge":    f"{_bv['avg_edge']:+.1%}",
                        "P&L":         f"${_bv['pnl']:+,.0f}",
                    })
                _df_buckets = pd.DataFrame(_bk_rows)
                st.dataframe(_df_buckets, width="stretch", hide_index=True)

            # ── All bets log ──────────────────────────────────────────────────
            with _bt_tab_bets:
                # Build CLV lookup keyed by (home_name, away_name) — average across bookmakers
                _clv_by_game: dict[tuple[str, str], float] = {}
                for _cr in _clv_recs:
                    _key = (_cr.get("home_team", ""), _cr.get("away_team", ""))
                    _val = _cr.get("clv_vs_closing") or 0.0
                    if _key in _clv_by_game:
                        _clv_by_game[_key] = (_clv_by_game[_key] + _val) / 2
                    else:
                        _clv_by_game[_key] = _val

                _bet_rows = []
                for b in _results.bets:
                    _clv_val = _clv_by_game.get((b.home_name, b.away_name))
                    _bet_rows.append({
                        "Date":       b.game_date,
                        "Matchup":    f"{b.away_name} @ {b.home_name}",
                        "Bet":        "HOME" if b.bet_on_home else "AWAY",
                        "Model Prob": f"{b.model_prob:.1%}",
                        "Edge":       f"{b.edge:+.1%}",
                        "Result":     "W" if b.won else "L",
                        "P&L":        f"${b.pnl:+,.0f}",
                        "CLV":        f"{_clv_val:+.2%}" if _clv_val is not None else "-",
                    })
                _df_bets = pd.DataFrame(_bet_rows)

                def _color_result(val):
                    if val == "W": return f"color: {NORD['green']}"
                    if val == "L": return f"color: {NORD['red']}"
                    return ""

                st.dataframe(
                    _df_bets.style.map(_color_result, subset=["Result"]),
                    width="stretch",
                    hide_index=True,
                )

            # ── vs Closing Line ───────────────────────────────────────────────
            with _bt_tab_clv:
                st.caption(
                    "Cross-references backtest bets against CLV records collected by poll_odds.py. "
                    "Only games present in both datasets are shown. "
                    "| **Bet beat close** — YES when our model had positive CLV on the bet side. "
                    "This is the real-world sharpness test: did the bet we placed beat the closing line?"
                )
                # Filter to bets that have CLV data
                _clv_bets = [
                    b for b in _results.bets
                    if (b.home_name, b.away_name) in _clv_by_game
                ]
                if not _clv_bets:
                    st.info(
                        "No overlap between backtest bets and CLV records yet. "
                        "CLV data is collected by poll_odds.py going forward — "
                        "as more games accumulate this tab will fill in."
                    )
                else:
                    _clv_rows_bt = []
                    for b in _clv_bets:
                        _raw_clv = _clv_by_game[(b.home_name, b.away_name)]
                        # CLV is from home team perspective; flip sign if we bet away
                        _bet_clv = _raw_clv if b.bet_on_home else -_raw_clv
                        _clv_rows_bt.append({
                            "Date":     b.game_date,
                            "Matchup":  f"{b.away_name} @ {b.home_name}",
                            "Bet":      "HOME" if b.bet_on_home else "AWAY",
                            "Result":   "W" if b.won else "L",
                            "CLV (home side)": f"{_raw_clv:+.2%}",
                            "CLV (bet side)":  f"{_bet_clv:+.2%}",
                            "Bet beat close":  "YES" if _bet_clv > 0 else "no",
                        })
                    _df_clv_bt = pd.DataFrame(_clv_rows_bt)

                    _n_clv_bt   = len(_clv_bets)
                    _clv_beat_n = sum(1 for r in _clv_rows_bt if r["Bet beat close"] == "YES")
                    _avg_bet_clv = sum(
                        _clv_by_game[(b.home_name, b.away_name)] * (1 if b.bet_on_home else -1)
                        for b in _clv_bets
                    ) / _n_clv_bt

                    _cb1, _cb2, _cb3 = st.columns(3)
                    _cb1.metric("Bets with CLV data", _n_clv_bt)
                    _cb2.metric("Bet beat close", f"{_clv_beat_n}/{_n_clv_bt} ({_clv_beat_n/_n_clv_bt:.0%})")
                    _cb3.metric("Avg CLV (bet side)", f"{_avg_bet_clv:+.2%}")
                    st.markdown("---")

                    def _color_clv_beat(val):
                        if val == "YES": return f"color: {NORD['green']}; font-weight: bold"
                        return ""

                    st.dataframe(
                        _df_clv_bt.style.map(_color_clv_beat, subset=["Bet beat close"]),
                        width="stretch",
                        hide_index=True,
                    )

            # ── Export ────────────────────────────────────────────────────────
            _bt_label = f"{_bt_div}_{'-'.join(str(s) for s in sorted(_bt_seasons))}_edge{_bt_edge}pct"
            if st.button("Export Backtest to CSV", key="export_bt"):
                _paths_bt = export_backtest_csvs(
                    _df_bets,
                    _df_monthly,
                    _df_buckets,
                    _bt_label,
                )
                if _paths_bt:
                    st.success(f"Exported {len(_paths_bt)} files to spreadsheets/backtests/")
                    for _p in _paths_bt:
                        st.caption(_p)

    elif _run_bt and not _bt_seasons:
        st.warning("Select at least one season.")
