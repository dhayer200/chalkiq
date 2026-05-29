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
        "avg_total":    130.0,
    },
}

N_SIMS = 100_000




# ── Data loading (cached per division) ────────────────────────────────────────

@st.cache_resource(ttl=3600, show_spinner="Loading game data…")
def load_engine(division: str) -> EloEngine:
    from src.ratings.elo import SPORT_CONFIGS
    cfg = DIVISION_CONFIG[division]
    games = fetch_season(
        cfg["season_start"], cfg["season_end"],
        cache_dir=cfg["cache_dir"],
        division=division,
        verbose=False,
    )
    elo_cfg = SPORT_CONFIGS.get(division, SPORT_CONFIGS["mens"])
    engine = EloEngine(
        k=elo_cfg["k"],
        home_advantage=elo_cfg["home_advantage"],
        scale=elo_cfg.get("scale", 300.0),
        season_regress=elo_cfg["season_regress"],
        tempo_adjust=elo_cfg["tempo_adjust"],
        season_boundary=elo_cfg.get("season_boundary", "ncaa"),
    )
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
st.caption("*the favorites win*")
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

tab_rank, tab_bracket, tab_eval, tab_math, tab_backtest = st.tabs([
    "📊  Power Rankings",
    "🏆  Bracket",
    "📈  Model Evaluation",
    "📐  Math",
    "📉  Backtest",
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

    with st.expander("18 · Offensive Efficiency (player scatter X-axis)"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Formula**")
            st.latex(r"\text{Off Eff} = \text{TS\%} \times 100 + \frac{\text{AST}}{36} - 1.5 \times \frac{\text{TO}}{36}")
            st.markdown("where **True Shooting %** is:")
            st.latex(r"\text{TS\%} = \frac{\text{PTS}}{2 \times (\text{FGA} + 0.44 \times \text{FTA})}")
            st.markdown(
                "All per-36 values are normalised: raw stat \u00f7 avg minutes \u00d7 36, "
                "so part-time and full-time players are on equal footing."
            )
            st.markdown("**Worked example** \u2014 20 PTS, 5 AST, 2 TO, 14 FGA, 4 FTA, 32 MIN:")
            st.latex(r"\text{TS\%} = \frac{20}{2(14 + 0.44 \times 4)} = \frac{20}{31.52} \approx 63.5\%")
            st.latex(r"\text{Off Eff} = 63.5 + 5.6 - 1.5 \times 2.25 = 65.7")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Offensive Efficiency answers: how many quality offensive actions does this player "
                "produce per 36 minutes, weighted by how efficient those actions are?\n\n"
                "**True Shooting %** is the foundation. It measures scoring efficiency across "
                "all three shot types using the 0.44 free-throw multiplier (the standard adjustment "
                "for and-ones and technical free throws). A player who scores 20 points by making "
                "layups and foul shots is more efficient than one who scores 20 on mid-range jumpers "
                "\u2014 TS% captures that.\n\n"
                "**+ AST/36** rewards playmaking. A pass that creates a basket has roughly the same "
                "possession value as a made 2-pointer.\n\n"
                "**\u2212 1.5 \u00d7 TO/36** penalises turnovers. The 1.5 weight reflects that "
                "surrendering a possession costs more than an assist earns \u2014 the opponent "
                "often converts it into easy points.\n\n"
                "**Per-36 normalisation** puts a 20-minute role player and a 35-minute starter "
                "on the same scale so the comparison is fair."
            )

    with st.expander("19 · Defensive Impact (player scatter Y-axis)"):
        col_f, col_e = st.columns([1, 1])
        with col_f:
            st.markdown("**Formula**")
            st.latex(r"\text{Def Impact} = 3 \cdot \text{STL}_{36} + 2 \cdot \text{BLK}_{36} + 0.3 \cdot \text{DREB}_{36} - \text{PF}_{36}")
            st.markdown("(subscript 36 = per-36-minute normalised rate)")
            st.markdown(
                "| Component | Weight | Rationale |\n"
                "|---|---|---|\n"
                "| STL | 3 | Turnover forced + live-ball transition opportunity |\n"
                "| BLK | 2 | Shot denied; ball stays live (no transition bonus) |\n"
                "| DREB | 0.3 | Secures stop; shared team activity, lower individual credit |\n"
                "| PF | \u22121 | Gifts opponent free-throw possessions |\n"
            )
            st.markdown("**Worked example** \u2014 1.5 STL, 0.8 BLK, 5.0 DREB, 2.0 PF per 36:")
            st.latex(r"\text{Def Impact} = 3(1.5) + 2(0.8) + 0.3(5.0) - 2.0 = 4.5 + 1.6 + 1.5 - 2.0 = 5.6")
        with col_e:
            st.markdown("**Simple explanation**")
            st.markdown(
                "Defensive Impact answers: how much does this player disrupt the opponent per 36 min?\n\n"
                "**Steals (\u00d73)** are the highest-value defensive play. A steal ends the "
                "opponent's possession, prevents a shot, and generates a transition opportunity "
                "\u2014 all at once. That triple benefit justifies the highest weight.\n\n"
                "**Blocks (\u00d72)** deny a shot but the ball can land anywhere. The opponent "
                "may retain possession, so the credit is lower than a steal.\n\n"
                "**Defensive rebounds (\u00d70.3)** finish a stop by securing the ball. "
                "The weight is low because rebounding is a team activity \u2014 someone has "
                "to get it, and position matters more than individual skill.\n\n"
                "**Personal fouls (\u22121)** are subtracted because they hand the opponent "
                "free throws at 75%+ efficiency. A physically aggressive defender who fouls "
                "constantly creates more damage than they prevent.\n\n"
                "**Quadrant reading:** Top-right = two-way stars. Top-left = defensive anchors. "
                "Bottom-right = efficient scorers who don't defend. Bottom-left = players "
                "a contender cannot carry."
            )


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
        "Live CLV tracking will resume when the season is active."
    )
    st.markdown("---")

    _bt_col1, _bt_col2, _bt_col3 = st.columns([2, 2, 2])
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
    _run_bt = st.button("Run Backtest", type="primary", key="run_bt")

    _ALL_SEASON_RANGES: dict[str, dict[int, tuple]] = {
        "mens": {
            2023: (date(2022, 11, 7),  date(2023, 4, 3)),
            2024: (date(2023, 11, 6),  date(2024, 4, 8)),
            2025: (date(2024, 11, 4),  date(2025, 4, 7)),
            2026: (date(2025, 11, 4),  date.today()),
        },
        "womens": {
            2023: (date(2022, 11, 7),  date(2023, 4, 3)),
            2024: (date(2023, 11, 6),  date(2024, 4, 8)),
            2025: (date(2024, 11, 4),  date(2025, 4, 7)),
            2026: (date(2025, 11, 4),  date.today()),
        },
    }
    @st.cache_data(show_spinner="Running backtest…")
    def _run_backtest(division: str, seasons: tuple[int, ...], min_edge: float):
        from src.backtest.engine import Backtester
        from src.utils.data import fetch_season

        all_games: list[dict] = []
        _sr = _ALL_SEASON_RANGES.get(division, _ALL_SEASON_RANGES["mens"])
        for _s in sorted(seasons):
            if _s not in _sr:
                continue
            _start, _end = _sr[_s]
            _cache = f"data/raw/{division}/{_s}" if _s < 2026 else f"data/raw/{division}"
            _games = fetch_season(_start, _end, cache_dir=_cache, division=division, verbose=False)
            all_games.extend(_games)

        from src.ratings.elo import SPORT_CONFIGS as _SC
        _ecfg = _SC.get(division, _SC["mens"])
        bt = Backtester(
            k=_ecfg["k"], home_advantage=_ecfg["home_advantage"],
            scale=_ecfg.get("scale", 300.0), decay_half_life=0.0,
            season_regress=_ecfg["season_regress"], tempo_adjust=_ecfg["tempo_adjust"],
            season_boundary=_ecfg.get("season_boundary", "ncaa"),
        )
        return bt.run(all_games, min_edge=min_edge, stake=100.0, warmup_games=50)

    if _run_bt and _bt_seasons:
        with st.spinner("Running backtest…"):
            _results = _run_backtest(
                _bt_div,
                tuple(sorted(_bt_seasons)),
                _bt_edge / 100,
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

            # ── Monthly breakdown ─────────────────────────────────────────────
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

    elif _run_bt and not _bt_seasons:
        st.warning("Select at least one season.")
