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
    SVG_W    = int(RX[-1] + BW + PAD)
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

            # Elbow connector from winner to next-round slot
            if r < 3:
                wy     = ya if is_a else yb
                w_cy   = wy + BH / 2
                next_cy = all_ys[r + 1][gi] + BH / 2
                p.append(draw_line(rx + BW, w_cy, RX[r + 1], next_cy))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{SVG_W}" height="{SVG_H}">\n'
        + "\n".join(p)
        + "\n</svg>"
    )


# ── Combined bracket HTML (all 4 regions, zoomable) ──────────────────────────

def combined_bracket_html(
    regions: dict, win_prob_fn, color: str, mirror_right_side: bool = True
) -> str:
    """HTML with all 4 region brackets in a traditional zoomable 2x2 layout."""
    # Traditional quadrant placement:
    # East (top-left), South (top-right), West (bottom-left), Midwest (bottom-right)
    region_layout = [["East", "South"], ["West", "Midwest"]]
    right_side_regions = {"South", "Midwest"} if mirror_right_side else set()
    svgs = {
        r: region_bracket_svg(
            regions[r], win_prob_fn, color, mirror=(r in right_side_regions)
        )
        for r in REGIONS
    }

    blocks = ""
    for row in region_layout:
        for r in row:
            blocks += (
                f'<div style="background:{NORD["bg"]};display:inline-block">'
                f'<div style="font-size:13px;font-weight:700;letter-spacing:.1em;'
                f'margin-bottom:6px;font-family:ui-sans-serif,system-ui,Arial,sans-serif;'
                f'color:{color}">{r.upper()}</div>'
                + svgs[r]
                + "</div>"
            )

    return f"""
<style>
#zm-wrap{{overflow:auto;background:{NORD["bg"]};border-radius:8px;padding:8px;width:100%}}
#zm-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;
  width:fit-content;zoom:.5;transform-origin:top left}}
.zm-btn{{background:{NORD["bg2"]};border:1px solid {NORD["bg3"]};color:{NORD["snow1"]};
  padding:4px 16px;border-radius:5px;cursor:pointer;font-size:15px;
  font-family:ui-sans-serif,system-ui,Arial,sans-serif}}
.zm-btn:hover{{background:{NORD["bg3"]}}}
</style>
<div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">
  <button class="zm-btn" onclick="adjZ(-.1)">&#8722;</button>
  <button class="zm-btn" onclick="adjZ(.1)">+</button>
  <button class="zm-btn" onclick="setZ(1)">100%</button>
  <span id="zm-lbl"
    style="color:{NORD["snow0"]};font-size:12px;font-family:ui-sans-serif,sans-serif">
    50%
  </span>
  <span style="color:{NORD["bg3"]};font-size:11px;
    font-family:ui-sans-serif,sans-serif;margin-left:8px">
    Scroll to pan &bull; +/- to zoom
  </span>
</div>
<div id="zm-wrap">
  <div id="zm-grid">{blocks}</div>
</div>
<script>
var _z=.5;
function setZ(s){{
  _z=Math.max(.15,Math.min(2,s));
  document.getElementById('zm-grid').style.zoom=_z;
  document.getElementById('zm-lbl').textContent=Math.round(_z*100)+'%';
}}
function adjZ(d){{setZ(_z+d);}}
setZ(.5);
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

tab_rank, tab_bracket, tab_eval, tab_matchup, tab_math = st.tabs([
    "📊  Power Rankings",
    "🏆  Bracket",
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
        st.dataframe(styled_all, height=1800, use_container_width=True)

    with col_chart:
        top20_names = [name for _, name, _ in rankings[:20]]
        top20_odds  = [champ_odds.get(tid, 0) for tid, _, _ in rankings[:20]]
        fig_bar = plotly_odds_bar(top20_names[::-1], top20_odds[::-1], color)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Top 20 championship odds from 10,000 Monte Carlo bracket simulations.")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 2 — Bracket
# ════════════════════════════════════════════════════════════════════════════ #

with tab_bracket:
    st.subheader(f"Tournament Bracket | {cfg['label']} Division")
    st.caption(
        "Bracket seeded by current Elo (S-curve). "
        "Probabilities from 10,000 simulations per region. "
        "Actual Selection Sunday seeding may differ."
    )

    # Final Four visual at the top
    fig_ff = draw_final_four(regions, adv_odds, engine.names, color)
    st.pyplot(fig_ff, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Combined bracket — all 4 regions in one zoomable view
    mirror_right = st.toggle(
        "Mirror right-side regions (traditional)",
        value=True,
        help=(
            "On: South/Midwest rounds progress toward center. "
            "Off: all regions render left-to-right."
        ),
    )
    bracket_html = combined_bracket_html(
        regions, engine.win_prob, color, mirror_right_side=mirror_right
    )
    components.html(bracket_html, height=900, scrolling=True)
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
                st.dataframe(styled, use_container_width=True, height=420)
        st.caption(
            "**R64** = survives first round  ·  **R32** = Round of 32  ·  "
            "**S16** = Sweet 16  ·  **E8** = Elite 8  ·  "
            "**FF** = Final Four  ·  **Title** = Championship"
        )


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 3 — Model Evaluation
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
        st.plotly_chart(fig_cal, use_container_width=True)

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
# TAB 4 — Matchup Calculator
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
        st.plotly_chart(fig_mu, use_container_width=True)
        st.caption("Neutral court assumption. Elo home-court adjustment not applied here.")


# ════════════════════════════════════════════════════════════════════════════ #
# TAB 5 — Math
# ════════════════════════════════════════════════════════════════════════════ #

with tab_math:
    st.subheader("The Math Behind This Dashboard")
    st.markdown(
        "Everything shown (rankings, win probabilities, bracket odds) comes from a small set "
        "of clean mathematical ideas. This page walks through each one with the formula and a "
        "plain-English explanation of what it means and why it works."
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

    # ── 12. Parameters used ──────────────────────────────────────────────────
    with st.expander("12 · Model parameters used in this dashboard"):
        st.markdown(
            "| Parameter | Value | What it controls |\n"
            "|---|---|---|\n"
            "| $K$ (update factor) | 24 | How fast ratings respond to results |\n"
            "| Initial rating | 1500 | Starting Elo for all new teams |\n"
            "| Home advantage | 100 pts | Temporary boost for home team |\n"
            "| Elo scale | 400 | Sensitivity of win probability to rating gap |\n"
            "| Simulations | 100,000 | Monte Carlo runs for bracket odds |\n"
            "| Season | Nov 4, 2025 – Feb 23, 2026 | Data range for ratings |\n"
            "| Top $N$ seeded | 64 | Teams included in bracket simulation |"
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
