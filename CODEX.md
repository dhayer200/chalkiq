# CODEX.md — ChalkIQ Handoff Document

> This file is written for OpenAI Codex (or any AI agent) picking up this project cold.
> It consolidates everything needed to understand, run, and continue ChalkIQ without prior context.
> The final section tells you how to hand back to Claude when done.

---

## 1. What Is ChalkIQ

ChalkIQ is a **sports prediction and quantitative signaling platform** built in Python.

**Mission:** Make sports betting a profitable, low-stakes hobby through rigorous quant modeling.

**Live products:**
- Streamlit dashboard: https://chalk-iq.streamlit.app/
- Landing page + newsletter system: https://chalkiq.com (Vercel-hosted)

**Owner:** Deep (project owner, handles strategy and deployment)

**Context:** This was presented at UT Austin Math for All on April 5-6, 2026 and is part of a larger
portfolio of IQ-branded projects (CompIQ, FridgeIQ, LudereIQ, AlgoIQ, etc.) tracked in
`/Users/mbp/brain/projects/README.md`.

---

## 2. Stack

| Layer | Technology |
|-------|-----------|
| Core model | Python 3.11+ |
| Dashboard | Streamlit + Plotly + Matplotlib |
| Data manipulation | Pandas |
| Live game data | ESPN Scoreboard API (free, no key needed) |
| Odds / CLV / props | The Odds API (budgeted, 20k req/month) |
| Backend (newsletter/Stripe) | Vercel serverless Python functions |
| Database (prod) | Neon Postgres (detected via `DATABASE_URL` env var) |
| Database (local dev) | SQLite at `data/subscribers.db` |
| Email | Resend API |
| Payments | Stripe ($9/month paid tier) |
| Deployment | Streamlit Community Cloud + Vercel |

---

## 3. Environment Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Fill in .env:
```

### Required Environment Variables

| Variable | Purpose |
|----------|---------|
| `ODDS_API_KEY` | The Odds API key |
| `RESEND_API_KEY` | Resend email API key |
| `STRIPE_SECRET_KEY` | Stripe secret key |
| `STRIPE_PRICE_ID` | Stripe price ID ($9/mo plan) |
| `CRON_SECRET` | Protects Vercel cron endpoint |
| `ADMIN_SECRET` | Protects admin API |
| `SITE_URL` | `https://chalkiq.com` in prod |
| `DATABASE_URL` | Neon Postgres connection string (omit for SQLite) |

**Always use `.venv/bin/python3` to run scripts locally.**

---

## 4. How to Run Everything

### Dashboard
```bash
streamlit run frontend/app.py
# Live at http://localhost:8501
```

### CLI (chalk)
```bash
./chalk slate                     # Today's CBB betting slate
./chalk slate --division nba      # NBA slate
./chalk slate --yesterday         # Yesterday's results
./chalk slate --week              # Next 7 days
./chalk bracket --sport cbb       # NCAA tournament bracket sim
./chalk player "Cameron Boozer"   # Player profile
./chalk leaders --top 50          # Player leaderboard
./chalk team "Duke"               # Team roster
./chalk backtest                  # Walk-forward Elo backtest
./chalk odds --once               # Fetch live odds once
./chalk props --once              # Fetch player props once
./chalk injuries --once           # Fetch injury reports once
./chalk boxscores                 # Fetch latest box scores
./chalk clv-backfill              # Backfill historical CLV
./chalk status                    # System health check
```

### Maintenance (run 2x/day)
```bash
bash scripts/maintain.sh              # Full refresh
bash scripts/maintain.sh --no-pull    # Skip git pull
DIVISIONS=mens,nba bash scripts/maintain.sh  # Custom sports
```

`maintain.sh` does in order:
1. `git pull`
2. Kill old daemons
3. `fetch_boxscores.py`
4. Restart `poll_odds.py` (every 30 min)
5. Restart `poll_props.py` (every 60 min)
6. Restart `poll_injuries.py` (every 10 min)
7. Run `generate_newsletter_data.py`

### Check Running Daemons
```bash
pgrep -a -f "poll_odds|poll_props|poll_injuries"
tail -f logs/poll_odds.log
tail -f logs/poll_props.log
tail -f logs/poll_injuries.log
```

---

## 5. File Structure (Critical Files)

```
chalkIQ/
  chalk                         # CLI entrypoint
  main.py                       # Original milestone-0 pipeline
  requirements.txt
  vercel.json                   # Vercel config + cron schedule (14:00 UTC daily)
  CLAUDE.md                     # Claude's instructions (keep in sync)
  CODEX.md                      # This file

  frontend/
    app.py                      # Main Streamlit dashboard (~3,600 lines)
                                # Tabs: Power Rankings | Bracket | Model Eval |
                                #       Matchup | Math | NCAA361 | Signals |
                                #       Backtest | Players | Sources

  src/
    ratings/elo.py              # EloEngine — core rating system
    bracket/simulator.py        # Monte Carlo bracket simulator (100k sims)
    bracket/structure.py        # NCAA S-curve seeding + bracket slots
    live/feed.py                # ESPN scoreboard: fetch_live_games(), fetch_other_games()
    live/model.py               # Live win probability (Stern 1994 random-walk + Elo)
    odds/api.py                 # Odds API client: fetch_odds(), fetch_player_props()
    odds/clv.py                 # CLV computation + line movement detection
    odds/match.py               # Fuzzy team name matching (ESPN <-> Odds API)
    odds/store.py               # JSONL persistence layer
    odds/budget.py              # QuotaBudget: tracks 20k/month API limit
    players/engine.py           # PlayerEloEngine (K=16, Hollinger Game Score)
    players/gamescores.py       # Game Score formula + ESPN box score parsing
    players/espn.py             # ESPN player stats aggregation
    signals/injuries.py         # ESPN injury polling + status change detection
    predictions/pregame.py      # matchup_prob(), matchup_table() wrappers
    slate/generate.py           # generate_slate() — central function for CLI + newsletter
    backtest/engine.py          # Walk-forward Elo backtest with P&L simulation
    utils/data.py               # fetch_season(), fetch_day() with disk caching
    utils/metrics.py            # log_loss(), brier_score(), calibration_bins()
    utils/efficiency.py         # KenPom-style adjusted efficiency (20-pass iterative)
    utils/rating_history.py     # Rating trajectories, market movers, NCAA361 index

  scripts/
    maintain.sh                 # Master 2x/day automation script
    poll_odds.py                # Daemon: moneylines every 30 min
    poll_props.py               # Daemon: player props every 60 min
    poll_injuries.py            # Daemon: ESPN injuries every 10 min
    fetch_boxscores.py          # Incremental box score fetcher
    daily_slate.py              # CLI slate printer
    bracket.py                  # CLI bracket simulation
    backtest.py                 # CLI backtest runner
    player_profile.py           # CLI player card, leaderboard, team roster
    export_dashboard.py         # Export JSON to web/assets/
    generate_newsletter_data.py # Pre-generate newsletter content (called by maintain.sh)
    fetch_historical_clv.py     # Backfill historical CLV (paid API tier)

  api/                          # Vercel serverless functions
    _shared/db.py               # Dual-backend DB (Neon Postgres / SQLite)
    subscribe.py                # POST /api/subscribe
    unsubscribe.py              # GET /api/unsubscribe?token=...
    checkout.py                 # POST /api/checkout (Stripe)
    stripe_webhook.py           # POST /api/stripe_webhook
    send_newsletter.py          # POST /api/send_newsletter (Vercel cron, 14:00 UTC)
    admin.py                    # GET/POST /api/admin

  web/                          # Vercel static site
    index.html                  # Landing page (Tailwind CSS, Chart.js)
    dashboard.html              # Data dashboard
    admin.html                  # Admin panel UI
    js/charts.js
    js/signup.js
    assets/data.json
    assets/dashboard.json

  data/
    raw/mens/                   # Men's CBB game JSONs (ESPN cache)
    raw/womens/                 # Women's CBB
    raw/nba/                    # NBA game data
    raw/mlb/                    # MLB game data
    odds/snapshots.jsonl        # Every odds fetch
    odds/clv.jsonl              # CLV records for completed games
    odds/historical_clv.jsonl   # Backfilled historical CLV
    odds/alerts.jsonl           # Line movement alerts
    odds/props.jsonl            # Player prop snapshots
    odds/quota.json             # API budget tracking
    signals/injury_state.json   # Last-known player injury statuses

  logs/                         # Daemon + cron output (NOT source, but tracked in git — hygiene issue)
  paper/chalkiq.pdf             # Academic paper (compiled from chalkiq.typ)
```

---

## 6. Key Architecture

### Elo Engine (`src/ratings/elo.py`)
- All teams start at **1500**
- Formula: standard Elo + margin-of-victory scaling (FiveThirtyEight method)
- `mov_factor = ln(|margin| + 1) * 2.2 / (rating_gap * 0.001 + 2.2)`
- Autocorrelation correction prevents blowout inflation
- Season regression: **33% toward 1500** each year
- Tempo normalization: MoV adjusted by estimated possessions (total_pts / 2.0, normed to 70-possession baseline)
- Sport-specific configs live in `SPORT_CONFIGS` dict in `elo.py`

### Key Constants

| Parameter | Value | File |
|-----------|-------|------|
| CBB Elo K | 24.0 | `src/ratings/elo.py` |
| CBB Elo scale | 300.0 | `src/ratings/elo.py` |
| Home advantage | 100.0 Elo pts | `src/ratings/elo.py` |
| Season regression | 33% toward 1500 | `src/ratings/elo.py` |
| Initial rating | 1500.0 | `src/ratings/elo.py` |
| Player Elo K | 16.0 | `src/players/engine.py` |
| Monte Carlo sims (dashboard) | 100,000 | `frontend/app.py` |
| Monte Carlo sims (CLI) | 10,000 | `main.py` |
| Min edge for betting | 3% (0.03) | `src/slate/generate.py` |
| Breakeven at -110 | 52.38% | `src/backtest/engine.py` |
| Live model sigma | 2.0 pts/sqrt(min) | `src/live/model.py` |
| Efficiency iterations | 20 passes | `src/utils/efficiency.py` |
| Odds poll interval | 30 min | `scripts/maintain.sh` |
| Props poll interval | 60 min | `scripts/maintain.sh` |
| Injury poll interval | 10 min | `scripts/maintain.sh` |
| Newsletter send time | 14:00 UTC daily | `vercel.json` |
| API budget limit | 20,000 req/month | `src/odds/budget.py` |

### Sport-Specific Elo Configs
- NBA: K=12 | MLB: K=6 | EPL: K=8 | MLS: K=10 | NHL: K=8 | UFC: K=40

### CLV (Closing Line Value)
`model_prob - closing_prob`. Positive = model was ahead of market.
Tracked in `data/odds/clv.jsonl`. Summary via `clv_summary()` in `src/odds/clv.py`.
Calibrated result: +3.5% average CLV with current CBB model (scale=300).

### Newsletter Flow
1. `generate_newsletter_data.py` runs (via `maintain.sh`)
2. Calls `generate_slate()` for each active sport
3. Saves to `newsletter_content` DB table
4. Vercel cron at 14:00 UTC hits `POST /api/send_newsletter`
5. Reads content, builds tier-specific HTML (free: 2 picks + CTA; paid: all picks)
6. Sends via Resend with per-subscriber unsubscribe tokens

### Subscriber DB (`api/_shared/db.py`)
- Prod: Neon Postgres (env var `DATABASE_URL` starts with `postgres`)
- Local: SQLite at `data/subscribers.db`
- Tables: `subscribers`, `newsletter_sends`, `newsletter_content`

---

## 7. Style Conventions (MUST FOLLOW)

- **Nord dark theme** throughout dashboard (`NORD` color dict in `frontend/app.py`)
- Men's division: `#88C0D0` (frost1 light blue). Women's: `#B48EAD` (purple)
- **No em dashes** in UI strings — use `,` `:` `;` or `|` instead
- Subheaders: `|` separator — e.g. `"Top 64 | Men's Division"`
- Table zero placeholders: `"-"` (not `"--"`)
- Use `width='stretch'` NOT `use_container_width=True` (deprecated Streamlit API)
- JSONL for all time-series data (one JSON object per line)
- Metric labels use `:` — e.g. `"Log Loss: Elo"`

---

## 8. Current State as of April 5, 2026

### What Works (Production-Ready)
- Elo rating system (CBB men's + women's), calibrated and validated
- Monte Carlo bracket simulation (100k sims)
- KenPom-style adjusted efficiency ratings
- Live Streamlit dashboard (3,600+ lines, 10 tabs)
- Odds integration via The Odds API
- CLV tracking and analysis pipeline
- Line Movement Awareness (LMA) with sharp money detection
- Player Elo engine (K=16, Hollinger Game Score)
- Player props edge detection
- ESPN injury polling with status change detection
- Backtesting engine with P&L, monthly breakdown, edge buckets
- Newsletter system (Resend + Vercel cron)
- Stripe checkout + webhook for paid tier ($9/mo)
- Admin panel API
- CLI (`./chalk`) with 13+ commands
- Automation (`scripts/maintain.sh`) running 2x/day
- Polling daemons for odds/props/injuries
- Budget-aware API management
- Live win probability (Stern 1994 method)
- Data export to JSON for landing page
- Academic paper (`paper/chalkiq.pdf`)

### What Is NOT Done (Open Tasks)
1. **Newsletter has 0 subscribers** — system is fully built, needs users
2. **Resend domain not verified** — currently sending from `onboarding@resend.dev`, need to verify `chalkiq.com`
3. **Bet sheet is empty** — model edges are below the 3% `min_edge` threshold; either lower threshold or improve model
4. **Multi-sport expansion** — MLB, NBA, EPL, MLS, NHL, UFC pipelines exist but coverage is uneven
5. **Repo hygiene** — 1,095+ uncommitted changes mixing live data (logs, JSONL) with source code
6. **Historical CLV backfill** — `fetch_historical_clv.py` exists; 20K Odds API key active (19,250 req remaining)

### Uncommitted Changes as of Last Claude Session
The following tracked files have substantial uncommitted changes:
- `api/_shared/db.py` (+66 lines) — Neon Postgres dual-backend additions
- `api/send_newsletter.py` (+138/-? lines) — newsletter improvements
- `scripts/daily_slate.py` (+603 lines) — major expansion
- `scripts/maintain.sh` (+45 lines) — injury daemon added
- `scripts/poll_injuries.py` (+115 lines) — major additions
- `scripts/poll_odds.py` (+154 lines) — major additions
- `scripts/poll_props.py` (+181 lines) — major additions
- `src/live/feed.py` (+20 lines)
- `src/odds/api.py` (+154 lines) — major additions
- `src/ratings/elo.py` (+44 lines)
- `src/signals/injuries.py` (+25 lines)
- `src/utils/data.py` (+4 lines)
- Live data files (logs, JSONL) — should NOT be committed, add to `.gitignore`

**Priority action:** Commit source changes, add logs/JSONL/data to `.gitignore`.

---

## 9. Data Flow

```
ESPN Scoreboard API (free, no key)
    |
    v
fetch_season() --> data/raw/{division}/*.json (disk-cached per day)
fetch_boxscores.py --> data/raw/{division}/boxscores/*.json
    |
    v
EloEngine.process_games() --> team ratings
PlayerEloEngine.process_games() --> player ratings
compute_efficiency() --> KenPom-style adjusted stats
    |
    v
The Odds API (budgeted, key required)
    |
    v
poll_odds.py --> data/odds/snapshots.jsonl
             --> data/odds/clv.jsonl (on game completion)
             --> data/odds/alerts.jsonl (line moves)
poll_props.py --> data/odds/props.jsonl
poll_injuries.py --> data/signals/injury_state.json
    |
    v
generate_slate() --> SlateResult (edges, bets, CLV stats)
    |
    v
frontend/app.py (Streamlit dashboard)
daily_slate.py (CLI)
generate_newsletter_data.py --> newsletter_content DB table
    |
    v
send_newsletter (Vercel cron 14:00 UTC) --> Resend emails
```

---

## 10. Known Issues / Gotchas

- **TLS cert fix already applied** — all pollers use `certifi.where()`. Do not remove this.
- **`active_sports_today()`** is called before any Odds API request to avoid wasting budget on days with no games.
- **`QuotaBudget`** persists to `data/odds/quota.json`. If you manually reset or delete this, the budget tracking resets.
- **Team name matching** between ESPN and The Odds API is fuzzy (`src/odds/match.py`). Manual overrides exist there — check before assuming a mismatch is a bug.
- **Streamlit deprecation:** `use_container_width=True` is deprecated — always use `width='stretch'`.
- **Dashboard is 3,600+ lines** in a single file (`frontend/app.py`). Search by tab name as a comment header to navigate.
- **SQLite vs Postgres:** `api/_shared/db.py` auto-detects based on `DATABASE_URL`. Locally, no env var needed (falls back to SQLite).
- **Logs are in git** — they should be gitignored but aren't fully. Don't modify log content directly.
- **`data/odds/*.jsonl` files** are large and growing. They should be gitignored in production; currently tracked.

---

## 11. GitHub Repository

```
https://github.com/dhayer200/chalkiq.git
```

Branch: `master` (main branch, also used for PRs)

---

## 12. Deployment

### Streamlit Dashboard
Deployed via Streamlit Community Cloud from `master` branch.
Entry point: `frontend/app.py`

### Landing Page + API
Deployed via Vercel.
- `web/` is the static output directory
- `api/*.py` are Python serverless functions
- Cron: `POST /api/send_newsletter` at `0 14 * * *` (14:00 UTC)
- Config: `vercel.json`

### Local Dev vs Production
- Local: SQLite DB, no `DATABASE_URL` needed, use `.env` file
- Production: Neon Postgres via `DATABASE_URL` env var set in Vercel dashboard

---

## 13. Recommended Next Steps for Codex

These are in priority order based on impact:

### Immediate (repo hygiene)
1. Add `logs/`, `data/odds/*.jsonl`, `data/raw/`, `data/signals/` to `.gitignore`
2. Commit all source-only changes (the 13 modified `.py` and `.sh` files listed in Section 8)
3. Separate live data from source code going forward

### Short-Term (subscriber growth)
4. Verify `chalkiq.com` with Resend to send from a professional domain
5. Add SEO meta tags and social sharing to `web/index.html`
6. Add a signup incentive (free pick preview, historical win rate chart)

### Model Improvement (bet sheet)
7. Lower `min_edge` threshold from 3% to 1-2% and observe bet sheet volume
8. Or: improve model by adding injury-adjusted Elo (already have injury data) into `generate_slate()`
9. Run `fetch_historical_clv.py` with paid Odds API tier for full 2023-2026 CLV dataset

### Multi-Sport Expansion
10. Validate NBA pipeline end-to-end (`./chalk slate --division nba`)
11. Validate MLB pipeline for 2026 season start
12. Add EPL support (Poisson goal model is better for soccer than Elo — consider adding `src/models/poisson.py`)

---

## 14. Handoff Back to Claude

When Codex has completed its work, update `CLAUDE.md` by appending the following section
at the very end of the file. Replace the placeholder text in brackets with actual details:

```
## Codex Handoff — [DATE]

Codex picked up from this CODEX.md on [DATE] and completed the following:

### Completed by Codex
- [list each task completed with brief description]

### Files Modified
- [list each file changed and what changed]

### New Files Created
- [list any new files]

### Still Open
- [list anything Codex started but did not finish]
- [list anything that was blocked or needs human input]

### Notes for Claude
- [anything surprising, non-obvious decisions made, or context Claude needs]
- Current branch: [branch name]
- Last commit: [commit hash and message]
- Run `./chalk status` to verify system health after picking up
```

After appending that section to `CLAUDE.md`, also run:
```bash
git add CLAUDE.md CODEX.md
git commit -m "docs: codex handoff notes [DATE]"
```

This ensures Claude sees the handoff context in the next session.

---

*This CODEX.md was generated on 2026-04-05 by Claude (claude-sonnet-4-6).*
*Source files consulted: CLAUDE.md, README.md, chalkIQ.md, claude-memory.md, old-paper.md,*
*vercel.json, scripts/maintain.sh, git log, git diff --stat, /brain/projects/README.md*
