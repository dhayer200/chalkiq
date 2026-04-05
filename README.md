# ChalkIQ

ChalkIQ is a sports prediction and quant signaling platform built around Elo, Monte Carlo simulation, calibration, CLV, live updates, and research.

## Current State

- The primary live product is the March Madness Streamlit app: https://chalk-iq.streamlit.app/
- College basketball is the strongest current sport, with MLB and NBA extensions already in the repo and still being improved.
- Newsletter, subscription, and landing-page infrastructure already exist in `api/` and `web/`.
- An automation path already exists through `scripts/maintain.sh`.

## Stack

- Python, Streamlit, Pandas, Plotly, Matplotlib
- ESPN-derived game and box score caches in `data/raw/`
- The Odds API for moneylines, CLV, and player props
- Vercel-hosted marketing/newsletter endpoints in `api/`

## Repo Layout

- `frontend/app.py`: main Streamlit product
- `src/`: ratings, live feeds, odds, players, utilities
- `scripts/`: backtests, bracket sims, fetchers, pollers, maintenance
- `web/`: landing pages and dashboard exports
- `api/`: Stripe and newsletter endpoints
- `chalk`: CLI entrypoint

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
streamlit run frontend/app.py
./chalk -h
```

## Maintenance

- `bash scripts/maintain.sh`
- `pgrep -a -f "poll_odds|poll_props"`
- Logs live in `logs/poll_odds.log`, `logs/poll_props.log`, and `logs/boxscores.log`

## Priorities

- Improve MLB and NBA evaluation, then expand to more sports
- Finish the paid product and newsletter flow
- Integrate the sports edge workflow with `ludereIQ`
- Keep the research and presentation story strong for April 2026

## Target Timing

- Automation target: April 1, 2026
- UT Austin Math for All presentation target: April 5-6, 2026
