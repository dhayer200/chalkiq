# ChalkIQ

Sports prediction platform: Elo ratings, CLV tracking, autonomous data pipeline, newsletter. **Men's college basketball** (live) and **college football** (Aug 2026).

**Live:** [chalkiq.com](https://chalkiq.com) · [Streamlit dashboard](https://chalk-iq.streamlit.app/)

## SDLC

All development follows the **Kapare workflow**:

**[/Users/mbp/brain/projects/deep-onboarding.md](../deep-onboarding.md)** (canonical playbook)

| Doc | Purpose |
|-----|---------|
| [docs/SDLC.md](docs/SDLC.md) | How Kapare maps to this repo |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Setup, tiers, pre-ship, daily commands |
| [docs/agentic/tickets/](docs/agentic/tickets/) | `CHALK-NNN` tickets |

```bash
bash scripts/pre-ship.sh    # required before claiming done
```

Cursor loads `.cursor/rules/kapare-sdlc.mdc` automatically in this workspace.

## Stack

- Python · Vercel (landing + API) · Neon Postgres · GitHub Actions (autonomous pipeline)
- ESPN (games) · The Odds API (lines, CLV) · Resend (email / SMS-via-email)

## Run locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
bash scripts/pre-ship.sh
streamlit run frontend/app.py   # optional dashboard
```

## Automation

No manual CLI required in production. See [docs/AUTOMATION.md](docs/AUTOMATION.md).

- **Midnight ET:** game fetch, backfill, odds, player ratings, SMS briefing
- **14:00 UTC:** newsletter send (Vercel cron)

## Repo layout

| Path | Role |
|------|------|
| `src/` | Elo, slate, odds, players, automation |
| `api/` | Subscribe, Stripe, newsletter, admin |
| `web/` | Landing page |
| `scripts/` | Daily collect, pre-ship, maintenance |
| `docs/agentic/` | Kapare tickets and plans |
| `.cursor/rules/` | SDLC rules for agents |
