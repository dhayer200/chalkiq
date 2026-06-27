# ChalkIQ — Development guide

**SDLC playbook:** [/Users/mbp/brain/projects/deep-onboarding.md](/Users/mbp/brain/projects/deep-onboarding.md)  
**Repo SDLC map:** [SDLC.md](./SDLC.md)  
**Automation:** [AUTOMATION.md](./AUTOMATION.md) · **CFB strategy:** [CFB_STRATEGY.md](./CFB_STRATEGY.md)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env — ODDS_API_KEY, DATABASE_URL, RESEND_API_KEY
```

## Daily commands

```bash
bash scripts/pre-ship.sh              # required before claiming done (Kapare step 9)
.venv/bin/pytest tests/ -v            # regression suite
python scripts/daily_collect.py --skip-sms   # autonomous pipeline (local test)
bash scripts/maintain.sh              # same as daily collect
./chalk status                        # health check
```

## Picking up work (Kapare)

1. Read or create ticket: `docs/agentic/tickets/CHALK-NNN-short-name.md`
2. Declare **tier (T0–T4)** in the ticket
3. T2+ risk / T3+ → write `docs/agentic/work/CHALK-NNN/plan.md`
4. T3+ → failing tests in `tests/` **before** implementation
5. Implement on a feature branch: `feature/CHALK-NNN-short-name`
6. `bash scripts/pre-ship.sh`
7. T3+ → Bugbot on diff; T4 → security-review
8. Commit / PR **only when user asks**

## Tier examples (ChalkIQ)

| Tier | Work | Tests |
|------|------|-------|
| T0 | Fix typo in `docs/` | None |
| T1 | Landing page CSS in `web/index.html` | pre-ship only |
| T2 | Newsletter signup UX | pre-ship + manual smoke |
| T3 | Elo tuning, CLV, slate edges, player 0–100 ratings | **pytest TDD** + Bugbot |
| T4 | Stripe webhook, subscriber DB, admin auth | TDD + security-review |

## Pre-ship (definition of done)

```bash
bash scripts/pre-ship.sh
```

Runs: Python compile check → pytest → import smoke for core modules.

## Scope

**Active:** men's college basketball, college football (Aug 2026).  
**Not in scope:** women's CBB, NBA, MLB, multi-sport expansion.

## Secrets

Never commit `.env`. Update `.env.example` when adding new env vars.
