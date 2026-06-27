# ChalkIQ SDLC

ChalkIQ follows the **Kapare 13-step workflow** defined in the workspace playbook:

**[/Users/mbp/brain/projects/deep-onboarding.md](/Users/mbp/brain/projects/deep-onboarding.md)**

Do not fork that doc here. This file only maps the generic pipeline to this repo.

## Quick map

| Kapare step | ChalkIQ artifact |
|-------------|------------------|
| 1 Slicer | `docs/agentic/tickets/CHALK-NNN-*.md` with `tier:` frontmatter |
| 4 Planner | `docs/agentic/work/CHALK-NNN/plan.md` |
| 5 Test Architect | `tests/test_*.py` (failing first, T3+) |
| 7 Builder | `src/`, `api/`, `scripts/`, `web/` |
| 8 Test Green | `pytest tests/` |
| 9 Verifier | `bash scripts/pre-ship.sh` |
| 10 Bugbot | Cursor Bugbot subagent on branch diff (T3+) |
| 11 Dockeeper | Update ticket, `.env.example`, `docs/DEVELOPMENT.md` |
| 13 Shipper | PR only when user asks |

## 6-step lite vs full 13-step

| Use 6-step lite | Use full 13-step |
|-----------------|------------------|
| T0 docs, T1 UI | T2 with risk |
| Simple T2 screens | All T3–T4 (Elo, CLV, slate, automation, Stripe, DB) |

## Human gates

Never commit, push, open PR, merge, or deploy unless explicitly asked. See deep-onboarding § Human gates.

## Day-one dev

See [DEVELOPMENT.md](./DEVELOPMENT.md).
