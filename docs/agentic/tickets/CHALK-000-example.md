---
id: CHALK-000
title: Example ticket (Kapare template)
tier: 2
status: example
---

## Problem

One sentence describing the user or system pain.

## User stories

**US-A:** As a [role], I want [goal], so that [outcome].

## Acceptance criteria

- [ ] Observable behavior 1
- [ ] Observable behavior 2
- [ ] `bash scripts/pre-ship.sh` passes

## Tier rationale

T2 — UI / read path; no new domain rules. (Use T3 if touching Elo, CLV, slate, or backtest.)

## Kapare checklist

- [ ] 1 Slicer — this ticket
- [ ] 2–3 Stories + Challenger (inline or separate)
- [ ] 4 Planner — `docs/agentic/work/CHALK-000/plan.md` if T2+ risk
- [ ] 5 Test Architect — failing tests **before** code if T3+
- [ ] 7 Builder — implement on `feature/CHALK-000-short-name`
- [ ] 8–9 Test green + `bash scripts/pre-ship.sh`
- [ ] 10 Bugbot — T3+ only
- [ ] 11 Dockeeper — ticket + `.env.example` if needed
- [ ] 13 Shipper — PR when user asks

## References

- Playbook: [/Users/mbp/brain/projects/deep-onboarding.md](/Users/mbp/brain/projects/deep-onboarding.md)
- Dev guide: [../../DEVELOPMENT.md](../../DEVELOPMENT.md)
