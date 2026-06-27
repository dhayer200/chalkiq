# Agentic tickets (Kapare)

Tickets follow the Kapare workflow in [/Users/mbp/brain/projects/deep-onboarding.md](/Users/mbp/brain/projects/deep-onboarding.md).

## Layout

```text
docs/agentic/
  tickets/CHALK-NNN-short-name.md   # slice + stories + AC + tier
  work/CHALK-NNN/plan.md            # planner output (T2+ risk, T3+)
```

## Naming

- Prefix: `CHALK-`
- Branch: `feature/CHALK-NNN-short-name`
- Example: [CHALK-000-example.md](./tickets/CHALK-000-example.md)

## Before you build

1. Ticket exists with `tier:` set
2. T3+ → failing tests written first
3. T4 → threat model noted in plan
4. After ship → check off acceptance criteria, update Dockeeper items
