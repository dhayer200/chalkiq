# Autonomous operation

ChalkIQ runs without manual CLI commands. All data collection is scheduled.

## The cold-start fix

Last season's pitfall: tracking didn't start when the season started, so the model had no baseline.

Now **every daily run**:
1. **Backfills full prior seasons** (CBB 2024–25, CFB 2024 + 2025) — fills any missing days
2. **Re-fetches the last 7 days** (late score corrections)
3. **July–August pre-season cron** runs daily with extra box score fetching before CFB kickoff

You should never start a season at 0% data coverage again.

## Schedules

| Job | When | What |
|-----|------|------|
| **Daily collect** | Midnight ET | Backfill, games, odds, injuries, box scores, player ratings, newsletter, SMS |
| **Pre-season prep** | Daily Jul–Aug 6am UTC | Same pipeline, skip odds, heavy CFB backfill |
| **Game day odds** | Fri–Mon every 2h | Odds + CLV + SMS update |
| **Newsletter send** | 14:00 UTC (Vercel) | Email subscribers |

## SMS briefings → 5128508472

**No Twilio or SMS API required.** Uses email-to-SMS via your existing Resend account:

```
5128508472 → 5128508472@txt.att.net  (AT&T)
```

Configured in GitHub secrets / `.env`:
```
SMS_BRIEFING_TO=5128508472
SMS_CARRIER=att
RESEND_API_KEY=...
```

If you're on T-Mobile or Verizon, set `SMS_CARRIER=tmobile` or `verizon`, or set `SMS_EMAIL` directly.

**Optional (Mac only):** `IMESSAGE_BRIEFING=true` sends via Apple Messages — works on local cron, not GitHub Actions.

Sample briefing:
```
ChalkIQ Jun 27
CBB: kickoff in 130d | data 98.2%
  Top: 1.Michigan 1892, 2.Duke 1888, 3.Florida 1859
CFB: kickoff in 57d | data 12.4%
CLV +3.9% avg | 57% beat close (1315g)
```

## Player effectiveness (0–100)

After box scores are fetched, each player gets a **position effectiveness score**:
- **50** = average starter at their position
- **70+** = very good | **85+** = elite

Exported to `data/players/{division}_effectiveness.json`. CBB uses Hollinger Game Score; CFB uses position-specific composites (QB passer proxy, RB ypc, WR production, defensive stats).

## GitHub setup

Repository secrets:
- `ODDS_API_KEY`
- `DATABASE_URL`
- `RESEND_API_KEY`

Push to GitHub to activate workflows.

## Local test

```bash
python scripts/daily_collect.py --skip-odds --skip-sms   # backfill only
python scripts/daily_collect.py --skip-sms               # full minus SMS
```
