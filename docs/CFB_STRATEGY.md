# College Football Strategy

ChalkIQ scope: **men's college basketball** (live) and **men's college football** (Aug 2026). This doc captures what went wrong in CBB and how CFB should differ from day one.

## What we got wrong in men's CBB

### 1. Scope creep before product-market fit
We built NBA, MLB, EPL, MLS, NHL, UFC, volleyball, tennis paths while the newsletter had **zero subscribers** and the bet sheet was **empty**. Multi-sport plumbing burned Odds API budget and maintenance attention without users.

**CFB fix:** No new sports until CFB slate + CLV are validated. Two divisions only: `mens`, `cfb`.

### 2. Marketing the wrong metric
The landing page highlighted **+43.5% in-sample backtest ROI**. That number is walk-forward on historical odds with selection bias at a 3% edge threshold. It is not what live bettors would have captured. CLV (+3.5% to +4.7% vs closing line) is the honest signal metric.

**CFB fix:** Lead with CLV, calibration (Brier, log loss), and beat-closing rate. Never headline backtest P&L on the public site. Freeze Elo params on 2024 data before 2025 kickoff; report out-of-sample only.

### 3. Empty newsletter from day one
Daily 14:00 UTC send with `min_edge=3%` produced **no picks** most days. Model beat closing lines on probability but not enough to clear a hard edge gate. Subscribers would get empty emails.

**CFB fix:**
- Calibrate `min_edge` on 2024 FBS closing lines before launch (likely 1.5–2.5% for CFB, not 3%).
- **Saturday-only** newsletter during CFB season (Thu/Fri optional for primetime). Skip Tue/Wed sends entirely.
- Show slate even when no bets qualify: ranked games + model vs market spread, not a blank sheet.

### 4. Cold-start season ignored in product
Elo needs ~15–20 games per team before ratings separate. Early-season CLV is negative by design. We did not communicate this or down-weight early picks.

**CFB fix:**
- Week 1–4: ratings-only mode, no bet flags. Display "calibrating" in slate.
- Apply higher `min_edge` in weeks 1–4, then step down to target threshold after week 5.
- Pre-seed ratings from prior season with 33% regression (same as CBB), but **do not** backtest in-sample on the same season used to tune K/scale.

### 5. Over-built dashboard, under-built distribution
~3,600-line Streamlit app with bracket sim, player props, KenPom efficiency, NCAA361. Meanwhile chalkiq.com had no newsletter signup and Resend still sends from `onboarding@resend.dev`.

**CFB fix:**
- **Landing page first:** signup, CLV track record, link to dashboard.
- Verify chalkiq.com domain on Resend before CFB launch.
- Dashboard is secondary; CLI slate + newsletter are primary delivery.

### 6. Player props before team model was monetizable
Player Elo (Hollinger Game Score) and prop polling ran while team-level edges were below threshold. Props burn API budget (1 req per event per market).

**CFB fix:** **No player props in v1.** CFB team markets (spread + moneyline) only. Revisit QB injury impact as a team Elo adjustment in v2, not a prop engine.

### 7. Daily polling on a weekly sport
CBB runs daily Nov–Mar; maintain.sh polls odds every 30 min year-round defaults included dead sports.

**CFB fix:**
- `active_sports_today()` already gates API calls; extend with **CFB season window** (late Aug – early Jan).
- Sat/Sun: poll every 30 min from 10:00 ET through last kickoff.
- Mon–Fri: poll only if games scheduled (Thu/ Fri MACtion, Thanksgiving week).
- Off-season (Feb–Aug): zero CFB Odds API calls.

### 8. FCS noise and team matching debt
CBB has ~360 D1 teams; matching ESPN to Odds API names was painful but bounded. CFB adds FCS opponents, duplicate names (Miami OH/FL), and inconsistent abbreviations.

**CFB fix:**
- **FBS-only** filter from first commit (`group=80` or conference allowlist).
- Build `match.py` overrides before first odds poll, not after CLV gaps appear.
- Manual QA on opening week: log unmatched games, fix overrides same day.

## CFB-specific model choices (starting hypotheses)

| Parameter | CBB (calibrated) | CFB (starting guess) | Rationale |
|-----------|------------------|----------------------|-----------|
| K | 24 | 10 | ~12 games/team vs ~30 |
| Scale | 300 | 350 | FBS has more parity than top-heavy CBB |
| Home advantage | 100 Elo (~3 pts) | 140 Elo (~4 pts) | Stronger HFA in college football |
| Season regression | 33% | 40% | Roster/coaching turnover higher |
| Tempo adjust | Yes | No | Football scoring is drive-based, not pace-normalized like hoops |
| Season boundary | Nov–Apr | Late Aug–Jan | Academic + bowl calendar |

**Calibration protocol (required before launch):**
1. Backfill 2023 + 2024 FBS games from ESPN (`data/raw/cfb/`).
2. Grid search K × scale × home_adv on **2023 only**.
3. Validate on **2024 only** — report CLV, Brier, log loss vs closing lines.
4. Freeze params. 2025 season is live evaluation only.

## CFB v1 scope (Aug 2026 kickoff)

**Ship:**
- Division `cfb` in ESPN fetch, odds poll, slate CLI, newsletter
- FBS power rankings (Elo)
- Saturday slate: spread + moneyline edges vs DraftKings/FanDuel
- CLV tracking on completed games
- Landing page section: "College Football — live Aug 2026"

**Do not ship in v1:**
- CFP bracket simulation (different structure from NCAAB; build in v2)
- Player props
- Live in-game win probability (needs football-specific σ and quarter clock)
- Women's sports, NBA, MLB, or any other division

## Division registry checklist

When implementing `cfb`, touch these files (same pattern as `mens`):

| File | Entry |
|------|-------|
| `src/utils/data.py` | `"cfb": ("football", "college-football")` |
| `src/live/feed.py` | Same + 4×15 min quarter clock |
| `src/signals/injuries.py` | Same ESPN path |
| `src/odds/api.py` | `SPORT_NCAAF = "americanfootball_ncaaf"` |
| `src/slate/generate.py` | `ODDS_SPORT`, `DIVISION_LABELS`, season start Aug |
| `src/ratings/elo.py` | `SPORT_CONFIGS["cfb"]` |
| `src/odds/match.py` | FBS team overrides |
| `scripts/maintain.sh` | `DIVISIONS=mens,cfb` in season |
| `frontend/app.py` | Add CFB toggle when ready |

## Success criteria for CFB 2025–26 evaluation season

- [ ] CLV vs closing line ≥ +2% on FBS games weeks 5–15 (out-of-sample)
- [ ] Beat-closing rate ≥ 55% on spread-implied probs
- [ ] Newsletter sends on 100% of Saturdays with games (non-empty content)
- [ ] Zero Odds API calls outside CFB season window
- [ ] Domain-verified email from `@chalkiq.com`

## North star

**Profitable hobby, low stakes:** help subscribers find mispriced college football lines before the market closes, with a verified CLV track record. Bracket pools and March Madness were CBB's natural fit; CFB's equivalent is **weekly slate + bowl/CFP season** — not copying the basketball bracket tab on day one.
