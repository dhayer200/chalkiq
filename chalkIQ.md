## starting plate:
right now we're on college basketball
expand to baseball, soccer, player props (in NBA! in baseball! or soccer!)

## mission statement:
making betting a profitable, low stakes hobby

## future:
### guaranteed:
find out what's gonna be the most important for us right now
finish the model for college basketball
- live evaluation on games (X)
- implement CLV (closing line value) (x)
- implement LMA (line movement awareness) (x)
- live injury weighting (x)
- implement player singaling (x)
- backtesting (x)
- less focus on the frontend (or even complete removal) (x)
implement for baseball, soccer, NBA (team + player) with same model ()
wnba menstrual cycle tracking ()
find/make frontend afterwards (x)
newsletter ()

### potential
#### academia route:
write the paper  ()

#### betting comps, trader insight equiv (direction 1):
just direction 1, charge $5/month for daily sports insights
little email newsletter giving run down on the main sports we work in that are in season
giving our picks, our data, comparison with mainstream books

combine both would be best

---

## 03-05-2026 Claude Session Notes — Model Assessment + Strategic Direction

---

What won't beat sportsbooks:
1. **Vig (juice)**: Books charge ~4-5% per bet. Need >53% accuracy on coin-flip bets just to break even.
2. **Line efficiency**: Vegas already incorporates everything Elo knows plus injuries, player matchups, public money flow.
3. **No player-level signal**: Team missing its star PG has same rating until it loses.
4. **No line shopping / timing signal**: Elo gives a static pregame number.

Where it could edge out value:
- Early season when ratings haven't converged
- Live betting — books sometimes misprice mid-game lines
- **Bracket pools** — this is exactly what it's built for

---

### Line Movement Awareness

When a book posts a game they set an **opening line**. It moves based on sharp vs. public money, injuries, etc.

Example:
```
Opening:  Duke -5.5
Morning:  Duke -6.5   ← sharp money on Duke
Gametime: Duke -4.0   ← injury leaked, or public fade
```

If you bet Duke -5.5 and it moves to -6.5, the market agreed with you. Moved the other way = market disagreed = bad sign.

---

### Closing Line Value (CLV)

The **closing line** = final line before tipoff = most accurate price (absorbed all information).

**CLV** = did you get a better number than the closing line?

```
You bet:       Duke -4.5
Closing line:  Duke -6.5
CLV: +2 points (you got a better number)
```

Consistent positive CLV = you're finding inefficiencies before the market corrects them.
Best long-run predictor of whether a betting strategy is skilled vs. lucky.

---

### Best Sport for Quant Modeling

**Tier 1: Baseball**
- Consensus quant's sport for decades
- Clean moneyline (no spread)
- ~2,430 games per season
- Pitcher isolation: 30-40% of outcome variance from starter matchup
- Rich public data: Statcast, FanGraphs, Baseball Savant

**Tier 2: Soccer**
- Poisson goal model — goals follow Poisson distribution almost perfectly
- Model attack/defense strength → derive exact win/draw/loss probabilities mathematically
- xG (expected goals) = one of best publicly available advanced stats in any sport
- **Draw pricing is consistently mispriced** — documented historical edge

**Tier 3: NFL**
- Only 272 games — small sample
- Spread-based (harder)
- Player props and in-game live betting significantly less efficient than game lines

**Tier 4: NBA**
- ~1,230 games, highly efficient game line market
- **Best use case: player props** (points, rebounds, assists) — priced less carefully
- Fatigue modeling (back-to-backs, travel) is exploitable

**Tier 5: Tennis**
- Pure head-to-head
- Surface matters enormously (clay vs. grass vs. hard)
- In-play betting = main opportunity, point-by-point state is a natural Markov chain

What serious quant bettors focus on:
1. Soccer — highest volume, most international inefficiencies
2. Baseball — clean domestic modeling
3. Tennis in-play — Markov chain models
4. **Player props across all sports** — consistently least efficient market

**The hierarchy:**
```
Level 1: Cool dashboard (current)
Level 2: Verifiable prediction track record (add CLV tracking)
Level 3: Published/presented research (write the paper)
Level 4: Multi-sport system (expand to MLB/NBA)
Level 5: Live edge detection vs. market (model vs. odds API)
```
