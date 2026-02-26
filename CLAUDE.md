# README - March Madness Live Power Rankings, Real Time Predictions, and Dynamic Bracket Engine
## Research Paper + Product Requirements Document (PRD)

Author: Deep (project owner)  
Version: 1.0 (planning document, no code)  
Status: Research and product design specification  

---

## Abstract

This document defines a full March Madness analytics project that combines:

1. **Team power rankings** (Elo style rating system),
2. **Pre game matchup predictions**,
3. **Live in game win probability updates** (updated as the game is happening),
4. **Dynamic tournament bracket simulation** (updates the projected winner as game states change),
5. **A frontend dashboard** to display rankings, probabilities, and the current best bracket,
6. **A portfolio quality research framing** suitable for math, finance, and quantitative storytelling.

The project is designed to look "cool on the surface" (live bracket, changing predictions, polished frontend) while having serious quantitative depth underneath (probability, calibration, simulation, scoring rules, uncertainty, and decision theory).

---

## 1. Why this project is mathematically strong and finance relevant

March Madness prediction is a compact lab for quantitative thinking:

- **You estimate uncertainty**, not certainty.
- **You update beliefs when new information arrives** (like markets reacting to news).
- **You compare models using scoring rules**, not vibes.
- **You simulate many futures**, like risk scenarios in portfolios.
- **You manage tradeoffs between accuracy and payoff**, like balancing expected return vs risk.

### Finance analogy (core story)
This project can be presented as:

- **Teams** -> assets
- **Ratings** -> latent quality / expected performance
- **Game outcomes** -> price moves / events
- **Bracket strategies** -> portfolio allocations
- **Tournament simulations** -> Monte Carlo risk scenarios
- **Live in game updates** -> intraday repricing after new information

This framing is excellent for math, statistics, economics, applied math, actuarial interest, data science, and finance oriented applications.

---

## 2. Project vision

Build a system that answers three layers of questions:

### Layer A - Before games start
- Who is strongest?
- What are the matchup win probabilities?
- Which bracket is most likely to occur?
- Which bracket strategy maximizes expected score in a pool?

### Layer B - While a game is live
- Who is likely to win **right now**?
- How did an upset shift the tournament outlook?
- If the underdog becomes the live favorite, how does that change:
  - power rankings,
  - Final Four odds,
  - championship odds,
  - optimal bracket picks?

### Layer C - After games finish
- Was the model well calibrated?
- Did predicted 70% events happen about 70% of the time?
- Was the model overconfident or underconfident?
- Which assumptions failed?

---

## 3. Research questions

This project can be framed as a research study with the following questions:

### RQ1. Ranking quality
Can an Elo based rating model produce useful team strength rankings for NCAA tournament prediction?

### RQ2. Probabilistic forecasting quality
How well calibrated are the model's predicted win probabilities under proper scoring rules (log loss and Brier score)?

### RQ3. Live repricing
Can a live game state model (score, time remaining, possession, etc.) improve win probability estimates and meaningfully update downstream tournament outcomes in real time?

### RQ4. Decision making under uncertainty
How do bracket strategies differ when optimizing for:
- highest expected score,
- highest probability of finishing top 10,
- highest probability of winning the pool outright?

That is a very finance flavored question: same data, different objective functions.

---

## 4. System overview (high level architecture)

The project has four quantitative engines and one presentation layer.

### 4.1 Engine 1 - Ratings engine (Power rankings)
Maintains team ratings from historical game outcomes.

Output:
- Team power rankings
- Pregame expected win probabilities

### 4.2 Engine 2 - Game prediction engine (Pregame)
Transforms ratings (and optional features) into matchup probabilities.

Output:
- Probability that team i beats team j before tipoff

### 4.3 Engine 3 - Live in game probability engine
Updates the win probability continuously using game state variables.

Output:
- Live win probability curve through time
- Probability swing after big events (runs, foul trouble, injuries if modeled)

### 4.4 Engine 4 - Tournament simulation engine
Re simulates the bracket many times as probabilities change.

Output:
- Updated bracket winner projections
- Final Four odds
- Championship odds
- Dynamic "most likely bracket" (or most likely path tree)

### 4.5 Frontend dashboard
Displays:
- Live game probabilities
- Team rankings
- Updated bracket
- Scenario comparisons
- Model confidence and calibration charts

---

## 5. Mathematical foundations

This is the heart of the project.

---

## 5.1 Random variable for game outcome

Let $Y_{ij}$ be the outcome of a game between team $i$ and team $j$:
$$
Y_{ij} =
\begin{cases}
1, & \text{if team } i \text{ wins} \\
0, & \text{if team } i \text{ loses}
\end{cases}
$$
We model:
$$
Y_{ij} \sim \text{Bernoulli}(p_{ij})
$$
where $p_{ij} = P(Y_{ij}=1)$.

### What this means (plain language)
A Bernoulli random variable is just a one trial event with two outcomes:
- 1 = success (team i wins)
- 0 = failure (team i loses)

The model's job is to estimate $p_{ij}$, the chance team $i$ wins.

---

## 5.2 Elo expected win probability

A standard Elo probability model is:
$$
E_i = \frac{1}{1 + 10^{(R_j - R_i)/400}}
$$
where:
- $R_i$ = rating of team $i$
- $R_j$ = rating of team $j$
- $E_i$ = expected probability that team $i$ wins

Because this is symmetric,
$$
E_j = 1 - E_i
$$
### Why it looks logistic
Yes, it is logistic shaped. In fact, it is a logistic model written in base 10 form.

If we define rating difference $\Delta R = R_i - R_j$, then:
$$
E_i = \frac{1}{1 + 10^{-\Delta R/400}}
$$
Using $10^x = e^{x \ln 10}$, this becomes:
$$
E_i = \frac{1}{1 + e^{-(\ln 10 / 400)\Delta R}}
$$
So Elo is a logistic curve with slope constant $\ln(10)/400$.

### Why base 10 and why divide by 400?
Historically, Elo was designed for chess and used a human friendly decimal scale.
- **Base 10** is historical convention.
- **400** sets the scale of sensitivity.

A rating difference of 400 implies:
$$
E_i = \frac{1}{1+10^{-1}} = \frac{1}{1+0.1} = \frac{10}{11} \approx 0.9091
$$
So "400 points stronger" means about a 90.9% expected win rate.

That is not a law of nature. It is a modeling choice / calibration convention.

---

## 5.3 Elo rating update rule

After the game ends, ratings are updated with:
$$
R_i' = R_i + K(S_i - E_i)
$$
$$
R_j' = R_j + K(S_j - E_j)
$$
where:
- $S_i \in \{0,1\}$ is the actual result for team $i$
- $E_i$ is the predicted probability for team $i$
- $K > 0$ is the update step size (learning rate)

### Interpretation
- If team $i$ wins and was underestimated ($S_i - E_i > 0$), rating goes up.
- If team $i$ loses and was overrated ($S_i - E_i < 0$), rating goes down.
- The bigger the surprise, the larger the update.

### Example
Suppose:
- $E_i = 0.80$ (model says team $i$ had 80% chance to win)
- team $i$ loses, so $S_i = 0$
- $K = 24$

Then:
$$
R_i' = R_i + 24(0 - 0.80) = R_i - 19.2
$$
The rating drops by 19.2 points because the model expected a win but got a loss.

---

## 5.4 Optional extension: margin of victory adjustment

A simple extension is to scale the update by game margin:
$$
R_i' = R_i + K \cdot g(\text{MOV}, \Delta R) \cdot (S_i - E_i)
$$
where:
- MOV = margin of victory
- $g(\cdot)$ is a damping function

Purpose:
- reward dominant wins somewhat more,
- avoid overreacting to blowouts against weak teams.

This is optional in v1. It can be a later feature.

---

## 5.5 Proper scoring rules (how we judge probability forecasts)

You asked a great conceptual question earlier: "Is log loss proof Elo works?"

Short answer: **No.**  
Log loss is **evidence about forecast quality**, not proof of truth.

A model can score well and still be improvable. But proper scoring rules give objective, mathematically correct ways to compare probabilistic forecasts.

### 5.5.1 Log loss (cross entropy for binary outcomes)

For one game with predicted probability $p$ and observed result $y \in \{0,1\}$:
$$
\ell_{\text{log}}(p,y) = -\Big[y\log(p) + (1-y)\log(1-p)\Big]
$$
Average over $n$ games:
$$
\text{LogLoss} = \frac{1}{n} \sum_{k=1}^{n} \ell_{\text{log}}(p_k, y_k)
$$
#### Why it matters
- Rewards confident correct predictions.
- Punishes confident wrong predictions heavily.
- Encourages calibrated uncertainty.

If you predict 0.99 and lose, log loss hits you hard. Good. That is what we want.

### 5.5.2 Brier score

For one game:
$$
\ell_{\text{Brier}}(p,y) = (p-y)^2
$$
Average over $n$ games:
$$
\text{Brier} = \frac{1}{n} \sum_{k=1}^{n} (p_k-y_k)^2
$$
#### Why it matters
- Easy to interpret as squared error on probabilities.
- Less harsh than log loss on extreme mistakes.
- Good companion metric to log loss.

### Interpretation of your intuition (very good intuition)
Yes - Brier score is kind of "covering our bases" in the sense that it gives a second lens on forecast quality.  
Not exactly "correcting mistakes" by itself, but it helps us compare models from a different penalty shape.

Using both log loss and Brier is strong practice.

---

## 5.6 Calibration (do probabilities mean what they say?)

A model is calibrated if events predicted at probability $p$ occur about $p$ fraction of the time.

Example:
- all games predicted at ~70%
- if the model is calibrated, about 70% of those teams should actually win

This can be checked with:
- reliability plots
- calibration bins
- expected calibration error (optional)
- observed vs predicted win rate tables

Calibration matters a lot in finance style decision making because bad calibration causes bad risk sizing.

---

## 5.7 Monte Carlo tournament simulation

Let a bracket path be a random outcome generated from matchup probabilities.

We simulate the tournament many times (for example $N=10{,}000$ or $100{,}000$).

For simulation $s$:
- draw each game result according to its estimated probability,
- propagate winners forward,
- record champion, Final Four teams, and bracket score.

Then estimate probabilities by empirical frequency:
$$
\widehat{P}(\text{team } i \text{ wins title}) = \frac{1}{N}\sum_{s=1}^{N} \mathbf{1}\{\text{team } i \text{ wins in sim } s\}
$$
where $\mathbf{1}\{\cdot\}$ is the indicator function.

### Why this matters
Closed form exact tournament probability calculations become messy fast.
Monte Carlo is flexible and practical.

This is the same spirit as simulation based risk analysis in finance.

---

## 5.8 Bracket as a decision problem (finance style objective functions)

A "best bracket" depends on the objective.

Let $B$ be your chosen bracket strategy and $S(B)$ be its pool score random variable.

Possible optimization targets:

### Objective A - Max expected score
$$
\max_B \; E[S(B)]
$$
Good for consistent performance, but may be too conservative in large pools.

### Objective B - Max probability of top 10 finish
$$
\max_B \; P(\text{rank}(B) \le 10)
$$
More tournament / competition aware.

### Objective C - Max probability of winning the pool
$$
\max_B \; P(\text{rank}(B)=1)
$$
This often pushes toward more contrarian picks (higher variance strategy).

This is directly analogous to portfolio design:
- maximize expected return,
- or maximize chance of beating a benchmark,
- or maximize chance of finishing first.

---

## 5.9 Live in game win probability model (real time updating)

This is the feature that makes the project feel advanced.

### 5.9.1 Pregame prior from Elo
Before tipoff, start with a pregame probability $p_0$ from Elo:
$$
p_0 = \frac{1}{1 + 10^{(R_j - R_i)/400}}
$$
Convert to log odds (logit) if needed:
$$
\ell_0 = \log\left(\frac{p_0}{1-p_0}\right)
$$
### 5.9.2 Game state vector
At time $t$, define a state vector $x_t$ such as:
$$
x_t = \big(
\text{score diff}_t,\;
\text{time remaining}_t,\;
\text{possession}_t,\;
\text{foul state}_t,\;
\text{timeouts}_t,\;
\ell_0
\big)
$$
You do not need every feature in v1. A strong v1 can use:
- score differential,
- time remaining,
- possession,
- pregame prior (Elo based).

### 5.9.3 Logistic live probability model
Model live win probability as:
$$
P(\text{team } i \text{ wins} \mid x_t) = \sigma(\beta_0 + \beta^\top x_t)
$$
where:
- $\sigma(z) = \frac{1}{1+e^{-z}}$ is the logistic function
- $\beta$ are learned coefficients

### Intuition
- Bigger lead -> win probability goes up
- Less time remaining -> current lead matters more
- Pregame favorite prior still matters, especially early

This is Bayesian in spirit even if implemented as logistic regression: you begin with a prior and let evidence (game state) update the belief.

---

## 5.10 Real time upset propagation into rankings and bracket odds

You specifically requested a real time effect where an underdog becoming the live favorite should update everything.

This requires two concepts:

### 5.10.1 Separate long term rating from live state strength
Do **not** fully rewrite season Elo mid game from one event.
Instead use two layers:

- **Base rating** $R_i^{base}$: slow moving team strength from season games
- **Live performance adjustment** $M_i(t)$: temporary in game state signal

Define a live effective rating:
$$
R_i^{live}(t) = R_i^{base} + M_i(t)
$$
A simple example:
$$
M_i(t) = \alpha_1 \cdot z(\text{score diff}_t) + \alpha_2 \cdot \text{late game leverage}_t + \alpha_3 \cdot \text{possession edge}_t
$$
This means:
- if the underdog is up late, their **live effective rating** can exceed the favorite temporarily,
- which can make them the live favorite for that game,
- which then changes downstream tournament simulations immediately.

### 5.10.2 Dynamic bracket recomputation
At each update tick (for example every possession, every 10 seconds, or every scoreboard event):
1. update live win probability for active games,
2. lock completed games,
3. condition the tournament state on current live probabilities,
4. rerun Monte Carlo simulation for remaining games,
5. publish updated:
   - championship odds,
   - Final Four odds,
   - projected bracket paths,
   - "most likely current bracket".

This is the exact "news shock repricing" behavior that makes the project look like a trading dashboard.

---

## 6. Data requirements (conceptual)

No code is requested here, but the research / product design should specify needed data.

### 6.1 Historical data (for ratings and model training)
Needed fields:
- date
- season
- team names / IDs
- opponent names / IDs
- final scores
- neutral site flag
- tournament flag
- home / away (if applicable pre tournament)
- overtime flag (optional)

### 6.2 Live game feed (for real time predictions)
Needed fields:
- current score
- game clock
- period / half
- possession (if available)
- fouls / bonus status (optional but useful)
- timeouts remaining (optional)
- game status (scheduled / live / final)

### 6.3 Bracket structure data
Needed fields:
- round
- game ID
- parent game IDs (for later rounds)
- seed
- region
- slot positions

---

## 7. Frontend design (portfolio quality deliverable)

You requested a frontend that displays all the data and updates the final winning bracket dynamically.

This is a major strength. Most student projects stop at a notebook or terminal output. A frontend turns it into a real product.

### 7.1 Frontend goals
- Make the math visible and understandable
- Make changes feel live and exciting
- Let a viewer understand "what changed and why"
- Show both the bracket and the probabilities behind it

### 7.2 Core screens

#### Screen A - Overview dashboard
Displays:
- Top power rankings
- Championship odds leaderboard
- Live games panel
- Biggest probability movers (up / down)
- Current projected champion
- Model confidence summary

#### Screen B - Bracket view (dynamic)
Displays:
- Tournament bracket tree
- Completed games locked in
- Live games highlighted
- Future games showing projected winners + probabilities
- "Most likely bracket path" and alternative scenarios

The bracket should update when:
- score changes,
- time changes,
- a live win probability flips,
- a game becomes final.

#### Screen C - Game detail page (live)
Displays:
- live win probability chart over time
- score timeline
- key events (lead changes, runs)
- pregame vs live probability
- "upset alert" indicator when underdog becomes favorite

#### Screen D - Team page
Displays:
- power rating
- rating history over season
- tournament odds by round
- upcoming matchup probability
- scenario table ("if they win this game, title odds become X")

#### Screen E - Calibration / model evaluation page
Displays:
- log loss
- Brier score
- reliability curve
- calibration bins
- maybe confusion style breakdown by favorite / underdog bins

This page signals real statistical maturity.

### 7.3 UX features that make it stand out
- Auto refresh live updates
- Probability change animations (+4.2%, -7.8%)
- Tooltips explaining math (Elo, log loss, Brier)
- Toggle between "most likely bracket" and "my bracket"
- Scenario mode:
  - "What if upset happens?"
  - "What if current live underdog wins?"

---

## 8. Methodology (research paper style workflow)

### Phase 1 - Build and validate ratings model
1. Implement Elo ratings on historical games
2. Generate pregame probabilities
3. Evaluate with log loss and Brier score
4. Tune $K$ and optional features (neutral court, recency weighting)

### Phase 2 - Build tournament simulator
1. Encode bracket structure
2. Simulate games from pregame probabilities
3. Estimate round advancement and title odds
4. Add bracket scoring objective functions

### Phase 3 - Build live win probability model
1. Define game state features
2. Train or specify live probability model
3. Test on historical play by play snapshots (if available)
4. Measure calibration of live probabilities

### Phase 4 - Propagate live updates through bracket engine
1. Inject live game state probabilities
2. Recompute tournament futures
3. Track probability swings caused by upsets
4. Publish change logs (before vs after)

### Phase 5 - Frontend and presentation
1. Build dashboard + bracket UI
2. Connect to model outputs
3. Add update cadence and animations
4. Prepare screenshots / demo video / writeup

---

## 9. Example live upset scenario (what you explicitly wanted)

Suppose:
- Favorite Team A starts at pregame win probability 78%
- Underdog Team B starts at 22%
- Midway through second half, Team B goes on a run and leads by 9
- The live model now estimates Team B at 64%

### What should happen immediately
1. **Live game card** flips Team B into favorite status.
2. **Bracket view** updates downstream winners in paths where this game feeds later rounds.
3. **Championship odds** update for both teams and their future opponents.
4. **Power rankings panel** shows:
   - base ranking unchanged (or mostly unchanged),
   - live effective ranking / momentum ranking temporarily updated.
5. **Probability movers** panel shows Team B as a major gainer.
6. **Most likely bracket** may change champion if this upset opens a path.

That is the "quant dashboard" moment. Very strong portfolio signal.

---

## 10. Model risks and limitations (important for mature writeup)

This section makes the project look serious and honest.

### 10.1 Elo is simple by design
Pros:
- interpretable
- fast
- strong baseline

Cons:
- limited feature richness
- no player level modeling
- no explicit injury effects unless added externally

### 10.2 Live model data quality risk
Live probabilities are only as good as the live feed:
- delayed scores
- missing possession
- inconsistent clocks
- broken events

### 10.3 Overfitting risk
A complex live model can look amazing on a small sample and fail out of sample.

Mitigation:
- train / validation split by season
- strict out of sample testing
- calibration checks

### 10.4 "Most likely bracket" is not "guaranteed bracket"
Very important to explain:
- the single most likely bracket path still has low probability,
- tournaments are high variance,
- uncertainty is the point, not a flaw.

That sentence alone is a good quant / finance signal.

---

## 11. Suggested deliverables for admissions / portfolio use

### Minimum strong deliverable (already impressive)
- Clean README research writeup
- Elo ratings and pregame probability model
- Tournament simulation with title odds
- Basic frontend bracket + rankings

### Advanced deliverable (excellent)
- Live win probability updates
- Dynamic bracket re simulation in real time
- Calibration page
- Scenario mode and pool strategy mode

### Presentation assets
- 2 minute demo video
- screenshots / gifs of live probability flips
- one page summary of math and findings
- link to repo and dashboard

---

## 12. Suggested README structure for the actual repository (future implementation)

```text
march-madness-quant/
|-- README.md
|-- docs/
|  |-- research-paper.md
|  |-- prd.md
|  |-- math-notes.md
|  `-- screenshots/
|-- data/
|  |-- raw/
|  |-- processed/
|  `-- bracket/
|-- notebooks/
|  |-- 01_elo_exploration.ipynb
|  |-- 02_model_eval.ipynb
|  |-- 03_bracket_simulation.ipynb
|  `-- 04_live_probability_analysis.ipynb
|-- src/
|  |-- ratings/
|  |-- predictions/
|  |-- live/
|  |-- bracket/
|  |-- api/
|  `-- utils/
|-- frontend/
|  |-- app/
|  |-- components/
|  `-- charts/
`-- tests/
```

---

## 13. What makes this especially good for "math + finance" branding

This project communicates:

- You can model uncertainty quantitatively.
- You understand probabilistic prediction vs deterministic guessing.
- You can evaluate forecasts using proper scoring rules.
- You can simulate future paths and compare strategies under different objectives.
- You can build a product interface, not just math in a notebook.
- You can explain the model to non technical people.

That is exactly the combo that plays well for applied math, economics, finance, data science, OR, stats, and quantitative social science.

---

## 14. Final summary

This project is not just "predicting basketball games."

It is a compact, high signal demonstration of:
- probability modeling,
- sequential updating,
- simulation,
- risk / reward tradeoffs,
- product thinking,
- and communication.

In finance language: it is a live uncertainty repricing and scenario engine with a visual dashboard.

That is a very strong story.

---

# PRD - Product Requirements Document (appendix at end of README)

## PRD 1. Product name
**March Madness Quant Dashboard**  
Subtitle: **Live Power Rankings, Win Probabilities, and Dynamic Bracket Projection**

---

## PRD 2. Product overview

A web based analytics dashboard that tracks NCAA tournament games and updates:

- team power rankings,
- live in game win probabilities,
- tournament advancement odds,
- projected bracket winners,
- and the current most likely final bracket path,

in near real time as games unfold.

The system is designed as both:
1. a learning / research project (math + probability + simulation), and
2. a portfolio quality product demonstration (frontend + live updating + explainable analytics).

---

## PRD 3. Goals and non goals

### Goals
- Provide interpretable power rankings using an Elo style model
- Show pregame and live win probabilities
- Recompute tournament futures as live games change
- Display a live updating bracket UI
- Surface probability movers and upset impacts
- Provide model evaluation metrics (log loss, Brier, calibration)

### Non goals (v1)
- Betting recommendations
- Real money wagering integration
- Player tracking / player level biomechanics
- Fully automated production grade data infrastructure
- Perfect prediction accuracy

---

## PRD 4. Target users

### Primary users
- Admissions reviewers / professors (portfolio audience)
- Students learning probability and data science
- Sports fans who want a quant view of tournament games
- Pool participants who want scenario analysis

### Secondary users
- Friends / peers evaluating bracket ideas
- Recruiters looking for analytics + product skills
- Finance / quant curious audiences who like uncertainty modeling examples

---

## PRD 5. Core user stories

### Rankings and predictions
- As a user, I want to see current team power rankings so I can understand team strength.
- As a user, I want to see pregame win probabilities so I can compare matchups.
- As a user, I want to see the projected champion and title odds leaderboard.

### Live game tracking
- As a user, I want live win probability updates during a game so I can see momentum and leverage shifts.
- As a user, I want an "underdog becomes favorite" alert so I can see meaningful upset moments.

### Bracket and scenario analysis
- As a user, I want the bracket to update as games progress and finish.
- As a user, I want future round probabilities to change in real time based on live outcomes.
- As a user, I want scenario mode ("if this upset happens") to see downstream effects.

### Model trust / transparency
- As a user, I want to see calibration and scoring metrics so I know whether the model is reliable.
- As a user, I want plain language explanations of Elo and probability so the dashboard is understandable.

---

## PRD 6. Functional requirements

### FR-1 Power rankings
- System shall compute and store team ratings from historical games.
- System shall display a sortable rankings table.
- System shall track rating changes over time (optional sparkline in v1.1).

### FR-2 Pregame matchup probabilities
- System shall generate pregame win probabilities from ratings.
- System shall display probabilities for scheduled and upcoming tournament games.
- System shall support neutral court assumption for tournament games.

### FR-3 Live win probability engine
- System shall ingest live game state updates (score, clock, status; possession if available).
- System shall update win probability at a configured cadence.
- System shall expose current probability and probability delta since prior update.

### FR-4 Upset propagation and dynamic futures
- System shall detect when the pregame underdog becomes the live favorite.
- System shall rerun remaining tournament simulations after material game state updates.
- System shall update championship odds, Final Four odds, and projected bracket path.

### FR-5 Dynamic bracket UI
- System shall render a bracket with completed, live, and projected games.
- System shall visually distinguish:
  - completed outcomes,
  - live games,
  - projected future games.
- System shall update projected winners and odds automatically.

### FR-6 Game detail page
- System shall display:
  - score,
  - clock,
  - live probability,
  - pregame probability,
  - probability timeline chart.
- System shall display notable events / lead changes if event data exists.

### FR-7 Model evaluation page
- System shall display log loss and Brier score for pregame predictions.
- System shall display calibration / reliability chart.
- System shall allow filtering by favorite probability buckets (optional v1.1).

### FR-8 Scenario mode (nice to have in v1, core in v2)
- System shall allow manual override of a live game outcome or probability.
- System shall recompute tournament futures under that scenario.
- System shall show before vs after deltas.

---

## PRD 7. Non functional requirements

### Performance
- Live update latency target (UI visible): under 2-5 seconds after new game state event
- Bracket futures recomputation target: under 3 seconds for moderate simulation count in v1
- Frontend interactions should feel smooth and readable on laptop screens

### Reliability
- System shall gracefully handle missing live fields (for example possession unavailable)
- System shall mark stale live data and timestamp last update
- System shall fail safely (show previous probabilities with stale warning)

### Explainability
- System shall provide tooltips / help text for:
  - Elo rating
  - win probability
  - log loss
  - Brier score
  - calibration

### Accessibility / UX
- Readable typography
- Color + label cues (not color only)
- Responsive layout for desktop first, tablet acceptable in v1

---

## PRD 8. Data model (conceptual entities)

### Entity: Team
- team_id
- team_name
- seed
- region
- base_rating
- live_effective_rating (derived, transient)
- title_odds
- final_four_odds
- elite_eight_odds
- sweet_sixteen_odds

### Entity: Game
- game_id
- round
- team_a_id
- team_b_id
- status (scheduled/live/final)
- pregame_prob_a
- live_prob_a
- score_a
- score_b
- clock
- possession
- winner_id (if final)

### Entity: BracketNode
- node_id
- round
- left_parent_game_id
- right_parent_game_id
- projected_winner_id
- projected_win_prob
- final_winner_id (if known)

### Entity: SimulationSnapshot
- timestamp
- simulation_count
- championship_odds_by_team
- projected_bracket_path
- model_version

---

## PRD 9. UX / frontend requirements (detailed)

### Main dashboard layout (desktop)
- Top bar:
  - tournament status
  - last updated timestamp
  - model version
- Left column:
  - live games list
  - biggest movers
- Center:
  - dynamic bracket
- Right column:
  - power rankings
  - championship odds
  - projected champion card

### Visual states for games
- Scheduled: muted / pending
- Live: highlighted with pulsing indicator
- Final: locked and checkmarked
- Upset in progress: badge "underdog live favorite"

### Required charts
- Live probability line chart (single game)
- Odds change bar chart (team delta)
- Calibration chart (reliability plot)
- Rating history line chart (optional v1.1)

### Interactions
- Click a game -> open game detail pane/page
- Hover a bracket node -> show probability tooltip and matchup assumptions
- Toggle:
  - "Most likely bracket"
  - "Highest expected score bracket" (if implemented)
  - "Scenario mode"

---

## PRD 10. Analytics and success metrics

### Product / UX metrics (portfolio demo quality)
- Time to first meaningful dashboard render
- Live update freshness (seconds)
- Number of successful live probability refreshes
- Number of scenario recomputations completed

### Model metrics
- Pregame log loss
- Pregame Brier score
- Calibration by bins
- Live probability calibration (future phase)
- Simulation stability (variance across runs at same state)

### Portfolio success criteria (qualitative)
- A viewer can understand the project story in under 2 minutes
- A viewer can see live probabilities and bracket updates
- A viewer can tell this is math + product, not just sports fandom

---

## PRD 11. Milestones (implementation roadmap)

### Milestone 1 - Baseline math engine (1-2 weeks)
- Elo ratings
- Pregame probabilities
- Evaluation metrics (log loss, Brier)
- Static output tables

### Milestone 2 - Tournament simulator (1-2 weeks)
- Bracket structure
- Monte Carlo simulations
- Title odds and round advancement odds
- Static "most likely bracket" projection

### Milestone 3 - Live probability engine (2-3 weeks)
- Live game state ingestion
- Live win probability updates
- Probability timeline tracking
- Upset detection rule

### Milestone 4 - Dynamic futures recomputation (1-2 weeks)
- Recompute tournament odds on live updates
- Delta views (before/after upset)
- Snapshot storage

### Milestone 5 - Frontend dashboard (2-4 weeks)
- Bracket UI
- rankings and odds tables
- live game cards
- charts and tooltips
- polish and demo prep

### Milestone 6 - Final writeup and presentation (1 week)
- README cleanup
- research summary
- screenshots / GIFs
- short demo video

---

## PRD 12. Risks and mitigations

### Risk: live data source instability
Mitigation:
- cache latest valid state
- stale data warnings
- fallback to manual scenario entry for demos

### Risk: simulation too slow during live updates
Mitigation:
- reduce simulation count dynamically during live play
- precompute some branches
- parallelize later (v2)

### Risk: misleading confidence
Mitigation:
- display calibration page
- label probabilities as estimates
- surface uncertainty and model limitations clearly

### Risk: project scope explosion
Mitigation:
- lock v1 scope:
  - Elo + pregame probabilities
  - live win probability
  - dynamic bracket updates
  - frontend core screens only

---

## PRD 13. V1 acceptance criteria (exact)

A v1 release is considered complete when all of the following are true:

1. **Power Rankings**
   - Rankings table displays all tournament teams with Elo based ratings.

2. **Pregame Predictions**
   - Every scheduled tournament game has a pregame win probability.

3. **Live Predictions**
   - For at least one live game, the dashboard updates win probability as score/clock changes.

4. **Upset Propagation**
   - When the pregame underdog becomes the live favorite, championship odds and projected bracket update within the target refresh window.

5. **Dynamic Bracket**
   - Bracket UI visually updates completed and projected winners.

6. **Model Evaluation**
   - Log loss and Brier score are displayed for pregame forecasts.

7. **Frontend Quality**
   - A reviewer can navigate dashboard -> game detail -> bracket -> evaluation page without confusion.

8. **Explainability**
   - Core math terms (Elo, win probability, log loss, Brier) have plain language explanations.

---

## PRD 14. Future extensions (v2+)

- Player level adjustments and injury inputs
- Bayesian hierarchical team strength model
- Possession based live model
- Pool strategy optimizer against opponent brackets
- "What changed?" AI generated textual summaries
- Historical tournament replay mode
- Portfolio mode showing methodology + results side by side

---

## Closing note

This project is a rare combo:
- mathematically rigorous enough to discuss probability, calibration, and simulation,
- product oriented enough to show frontend and systems thinking,
- and exciting enough that people actually want to watch it.

That makes it an excellent flagship project for a math + finance identity.

## REOPEN: March Madness -> Quant Signaling & User Created Power Rankings, Brackets, etc. 25 Febuary
- Elo system (x)
- Monte carlo -> predictions (x)
- Live updates (x)
- Fake parlays (x)
- Graphs for info (x)
- Efficiency (offensive & defensive) ( )
- Treating CB teams as stock
- Replicated for CB Players
- Treating CB players as stock
- Repo replicated for NBA ( )
- Replicated for individual teams & players, and treating both as stock ( )
- Transfer idea to alpaca-py ( )
- Mini paper written ( )
- Presented for peer review from professors, calc & UT math ( )
- NEED TO BE FINISHED BY SUNDAY.
- present at math for all UT austin
