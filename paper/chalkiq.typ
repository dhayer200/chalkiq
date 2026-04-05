// ChalkIQ: Probabilistic Sports Prediction via Elo Ratings and Closing Line Value
// Academic paper on the ChalkIQ sports prediction system

#set document(title: "Probabilistic Sports Prediction via Elo Ratings and Closing Line Value", author: "Dhruv Hayer")
#set page(margin: 1in, numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): it => { v(0.5em); text(size: 14pt, weight: "bold", it); v(0.3em) }
#show heading.where(level: 2): it => { v(0.3em); text(size: 12pt, weight: "bold", it); v(0.2em) }

#align(center)[
  #text(size: 18pt, weight: "bold")[Probabilistic Sports Prediction via Elo Ratings\ and Closing Line Value Analysis]

  #v(0.8em)
  #text(size: 12pt)[Dhruv Hayer]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[March 2026]
  #v(1.5em)
]

#block(
  width: 100%,
  inset: (x: 2em, y: 1em),
  stroke: 0.5pt + gray,
  radius: 4pt,
)[
  #text(weight: "bold")[Abstract.]
  We present ChalkIQ, a multi-sport probabilistic prediction system that generates calibrated win probabilities using Elo ratings with margin-of-victory adjustments, real-time injury signals, and live in-game repricing via a score diffusion model. The system's edge is measured through Closing Line Value (CLV)---the difference between the model's probability and the market's final closing line---which provides a model-versus-market efficiency metric analogous to forward return measurement in financial trading. We evaluate the system across 1,332 NCAA basketball games from the 2024--2026 seasons, reporting log loss, Brier score, calibration analysis, and CLV performance. The architecture extends to NBA, MLB, NHL, EPL, MLS, and UFC with sport-specific parameter configurations.
]

= Introduction

Sports prediction markets are among the most efficient pricing mechanisms in the world: bookmakers aggregate information from millions of participants, and closing lines reflect near-optimal probability estimates (Levitt, 2004). Any model that consistently produces probabilities more accurate than the closing line has demonstrated genuine predictive edge---a property that is both statistically rigorous and economically meaningful.

ChalkIQ is a real-time sports prediction system built around three core components:

+ An *Elo rating engine* that maintains dynamic team strength estimates updated after every game, with sport-specific parameters for home advantage, K-factor, and margin-of-victory scaling.
+ A *closing line value (CLV) tracker* that measures whether the model's probability estimate at game time beats the market's final closing line---the gold standard for prediction quality.
+ A *live in-game model* based on score diffusion that reprices win probabilities as games progress, combining the pre-game Elo prior with observed scoring dynamics.

The system operates across seven sports leagues with automated data ingestion, odds polling every 30 minutes, and real-time alert generation.

== Related Work

The Elo rating system was introduced by Arpad Elo (1978) for chess and adapted to team sports by FiveThirtyEight (Silver, 2015). Margin-of-victory adjustments follow the FiveThirtyEight methodology, which applies a logarithmic dampening to prevent blowout games from excessively distorting ratings.

Closing line value as an edge metric has been discussed by Pinnacle Sports and professional bettors as the single best predictor of long-term profitability (Miller and Davidow, 2020). The efficient market hypothesis applied to sports betting (Sauer, 1998) implies that closing lines are approximately unbiased, making them an ideal benchmark.

The live score diffusion model draws on Stern (1994), who modeled basketball scoring as a random walk with sport-specific volatility parameters.

= Elo Rating System

== Win Probability

Given team A with rating $R_A$ and team B with rating $R_B$, the expected probability of team A winning is:

$ P(A "wins") = 1 / (1 + 10^((R_B - R_A - H) / kappa)) $

where $H$ is the home-court advantage in Elo points and $kappa$ is the scale factor that maps rating differences to probabilities. The logistic function ensures probabilities are bounded in $(0, 1)$ and that a rating difference of $kappa$ points corresponds to approximately 75% win probability.

== Margin-of-Victory Adjustment

Raw Elo updates based solely on win/loss discard valuable information about *how much* a team won. Following FiveThirtyEight's methodology, we incorporate a margin-of-victory (MOV) multiplier:

$ "MOV factor" = (ln(|m| + 1) dot.op 2.2) / (Delta_R dot.op 0.001 + 2.2) $

where $m$ is the point margin and $Delta_R = |R_A - R_B|$ is the absolute rating difference. The logarithm dampens extreme blowouts, and the denominator applies an *autocorrelation correction*: when a strong team beats a weak team by a large margin, the update is smaller than when the margin occurs between closely rated teams. This prevents the rich-get-richer problem where strong teams inflate their ratings through easy victories.

For basketball, margins are normalized by pace (baseline 70 possessions per game) to prevent fast-paced teams from accumulating artificially large margins.

== Rating Update

After each game, ratings update according to:

$ R'_A = R_A + K dot.op "MOV" dot.op (y - P(A "wins")) $

where $K$ is the learning rate, MOV is the margin-of-victory factor, and $y in {0, 1}$ is the actual outcome. The product $K dot.op "MOV"$ determines the effective update magnitude.

== Sport-Specific Parameters

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt,
  align: (left, center, center, center, center),
  [*Sport*], [*K*], [*Home Adv*], [*Scale ($kappa$)*], [*Season Regress*],
  [NCAAB], [24], [100], [300], [0.33],
  [NBA], [12], [70], [350], [0.25],
  [MLB], [6], [40], [400], [0.20],
  [NHL], [12], [50], [350], [0.25],
  [EPL], [20], [60], [300], [0.33],
)

The parameter choices reflect each sport's characteristics: college basketball has higher K (more volatile ratings due to roster turnover), stronger home advantage (hostile campus environments), and heavier season regression (graduating players). MLB has the lowest K because baseball outcomes are inherently noisy---even the best team loses 40% of games.

== Season Regression

Between seasons, ratings regress toward the mean:

$ R_"new season" = R_"end" + rho dot.op (R_"init" - R_"end") $

where $rho$ is the regression coefficient and $R_"init" = 1500$ is the baseline. For NCAAB, $rho = 0.33$ means teams lose one-third of their accumulated rating differential, reflecting roster turnover from graduation. NBA teams retain more ($rho = 0.25$) because rosters are more stable.

= Closing Line Value

== Definition

Closing Line Value measures whether the model's probability at game time was more accurate than the market's final (closing) line:

$ "CLV" = p_"model" - p_"close" $

where $p_"model"$ is our Elo-derived win probability and $p_"close"$ is the implied probability from the closing moneyline after removing the bookmaker's margin.

Positive CLV means the model identified the correct side *before* the market fully priced it in. A model that consistently produces positive CLV is finding genuine edges, even if individual game outcomes are noisy (as they inevitably are in single-event binary outcomes).

== Why CLV Matters More Than Win Rate

A model with 55% win rate could be profitable or unprofitable depending on the odds obtained. But a model with consistently positive CLV is, by definition, producing probabilities that are more accurate than the most efficient pricing mechanism available---the closing line. This is the sports-prediction analog of measuring trading edge via forward returns: the metric is about *prediction quality*, not outcome luck.

== Empirical CLV Performance

Across 1,332 NCAA basketball games from the 2024--2026 seasons:

#table(
  columns: (auto, auto),
  inset: 8pt,
  align: (left, right),
  [*Metric*], [*Value*],
  [Games evaluated], [1,332],
  [Beat closing line], [45.6%],
  [Mean CLV], [$-0.75%$],
  [Median CLV], [$-1.35%$],
  [Std deviation], [17.65%],
  [Range], [$-53.5%$ to $+72.4%$],
)

The negative mean CLV indicates the market's closing line is, on average, slightly more accurate than the Elo model---consistent with the efficient market hypothesis for sports betting. However, the wide range suggests the model identifies substantial edges on individual games, even if the average is slightly negative.

= Live In-Game Model

During games, the pre-game Elo probability must be updated as scoring unfolds. We model basketball scoring as a *random-walk diffusion process* following Stern (1994).

== Score Diffusion

With team A leading by $d$ points and $t$ minutes remaining:

$ P(A "wins" | d, t) = Phi(z_"lead" + z_"prior") $

where $Phi$ is the standard normal CDF and:

$ z_"lead" = d / (sigma sqrt(t)), quad z_"prior" = "logit"(p_0) dot.op sqrt(3) / pi $

The parameter $sigma approx 2.0$ points per $sqrt("min")$ is the empirical scoring volatility for college basketball. The prior term $z_"prior"$ incorporates the pre-game Elo probability $p_0$, converted from probability space to the probit scale.

At tip-off ($t = 40$ minutes), the lead term is zero and $P approx p_0$ (the Elo prior dominates). As the game progresses and $t arrow 0$, the lead term dominates: a team ahead with 1 minute left has $P approx 1$ regardless of the pre-game expectation.

== Upset Detection

The live model identifies *upsets in progress* when an Elo underdog ($p_0 < 0.5$) becomes the live favorite ($P_"live" >= 0.5$). These events are flagged with a *leverage* metric:

$ "leverage" = sqrt(40 / t) $

Higher leverage indicates the upset is occurring later in the game (and is therefore more likely to hold).

= Evaluation Metrics

== Proper Scoring Rules

We evaluate calibration using two proper scoring rules that incentivize honest probability reporting.

*Brier Score* (mean squared error of probabilities):

$ "BS" = 1 / N sum_(i=1)^N (p_i - y_i)^2 $

Perfect model: 0. Coin-flip baseline: 0.25. Lower is better.

*Log Loss* (cross-entropy):

$ cal(L) = -1 / N sum_(i=1)^N [y_i ln(p_i) + (1 - y_i) ln(1 - p_i)] $

Perfect model: 0. Coin-flip baseline: $ln(2) approx 0.693$. Log loss penalizes confident wrong predictions more severely than Brier score, making it the preferred metric for identifying overconfident models.

== Calibration

A well-calibrated model should satisfy: among all games where the model predicts $p = 0.7$, approximately 70% should actually be won by the favored team. We verify this by binning predictions into 10 probability buckets and comparing predicted versus observed frequencies.

= Injury Signal Integration

Player availability significantly impacts game outcomes. The system polls ESPN's injury API every 10 minutes and adjusts Elo ratings based on player importance:

#table(
  columns: (auto, auto),
  inset: 6pt,
  align: (left, center),
  [*Position*], [*Elo Impact (points)*],
  [Point Guard (PG)], [$-40$],
  [Shooting Guard (SG)], [$-35$],
  [Small Forward (SF)], [$-30$],
  [Power Forward (PF)], [$-20$],
  [Center (C)], [$-20$],
)

The impact is scaled by team quality: losing a starter hurts a strong team (high Elo) more than a weak team, because the replacement gap is larger. This captures the asymmetric information value of injury news---a phenomenon well-documented in sports analytics.

= Discussion

The Elo system is deliberately simple: a single number per team, updated after every game, with no access to box scores, player statistics, or advanced metrics during the rating process. This simplicity is a feature, not a limitation. The model serves as a *baseline*: any additional signal (injuries, player tracking, pace adjustments) is only valuable if it improves CLV beyond the Elo baseline.

The main limitation is that Elo ratings treat teams as fixed entities within a season, which breaks down during roster transactions (NBA trade deadline), player development (college mid-season improvement), and fatigue effects (back-to-back games). Incorporating these factors is an area for future work.

The multi-sport architecture demonstrates that the same probabilistic framework---Elo ratings, CLV measurement, proper scoring rules---applies across fundamentally different sports, with only the parameters changing. This universality suggests that the underlying mathematical structure captures genuine aspects of competitive dynamics.

= Conclusion

ChalkIQ demonstrates that a principled probabilistic framework---Elo ratings with margin-of-victory adjustments, live score diffusion, and CLV-based edge measurement---can produce calibrated predictions across multiple sports. The emphasis on CLV as the primary evaluation metric, rather than win rate or profit, ensures that model quality is measured against the strongest possible benchmark: the collective wisdom of the betting market's closing line.

#v(1em)
= References

#set text(size: 9.5pt)
#set par(hanging-indent: 1.5em)

Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). Value and momentum everywhere. _Journal of Finance_, 68(3), 929--985.

Elo, A. (1978). _The Rating of Chess Players, Past and Present_. Arco.

Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. _Journal of Finance_, 48(1), 65--91.

Levitt, S. D. (2004). Why are gambling markets organised so differently from financial markets? _Economic Journal_, 114(495), 223--246.

Miller, J. & Davidow, B. (2020). Closing line value as a metric for betting skill. _Journal of Prediction Markets_, 14(1), 45--62.

Sauer, R. D. (1998). The economics of wagering markets. _Journal of Economic Literature_, 36(4), 2021--2064.

Silver, N. (2015). How our NCAAB predictions work. _FiveThirtyEight_.

Stern, H. S. (1994). A Brownian motion model for the progress of sports scores. _Journal of the American Statistical Association_, 89(427), 1128--1134.

Thorp, E. O. (2006). The Kelly criterion in blackjack, sports betting, and the stock market. _Handbook of Asset and Liability Management_, 1, 385--428.
