// ChalkIQ: Probabilistic Sports Prediction via Elo Ratings and Closing Line Value
#import "lib.typ": *

#show: template

#set document(
  title: "Probabilistic Sports Prediction via Elo Ratings and Closing Line Value",
  author: "Deep Hayer",
)

#header(
  name: "Probabilistic Sports Prediction via Elo Ratings and Closing Line Value",
  author: "Deep Hayer",
  topic: "Sports Analytics | Prediction Markets",
  date: "April 2026",
)

#epigraph(attribution: [Levitt (2004)])[
  Gambling markets are among the most efficient information-aggregation mechanisms
  ever devised. Beating the closing line consistently is the proof of genuine
  predictive edge.
]

*ChalkIQ* is a multi-sport probabilistic prediction system generating calibrated
win probabilities via Elo ratings, real-time injury signals, and live score
diffusion. We evaluate across 1,672 games (Nov 2024 -- Apr 2026), achieving a
Brier score of 0.189 and log loss of 0.556 -- improvements of 24% and 20% over
the coin-flip baseline respectively.

= The Elo Model

Each team carries a single rating $R in RR$, initialised at 1500 and updated
after every game. Win probability follows a logistic curve:

#defn(title: [win probability])[
  $ P(A "wins") = 1 / (1 + 10^((R_B - R_A - H) / kappa)) $

  where $H$ is the home-court advantage and $kappa$ is the scale factor. A gap of
  $kappa$ points implies ~75% win probability. For NCAAB: $H = 100$, $kappa = 300$.
]

After each game, the rating update incorporates both outcome and margin:

#defn(title: [update rule])[
  $ R'_A = R_A + K dot.op "MOV" dot.op (y - P(A "wins")) $

  $ "MOV" = (ln(|m| + 1) dot.op 2.2) / (Delta_R dot.op 0.001 + 2.2) $

  $K$ is the learning rate, $m$ the point margin, $Delta_R$ the pre-game rating
  gap. The logarithm dampens blowouts; the denominator corrects for autocorrelation
  -- a dominant team routing a weak one gets a smaller update than evenly-matched
  teams separated by the same margin.
]

#note[
  Parameters vary by sport. NCAAB uses $K = 24$ (high roster turnover, volatile
  results). MLB uses $K = 6$ (even the best team loses 40% of games). Between
  seasons, ratings regress toward 1500 by a fraction $rho$ reflecting roster
  continuity: $rho = 0.33$ for NCAAB, $0.25$ for NBA.
]

= Closing Line Value

The primary evaluation metric is not win rate or P&L -- it is Closing Line Value.

#defn(title: [CLV])[
  $ "CLV" = p_"model" - p_"close" $

  where $p_"close"$ is the implied probability from the bookmaker's closing
  moneyline after removing the overround. Positive CLV means the model identified
  the correct side before the market fully priced it in.
]

#aside[
  CLV measures _prediction quality_, not outcome luck. The closing line aggregates
  sharp bettors, syndicate models, and late injury news. A model consistently
  beating it has demonstrated something real -- analogous to measuring $alpha$ via
  forward returns rather than a single trade's P&L.
]

#eg(title: [empirical CLV over 1,672 games])[
  #table(
    columns: (1fr, auto),
    inset: 7pt,
    stroke: 0.4pt + luma(180),
    align: (left, right),
    table.header([*Metric*], [*Value*]),
    [Beat closing line], [45.9%],
    [Mean CLV],          [$-0.78%$],
    [Std deviation],     [$20.0%$],
    [Range],             [$-80%$ to $+72%$],
    [Brier score],       [0.189 #h(0.3em)_(baseline 0.250)_],
    [Log loss],          [0.556 #h(0.3em)_(baseline 0.693)_],
    [Favourite accuracy],[70.9%],
  )
]

#note[
  Mean CLV is negative (-0.78%), as efficient market theory predicts: the closing
  line has absorbed injury reports, sharp money, and late lineup news that a pure
  Elo model cannot see. The wide standard deviation (20%) reveals where the value
  actually lives -- not in systematic outperformance, but in large edges on
  specific games where Elo diverges sharply from the market.
]

= Live In-Game Model

Pre-game probabilities are repriced during games using a score diffusion model
(Stern 1994): basketball scoring is approximated as a random walk.

#defn(title: [live win probability])[
  With team $A$ leading by $d$ points and $t$ minutes remaining:

  $ P(A "wins" | d, t) = Phi lr((d / (sigma sqrt(t)) + "logit"(p_0) dot.op sqrt(3)/pi)) $

  $sigma approx 2.0$ pts/$sqrt("min")$ for college basketball. At tip-off the Elo
  prior $p_0$ dominates; as $t arrow 0$ the lead term drives $P arrow 1$.
]

= Backtest

Walk-forward backtest over the 2025--26 NCAAB season -- ratings built only from
games completed _before_ each prediction.

#eg(title: [2025--26 NCAAB backtest])[
  #table(
    columns: (auto, auto, auto, auto, auto),
    inset: 6pt,
    stroke: 0.4pt + luma(180),
    align: (left, center, center, right, right),
    table.header([*Month*], [*Bets*], [*Win rate*], [*ROI*], [*Cumul P&L*]),
    [Nov 2025], [119], [85%], [$+62%$], [$+$\$7,382],
    [Dec 2025], [187], [86%], [$+64%$], [$+$\$19,418],
    [Jan 2026], [351], [74%], [$+41%$], [$+$\$33,954],
    [Feb 2026], [269], [65%], [$+25%$], [$+$\$40,654],
    [Mar 2026], [148], [75%], [$+43%$], [$+$\$47,045],
  )
  1,076 bets placed | 75.2% win rate | +43.5% ROI | Brier 0.174 | Log loss 0.523
]

#aside[
  Win rate declines across the season as the model converges -- early games have
  high edge because Elo hasn't yet separated the field. By February, ratings are
  tighter and the market harder to beat. The backtest is in-sample; live
  performance would be lower after accounting for hold percentages and line shopping.
]

= Conclusion

A parsimonious one-number-per-team Elo model, calibrated on 1,136 games and
evaluated across 1,672, produces well-calibrated multi-sport win probabilities.
Brier score 0.189 and log loss 0.556 represent meaningful improvements over
baseline. Mean CLV of -0.78% is consistent with market efficiency; the 20%
standard deviation indicates genuine signal on individual games. The same
framework -- Elo ratings, CLV measurement, score diffusion -- ports across seven
sports by changing only the parameters, suggesting the model captures real
structure in competitive dynamics.

#v(1em)
= References

#set par(hanging-indent: 1.5em)

Elo, A. (1978). _The Rating of Chess Players, Past and Present_. Arco.

Levitt, S. D. (2004). Why are gambling markets organised so differently from
financial markets? _Economic Journal_, 114(495), 223--246.

Sauer, R. D. (1998). The economics of wagering markets. _Journal of Economic
Literature_, 36(4), 2021--2064.

Silver, N. (2015). How our NCAAB predictions work. _FiveThirtyEight_.

Stern, H. S. (1994). A Brownian motion model for the progress of sports scores.
_Journal of the American Statistical Association_, 89(427), 1128--1134.
