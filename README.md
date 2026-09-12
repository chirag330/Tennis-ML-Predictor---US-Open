# Tennis-ML-Predictor---US-Open
This project leverages data science and machine learning to build a prediction model for the 2026 US Open happening in New York. This project would employ models like logistic regression and XGBoost to determine the win probability of the player, resulting in a statistical approach for placing tennis bets.

## Datasets
- I am using the publicly available datasets published by ATP from 2021 - 2026. These datasets are thorough, with detailed information about every match, including the surface types, odds for both players, and the break points.
- The link for the datasets is: https://tennisdata.app/downloads/
- I am using all 6 available datasets from 2021 - 2026
- The initial plan is to use the 2021-2024 datasets for training, the 2025 dataset to test the model, and the real 2026 data to verify and to place bets.
- The plan for split is TRAIN: 2021-2024; VALIDATION: 2025 and the 2026 is the data is the one to be predicted so that would remain untouched.

## Baseline Model
- The first primitive model that was fit for the predication employed the assumption that the winning player had a higher ranking than their opponent. 
- A new column was created that had the binary information whether the home player won or not.
- Then, another column was created which was the binary information indicating wheter rank(home_player) < rank(away_player)
- Then, the accuracy score was calculated using the formula: number of correct predications / total number of predications
- The accuracy was about 62.54%

## Logistic Regression V1
- The next model I implemented was logistic regression.
- This was the first version and the covariates were difference in ranking and the difference in the points between the home and away players.
- The Y variable was the probability that the home player wins.
- The training data was the information from the years 2021-2024 and the validation data was 2025.
- For implementing this model, sklearn was used, particularly the sklearn.linear_model and LogisticRegression
- After fitting the model, the coefficients were calculated and the predictions and prediction probabilities were compared against the actual values.
- Using sklearn.metrics, the accuracy score was calculated which turned out to be 62.72% which was only significantly better than the baseline model.

## Logistic Regression V2
| Category  | Historical feature                    | Derived from          |
| --------- | ------------------------------------- | --------------------- |
| Baseline  | `rank_diff`                           | rank                  |
| Baseline  | `points_diff`                         | ranking points        |
| Results   | `win_rate_last_10_diff`               | `winner_code`         |
| Dominance | `games_won_pct_last_10_diff`          | individual set scores |
| Dominance | `avg_game_diff_last_10_diff`          | individual set scores |
| Serving   | `avg_aces_last_10_diff`               | aces                  |
| Serving   | `avg_double_faults_last_10_diff`      | double faults         |
| Serving   | `avg_service_points_won_last_10_diff` | service points won %  |
| Returning | `avg_return_points_won_last_10_diff`  | return points won %   |
| Pressure  | `avg_break_points_won_last_10_diff`   | break points won %    |
| Pressure  | `avg_break_points_saved_last_10_diff` | break points saved %  |
### V2 — Historical Player Performance Features


V2 extends the initial ranking-based logistic regression model by incorporating historical player performance. The goal was to test whether recent form, surface-specific performance, and match statistics provide additional predictive information beyond ATP ranking and ranking points.

To prevent data leakage, match statistics were converted into **lagged rolling features**. For each player and match, performance metrics were calculated using only matches that occurred before the current match. Most recent-form statistics use a rolling window of the player's previous 10 matches.

#### Features

V2 uses 11 predictors:

- ATP rank difference
- ATP ranking points difference
- Win rate over the previous 10 matches
- Percentage of games won over the previous 10 matches
- Historical hard-court win rate
- Average aces over the previous 10 matches
- Average double faults over the previous 10 matches
- Average service points won over the previous 10 matches
- Average return points won over the previous 10 matches
- Average break points won over the previous 10 matches
- Average break points saved over the previous 10 matches

Player-level statistics were converted into home-vs-away differences before being passed to the model.

`sets_won_pct_last_10` was also engineered but excluded from the final V2 feature set because it was highly correlated with recent win rate (r = 0.945) and games-won percentage (r = 0.901).

#### Data Availability

After requiring all V2 predictors to be available:

- Total eligible completed matches: **74,304**
- Matches available for V2: **47,017**
- Data retained: **63.3%**

ATP ranking and ranking points were the primary source of missing data, with approximately 32% missingness.

#### Results

| Model | Accuracy | Log Loss |
|---|---:|---:|
| V1 — Rank + Points | 62.72% | 0.6467 |
| V2 — Rank + Historical Performance | **63.32%** | **0.6407** |

V2 produced a modest improvement in both classification accuracy and probability quality. This suggests that recent player performance contains useful predictive information beyond ATP ranking alone.

However, the improvement was relatively small, indicating that simple rolling statistics still fail to capture important aspects of player strength — particularly the **quality of opponents faced**. For example, an 80% recent win rate against lower-level opposition is treated similarly to an 80% win rate against elite ATP players.

This motivates V3, which introduces a chronological **Elo rating system** to estimate player strength based not only on wins and losses, but also on the strength of the opponents involved.

## Model V3 — Adding Elo Player Strength

Model V3 extends the previous logistic regression model by introducing an
**Elo rating system** to represent player strength.

While ATP rankings, ranking points, and recent win rates provide useful
information, they do not directly account for the strength of the opponents
a player has faced. Elo addresses this by updating player ratings based on
both the result of a match and the strength of the opponent.

### Historical Elo Construction

Rather than using current Elo ratings from an external source, historical
Elo ratings were calculated directly from the match dataset.

All completed matches were processed chronologically. Each player began with
an Elo rating of **1500**, and ratings were updated after every match using:

Expected win probability:

\[
P(A) = \frac{1}{1 + 10^{(R_B-R_A)/400}}
\]

Rating update:

\[
R_{new} = R_{old} + K(S-P)
\]

where:

- `R_old` is the player's rating before the match
- `P` is the expected probability of winning
- `S` is the actual result (1 for a win, 0 for a loss)
- `K = 32` controls how quickly ratings respond to new results

A player's **pre-match Elo rating** is stored before processing the result of
that match. This is important because it prevents future information from
leaking into the prediction.

The primary Elo feature used by the model is:

`elo_diff = home_elo - away_elo`

A positive value therefore represents an Elo advantage for the home player.

### Elo Validation

Before adding Elo to the full model, Elo was evaluated independently on the
2025 validation season.

| Model | Accuracy | Log Loss |
|---|---:|---:|
| Elo only | 62.83% | 0.6416 |

The relationship between Elo difference and match outcomes was also examined.
As the Elo gap increased, the higher-rated player's observed win rate increased
consistently:

| Absolute Elo Difference | Elo Favorite Win Rate |
|---|---:|
| 0–50 | 53.6% |
| 50–100 | 61.1% |
| 100–150 | 66.9% |
| 150–200 | 73.4% |
| 200–300 | 76.5% |
| 300–500 | 87.0% |
| 500+ | 97.1% |

This monotonic relationship provided a useful sanity check that the
chronological Elo implementation was capturing meaningful differences in
player strength.

### Surface-Specific Elo Experiment

A separate hard-court Elo rating was also tested. Unlike overall Elo, this
rating was updated only after hard-court matches.

On the same 7,297 hard-court matches from the 2025 validation season:

| Elo Model | Accuracy | Log Loss |
|---|---:|---:|
| Overall Elo | 62.83% | 0.6418 |
| Hard-court Elo | 61.09% | 0.6524 |

Overall Elo performed better than hard-court Elo on both metrics. The
surface-specific rating likely suffers from having substantially fewer
historical observations per player.

For this reason, **overall Elo was retained for V3**, while hard-court Elo
was excluded from the current model.

### V3 Features

V3 uses 12 predictors:

- `elo_diff`
- `rank_diff`
- `points_diff`
- `win_rate_last_10_diff`
- `games_won_pct_last_10_diff`
- `hard_win_rate_diff`
- `avg_aces_last_10_diff`
- `avg_double_faults_last_10_diff`
- `avg_service_points_won_last_10_diff`
- `avg_return_points_won_last_10_diff`
- `avg_break_points_won_last_10_diff`
- `avg_break_points_saved_last_10_diff`

All rolling performance statistics continue to use only matches occurring
before the match being predicted.

### Model Training

The chronological split remained:

- **Training:** 2021–2024
- **Validation:** 2025
- **Held-out test:** 2026

After removing observations with missing values across the required V3
features:

- Training matches: **28,470**
- Validation matches: **8,677**
- 2026 held-out matches: **9,870**
- Total usable matches: **47,017**

The 2026 data remains untouched for final evaluation.

V3 uses a pipeline consisting of:

1. `StandardScaler`
2. `LogisticRegression`

Feature scaling was introduced because predictors such as ranking points,
Elo differences, and percentage-based statistics exist on very different
numerical scales. Scaling also makes the regularization applied by logistic
regression more appropriate across features.

### V3 Results

Performance on the 2025 validation set:

| Model | Accuracy | Log Loss |
|---|---:|---:|
| V1 — Ranking + Points | 62.72% | 0.6467 |
| V2 — Ranking + Historical Performance | 63.32% | 0.6407 |
| V3 — V2 Features + Elo | **64.30%** | **0.6330** |

V3 produced the best validation performance so far.

Compared with V2, adding Elo increased accuracy by approximately **1 percentage
point** while also reducing log loss. The improvement in log loss is
particularly important because the eventual objective is to generate useful
match-win probabilities rather than only binary winner predictions.

### V3 Feature Coefficients

Because V3 standardizes the predictors before logistic regression, coefficient
magnitudes provide a useful indication of their relative contribution within
the model.

The largest coefficients were:

| Feature | Standardized Coefficient |
|---|---:|
| `elo_diff` | **+0.478** |
| `rank_diff` | +0.247 |
| `win_rate_last_10_diff` | -0.247 |
| `points_diff` | +0.199 |
| `games_won_pct_last_10_diff` | +0.179 |
| `avg_service_points_won_last_10_diff` | +0.085 |
| `avg_return_points_won_last_10_diff` | +0.078 |
| `avg_aces_last_10_diff` | +0.076 |

`elo_diff` became the strongest individual predictor in V3, suggesting that
opponent-adjusted player strength contains substantial information beyond ATP
ranking and recent performance alone.

Some rolling statistics produced unexpected coefficient directions, most
notably `win_rate_last_10_diff`. These coefficients should not be interpreted
independently because several predictors measure overlapping aspects of player
strength. Further analysis of correlation, multicollinearity, and conditional
feature effects will therefore be performed before deciding whether these
features should be removed or modified.

### Current Conclusion

The progression from V1 to V3 suggests that combining multiple forms of
pre-match information improves prediction quality:

- ATP ranking provides an official measure of player standing.
- Recent rolling statistics capture current form and match performance.
- Elo provides an opponent-adjusted estimate of underlying player strength.

The next stage will investigate the behavior of the existing features and
continue feature engineering before testing more flexible machine-learning
models.

### V4_0: XGBoost Baseline Model
- This model used a different machine learning model: XGBoost to study the non-linear relationship
  between the features
- The model utilized the 11 features that were used previously except the Games Won Percentage from the 
  last 10 matches played.
- Games Won Percentage was dropped from the features because an ablation test suggested that removing 
  that feature would result in better results for the model
- The XGBClassifier was trained based on the 11 features with a max depth of 3, learning rate of 0.03 
  and 300 trees
- After fitting the model, it was found that XGBoost used Elo+ATP Rankings+ATP Points as the primary
  features that dominated the model training
- However, this version of XGBoost yielded: V4 XGBoost Accuracy: 0.640198225193039
  V4 XGBoost Log Loss: 0.635078305969855 which was worse than V3.


### V3.1: Logistic Regression Feature Ablation

Before moving further into tree-based models, feature ablation was performed on
the V3 Logistic Regression model to investigate whether all engineered features
were contributing useful predictive information.

The strongest improvement came from removing
`games_won_pct_last_10_diff`, which was highly correlated with other recent-form
features.

| Model | Accuracy | Log Loss |
|---|---:|---:|
| V3 Original | 64.30% | 0.63303 |
| V3 without Win Rate | 64.48% | 0.63369 |
| **V3.1 without Games Won %** | **64.65%** | **0.63249** |

Removing the games-won percentage improved both accuracy and log loss.

An additional ablation experiment removed `elo_diff` from V3.1:

| Model | Accuracy | Log Loss |
|---|---:|---:|
| **V3.1 with Elo** | **64.65%** | **0.63249** |
| V3.1 without Elo | 63.36% | 0.64075 |

Removing Elo caused accuracy to decrease by approximately 1.29 percentage
points and substantially increased log loss. This provided further evidence
that the chronological Elo rating is contributing predictive information beyond
ATP ranking, ranking points, and recent performance statistics.

V3.1 remains the best-performing model on the 2025 validation set so far.


### V4.1: Multi-Window Recent Form

The first extension of the XGBoost model investigated whether recent player form
could be represented at multiple time scales.

In addition to the existing win rate over the previous 10 matches, two new
lagged features were created:

- `win_rate_last_5_diff`
- `win_rate_last_20_diff`

This allows the model to distinguish short-term form from more stable
medium-term performance.

For example, a player may have a 100% win rate over their previous five matches,
a 90% win rate over their previous ten matches, and an 80% win rate over their
previous twenty matches.

As with all historical performance features, the current match is excluded
using a one-match lag before calculating the rolling statistics.

The same XGBoost hyperparameters and the same training/validation population
were retained so that V4.0 and V4.1 could be compared directly.

| Model | Accuracy | Log Loss |
|---|---:|---:|
| V4.0 XGBoost | 64.02% | 0.63508 |
| **V4.1 + Multi-Window Form** | **64.09%** | **0.63498** |

The additional form windows produced a small improvement in both metrics,
suggesting that recent form at different time scales contains some additional
signal, although the improvement was limited.


## V5: Opponent-Adjusted Recent Form

V5 introduced a more sophisticated measure of recent form that accounts for
the strength of the opponents a player faced.

Raw win rate treats all victories equally. For example, an 80% win rate against
lower-rated opponents is treated the same as an 80% win rate against elite
opposition.

To address this limitation, each player's pre-match Elo rating and their
opponent's pre-match Elo rating were used to calculate the expected probability
of winning each historical match:

\[
P(\text{win}) =
\frac{1}
{1 + 10^{(R_{\text{opponent}} - R_{\text{player}})/400}}
\]

An Elo performance residual was then calculated:

\[
\text{Elo Performance}
=
\text{Actual Result}
-
\text{Expected Win Probability}
\]

where the actual result is 1 for a win and 0 for a loss.

This produces an opponent-adjusted measure of performance.

For example:

- An underdog with a 20% expected win probability who wins receives a large
  positive residual of approximately +0.80.
- A heavy favorite with an 80% expected win probability who loses receives a
  large negative residual of approximately -0.80.

Rolling averages of this residual were then calculated over the player's
previous:

- 5 matches
- 10 matches
- 20 matches

creating:

- `elo_form_last_5_diff`
- `elo_form_last_10_diff`
- `elo_form_last_20_diff`

All rolling calculations are lagged by one match, ensuring that the result of
the match being predicted cannot enter its own features.

### V5 Results

The opponent-adjusted form features were added to the V4.1 XGBoost model while
keeping the same XGBoost configuration.

| Model | Accuracy | Log Loss |
|---|---:|---:|
| V4.0 XGBoost | 64.02% | 0.63508 |
| V4.1 + Multi-Window Form | 64.09% | 0.63498 |
| **V5 + Opponent-Adjusted Form** | **64.28%** | **0.63448** |

V5 improved both accuracy and log loss compared with V4.1.

This suggests that recent results become more informative when they are
evaluated relative to the quality of opposition rather than treating every win
and loss equally.

However, V5 XGBoost still does not outperform V3.1 Logistic Regression, which
currently remains the strongest validation model.


## Current Model Comparison

| Version | Model | Main Addition | Accuracy | Log Loss |
|---|---|---|---:|---:|
| Baseline | Higher-ranked player | ATP Rank | 62.54% | — |
| V1 | Logistic Regression | Rank + Points | 62.72% | 0.6467 |
| V2 | Logistic Regression | Historical Performance | 63.32% | 0.6407 |
| V3 | Logistic Regression | Elo | 64.30% | 0.63303 |
| **V3.1** | **Logistic Regression** | **Feature Ablation** | **64.65%** | **0.63249** |
| V4.0 | XGBoost | Nonlinear Model | 64.02% | 0.63508 |
| V4.1 | XGBoost | 5/10/20 Match Form | 64.09% | 0.63498 |
| V5 | XGBoost | Opponent-Adjusted Form | 64.28% | 0.63448 |

The current best model is **V3.1 Logistic Regression**, with a validation
accuracy of approximately **64.65%** and log loss of **0.63249**.

Although XGBoost has not yet surpassed Logistic Regression, the progression from
V4.0 to V5 shows that richer feature engineering is improving its performance.


## Current Development: Activity and Match Context

The next stage of feature engineering investigates player activity and match
readiness.

Features currently being developed include:

- Days since the player's previous match
- Number of matches played in the previous 30 days
- Number of matches played in the previous 90 days
- Number of prior matches available for the player

These features are intended to capture information that ranking, Elo, and
recent win rate cannot directly represent, including inactivity, match
sharpness, workload, and experience.

Future model versions will also investigate richer match context and player
representations before comparing additional machine-learning algorithms.


