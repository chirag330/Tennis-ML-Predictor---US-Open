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



