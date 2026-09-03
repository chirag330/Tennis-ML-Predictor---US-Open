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



