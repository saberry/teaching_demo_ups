# Key Findings: Predicting Ticket Price Moves with XGBoost

## Approach
- **Data:** `ticket_options_cleaned.csv`, restricted to **Friday, Saturday and Sunday games** (54,507 team-game rows after feature creation).
- **Target transform:** `LL_price` is log-transformed (`log1p`) and differenced within each team, so the model predicts the *change in log price* from the team's previous game. Predictions are converted back to dollars for scoring.
- **Features:** lags 1–3 and 3-game rolling mean/std of `masseyRat`, `Rank`, `Wins`, `Losses` and lagged log price, plus month, day of week and `computerPick` (the current game's binary pick, its previous-game value and a 3-game rolling mean). `computerPick` is missing for about 5% of the weekend rows; those rows are kept and XGBoost handles the gaps natively. All lag and rolling features are shifted one game to prevent leakage; the current-game `computerPick` is used as-is because it is a pre-game pick. The weekend filter is applied *after* the lags are built, so "previous game" still means the true previous game.
- **Held-out final period:** the chronologically last 20% of price-move games (114 of 568) is never used for tuning or model selection.
- **Validation and tuning:** `TimeSeriesSplit` with 5 folds on the development set; after a single default run, `HalvingGridSearchCV` (MAE scoring, `n_estimators` 50–300 as the resource) searches a small grid over `max_depth`, `learning_rate`, `min_child_weight` and `reg_lambda`.

## Finding 1: Most prices never change
Only 568 of 54,507 weekend games (about 1%) had a price move. Modeling all games, a "price stays the same" baseline beat XGBoost (shown in an earlier all-day run), so the regressor is trained only on games where the price moved.

## Finding 2: Regressor on price-move games (454 development games)
| | Default XGBoost | Tuned XGBoost | Baseline (mean move) |
|---|---|---|---|
| Mean CV MAE | 17.26 | 13.18 | 21.77 |
| Mean CV RMSE | 27.92 | 24.45 | – |

- Best parameters: `learning_rate=0.03`, `max_depth=3`, `min_child_weight=10`, `reg_lambda=10`, `n_estimators=150`. Shallow, heavily regularized trees suit the small sample.
- Tuning cut cross-validated MAE by about 24%.

## Finding 3: Held-out period is less flattering
| | Tuned XGBoost | Baseline |
|---|---|---|
| MAE | 23.60 | 31.37 |
| RMSE | 40.68 | – |

The model still beats the baseline, but by about 25% rather than the roughly 40% seen in cross-validation (13.18 vs 21.77). Expect real-world error nearer the held-out figure. The saved model (`xgb_price_change.json`) is refit on all price-move games after this check.

## Finding 4: Two-stage model (classifier → regressor)
A stage-one `XGBClassifier` predicts whether a price moves (up-weighted for rarity, tuned on average precision, threshold 0.896 chosen on out-of-fold dev predictions); stage two is the development-only regressor.

| Classifier | Cross-validated | Held-out |
|---|---|---|
| ROC AUC | 0.869 | 0.938 |
| Average precision (base rate) | 0.097 (0.010) | 0.214 (0.021) |
| F1 | 0.183 | 0.259 |

Price error on **all** 5,352 held-out games, in dollars:

| Approach | MAE | RMSE |
|---|---|---|
| No change | 0.49 | 5.88 |
| Two-stage (move if flagged) | 0.96 | 6.45 |
| Two-stage (probability-weighted) | 1.15 | 6.44 |

- The classifier ranks games well (AUC about 0.94 held-out, roughly 10x lift over the base rate), but precision is low, and false alarms are costly in dollars.
- As a result the two-stage forecast is still **worse than "no change"** overall. It is better used as a move-alert ranking than a price forecaster.

## Effect of the weekend filter
Compared with using all days of the week, the data shrinks to about 44% of the rows and the number of price moves roughly halves (1,059 → 568), making every estimate noisier. Tuned regressor CV MAE was about the same (13.18 vs 12.47) and held-out MAE slightly worse (23.60 vs 20.33), while the classifier's held-out AUC improved (0.938 vs 0.885). These differences are within what small samples can produce, so don't read them as a real filter effect.

## Feature importance and SHAP
`shap_importance.png`, `shap_beeswarm.png`, `shap_dependence_top_feature.png` and `shap_single_prediction.png` were regenerated for the final regressor. Earlier all-day runs had smoothed Massey rating and rank (3-game rolling means) as the top drivers; the plots were not reviewed for this weekend-only run.

## Effect of computerPick
Adding `computerPick` changed results only slightly: tuned CV MAE 12.95 → 13.18, held-out MAE 22.17 → 23.60, classifier held-out AUC 0.936 → 0.938 and average precision 0.196 → 0.214. With 568 price moves these shifts are within noise, so the feature gives no clear gain for the regressor and at most a small one for move detection. SHAP plots were regenerated but not reviewed.

## Caveats
- With 568 price moves, results are noisy; the held-out scores are single estimates.
- Future improvements: move-specific features (days to game, opponent, venue, inventory) and a cost-aware classification threshold.

## Files
| File | Purpose |
|---|---|
| `feature_prep.py` | Builds target and lag/rolling features, keeps Fri/Sat/Sun games → `ticket_features.csv` |
| `model_xgboost.py` | Regressor: filters to price moves, holds out the final 20%, default vs `HalvingGridSearchCV`-tuned, saves `xgb_price_change.json` (and a dev-only copy) |
| `model_two_stage.py` | Classifier, threshold selection, combined held-out evaluation, saves `xgb_move_classifier.json` |
| `explain_model.py` | SHAP plots for the regressor (`shap_*.png`) |
