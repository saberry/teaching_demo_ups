import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

options = pd.read_csv("ticket_features.csv", parse_dates=["dateTime"])

# Only keep games where the price actually moved from the previous game.
options = options[options["target"] != 0].reset_index(drop=True)
print("Games with a price move:", len(options))

drop_cols = ["name", "dateTime", "LL_price", "log_price", "target",
             "prev_log_price", "masseyRat", "Rank", "WL", "Wins", "Losses"]
features = [c for c in options.columns if c not in drop_cols]

X_all = options[features]
y_all = options["target"]

# The final 20% of games (chronologically) is never used for tuning or model selection.
split_at = int(len(options) * 0.8)
X, y = X_all.iloc[:split_at], y_all.iloc[:split_at]
X_hold, y_hold = X_all.iloc[split_at:], y_all.iloc[split_at:]
print("Dev games:", len(X), " Held-out games:", len(X_hold))

default_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, random_state=42)

tscv = TimeSeriesSplit(n_splits=5)

# Modest tuning: a small grid, with boosting rounds as the resource that gets
# larger for the surviving candidates.
param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [1, 10],
}
search = HalvingGridSearchCV(
    XGBRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42),
    param_grid,
    resource="n_estimators",
    min_resources=50,
    max_resources=300,
    factor=3,
    scoring="neg_mean_absolute_error",
    cv=tscv,
    random_state=42,
)
search.fit(X, y)
print("Best parameters:", search.best_params_)
tuned_model = search.best_estimator_

models = {"default": default_model, "tuned": tuned_model}
results = {}

for name, model in models.items():
    scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred_change = model.predict(X.iloc[test_idx])

        # Undo the transform: previous log price + predicted change -> dollars.
        prev_log = options["prev_log_price"].iloc[test_idx]
        pred_price = np.expm1(prev_log + pred_change)
        actual_price = options["LL_price"].iloc[test_idx]

        rmse = root_mean_squared_error(actual_price, pred_price)
        mae = mean_absolute_error(actual_price, pred_price)
        # Baseline: always guess the average price move seen in training.
        mean_move = y.iloc[train_idx].mean()
        naive_mae = mean_absolute_error(actual_price, np.expm1(prev_log + mean_move))
        scores.append((rmse, mae, naive_mae))
        print(f"{name} fold {fold}: RMSE={rmse:.2f}  MAE={mae:.2f}  (mean-move MAE={naive_mae:.2f})")

    results[name] = np.mean(scores, axis=0)
    print(f"{name} mean RMSE/MAE/baseline MAE:", results[name].round(2))

# Keep whichever configuration had the lower cross-validated dollar MAE.
best_name = min(results, key=lambda n: results[n][1])
final_model = models[best_name]
print("Selected:", best_name)

# One-time check on the untouched final period, trained on all development data.
final_model.fit(X, y)
# Development-only version, used by the two-stage evaluation so the held-out period stays unseen.
final_model.save_model("xgb_price_change_dev.json")
hold_pred = np.expm1(options["prev_log_price"].iloc[split_at:] + final_model.predict(X_hold))
hold_actual = options["LL_price"].iloc[split_at:]
hold_naive = np.expm1(options["prev_log_price"].iloc[split_at:] + y.mean())
print(f"Held-out RMSE={root_mean_squared_error(hold_actual, hold_pred):.2f}  "
      f"MAE={mean_absolute_error(hold_actual, hold_pred):.2f}  "
      f"(mean-move MAE={mean_absolute_error(hold_actual, hold_naive):.2f})")

# The saved model is refit on every price-move game, including the held-out period.
final_model.fit(X_all, y_all)
final_model.save_model("xgb_price_change.json")
importance = pd.Series(final_model.feature_importances_, index=features).sort_values(ascending=False)
print(importance.head(10))
