import numpy as np
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.metrics import (average_precision_score, f1_score,
                             mean_absolute_error, precision_recall_curve,
                             roc_auc_score, root_mean_squared_error)
from sklearn.model_selection import HalvingGridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier, XGBRegressor

options = pd.read_csv("ticket_features.csv", parse_dates=["dateTime"])
options["moved"] = (options["target"] != 0).astype(int)

drop_cols = ["name", "dateTime", "LL_price", "log_price", "target", "moved",
             "prev_log_price", "masseyRat", "Rank", "WL", "Wins", "Losses"]
features = [c for c in options.columns if c not in drop_cols]

# Same cutoff as the regressor: the last 20% of price-move games, by date.
moved_games = options[options["moved"] == 1]
cutoff = moved_games["dateTime"].iloc[int(len(moved_games) * 0.8)]
dev = options[options["dateTime"] < cutoff].reset_index(drop=True)
hold = options[options["dateTime"] >= cutoff].reset_index(drop=True)
print(f"Dev games: {len(dev)} ({dev['moved'].sum()} moves)  "
      f"Held-out games: {len(hold)} ({hold['moved'].sum()} moves)")

X, y = dev[features], dev["moved"]

# Moves are rare, so up-weight them and tune on average precision.
imbalance = (y == 0).sum() / (y == 1).sum()
tscv = TimeSeriesSplit(n_splits=5)

search = HalvingGridSearchCV(
    XGBClassifier(scale_pos_weight=imbalance, subsample=0.8,
                  colsample_bytree=0.8, random_state=42),
    {"max_depth": [2, 3, 4], "learning_rate": [0.05, 0.1],
     "min_child_weight": [1, 5, 10]},
    resource="n_estimators", min_resources=50, max_resources=300, factor=3,
    scoring="average_precision", cv=tscv, random_state=42,
)
search.fit(X, y)
print("Best parameters:", search.best_params_)
classifier = search.best_estimator_

# Out-of-fold probabilities let us pick the decision threshold without touching the held-out period.
oof_prob, oof_true = [], []
for train_idx, test_idx in tscv.split(X):
    classifier.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_prob.append(classifier.predict_proba(X.iloc[test_idx])[:, 1])
    oof_true.append(y.iloc[test_idx])
oof_prob, oof_true = np.concatenate(oof_prob), np.concatenate(oof_true)

print(f"CV average precision={average_precision_score(oof_true, oof_prob):.3f} "
      f"(base rate {oof_true.mean():.4f})  ROC AUC={roc_auc_score(oof_true, oof_prob):.3f}")

precision, recall, thresholds = precision_recall_curve(oof_true, oof_prob)
f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
threshold = thresholds[f1.argmax()]
print(f"Chosen threshold={threshold:.3f}  CV F1={f1.max():.3f}")

classifier.fit(X, y)
hold_prob = classifier.predict_proba(hold[features])[:, 1]
hold_flag = hold_prob >= threshold
print(f"Held-out: average precision={average_precision_score(hold['moved'], hold_prob):.3f} "
      f"(base rate {hold['moved'].mean():.4f})  ROC AUC={roc_auc_score(hold['moved'], hold_prob):.3f}  "
      f"F1={f1_score(hold['moved'], hold_flag):.3f}")

# Stage two: the dev-only regressor from model_xgboost.py estimates the size of the move.
regressor = XGBRegressor()
regressor.load_model("xgb_price_change_dev.json")
pred_move = regressor.predict(hold[features])

prev_log = hold["prev_log_price"]
actual = hold["LL_price"]
price_no_change = np.expm1(prev_log)
price_hard = np.where(hold_flag, np.expm1(prev_log + pred_move), price_no_change)
price_soft = np.expm1(prev_log + hold_prob * pred_move)

print("\nHeld-out price error across ALL games (dollars):")
for label, pred in [("No change", price_no_change),
                    ("Two-stage (hard)", price_hard),
                    ("Two-stage (probability-weighted)", price_soft)]:
    print(f"{label:34s} MAE={mean_absolute_error(actual, pred):.4f}  "
          f"RMSE={root_mean_squared_error(actual, pred):.2f}")

classifier.fit(options[features], options["moved"])
classifier.save_model("xgb_move_classifier.json")
