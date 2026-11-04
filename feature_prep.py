import numpy as np
import pandas as pd

options = pd.read_csv("ticket_options_cleaned.csv", parse_dates=["dateTime"])
options = options.sort_values(["name", "dateTime"]).reset_index(drop=True)

# Log stabilizes the variance; differencing within team removes the trend.
options["log_price"] = np.log1p(options["LL_price"])
by_team = options.groupby("name")
options["prev_log_price"] = by_team["log_price"].shift(1)
options["target"] = options["log_price"] - options["prev_log_price"]

# Every feature is shifted one game so that nothing from the current game leaks in.
for col in ["masseyRat", "Rank", "Wins", "Losses", "log_price"]:
    shifted = options.groupby("name")[col].shift(1)
    options[f"{col}_lag1"] = shifted
    options[f"{col}_lag2"] = options.groupby("name")[col].shift(2)
    options[f"{col}_lag3"] = options.groupby("name")[col].shift(3)

    rolling = shifted.groupby(options["name"]).rolling(3)
    options[f"{col}_roll3_mean"] = rolling.mean().reset_index(level=0, drop=True)
    options[f"{col}_roll3_std"] = rolling.std().reset_index(level=0, drop=True)

# computerPick is a binary pre-game pick, so the current game's value is usable as-is.
# It has gaps, which XGBoost handles natively, so those rows are kept.
options["computerPick_lag1"] = options.groupby("name")["computerPick"].shift(1)
options["computerPick_roll3_mean"] = (
    options.groupby("name")["computerPick"].shift(1)
    .groupby(options["name"]).rolling(3, min_periods=1).mean()
    .reset_index(level=0, drop=True)
)

options["month"] = options["dateTime"].dt.month
options["day_of_week"] = options["dateTime"].dt.dayofweek

# Filter after the lags are built so "previous game" still means the true previous game.
options = options[options["day_of_week"].isin([4, 5, 6])]  # Fri, Sat, Sun

required = [c for c in options.columns if not c.startswith("computerPick")]
options = options.dropna(subset=required).sort_values("dateTime").reset_index(drop=True)

options.to_csv("ticket_features.csv", index=False)
print(options.shape)
