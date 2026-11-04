import polars as pl
import pandas as pd

options = pd.read_csv("ticket_options.csv")

short_months = list(pd.date_range(start='2026-01-01', periods=12, freq='ME').strftime('%b'))

dict_months = {month: i for i, month in enumerate(short_months, start=1)}

for key, value in dict_months.items():
    options['WL'] = options['WL'].str.replace(key, str(value))

options = pl.DataFrame(options)

options = options.select(
    ['name', 'dateTime', 'LL_price', 'masseyRat', 'Rank', 'WL', 'computerPick']
)

options = options.with_columns(pl.col('WL').str.replace('00', '0'))

options = options.with_columns(
    pl.col("WL").str.extract(r"(^\d+)").alias("Wins"),
    pl.col("WL").str.extract(r"(\d+$)").alias("Losses")
)

options = options.with_columns(
    pl.col('dateTime').str.strptime(pl.Datetime, format="%m/%d/%Y %H:%M")
)

options.write_csv("ticket_options_cleaned.csv")

'''
Use xgboost to predict the LL_price field within the #ticket_options_cleaned.csv file. 
A two-stage model (classifier, then this regressor) will need to be built, 
with the classifier predicting whether the price will move at all, 
and the regressor predicting the magnitude of the price change.
Since this is time series data, appropriately transform the LL_price variable. 
You will also need to use the TimeSeriesSplit function from sklearn. 
Create features with appropriate lags and rolling windows from masseyRat, Rank, Wins, Losses, and computerPick. 
Create a separate file for all data prep and for modeling.
Add a day of week and month feature to the data, and only keep rows where the
day of the week is Friday, Saturday, or Sunday. 
All code should be written in a human-centered approach and avoid writing unnecessary functions to perform tasks.
If the outcome needs to be changed due to little movement, change the target to only predict games where the price moves. 
Save the xgboost model and create a new file with explainer visualizations.
After a single run of the model, perform modest parameter tuning with HalvingGridSearchCV
Include a held-out final period for evaluation.
'''