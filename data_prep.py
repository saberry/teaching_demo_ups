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
