import pandas as pd
from sqlalchemy import create_engine


def load_from_sqlite(db_path="financial_data.db", table_name="stocks"):
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql(table_name, engine, parse_dates=["Date"])
    df.sort_values(["ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, engine


def add_moving_averages(df):
    df["ma_7"] = df.groupby("ticker")["Close"].transform(lambda x: x.rolling(7).mean())
    df["ma_30"] = df.groupby("ticker")["Close"].transform(lambda x: x.rolling(30).mean())
    return df


def add_volatility(df, window=30):
    # daily_return already exists from data_pipeline.py's clean_data step
    df["volatility_30d"] = df.groupby("ticker")["daily_return"].transform(
        lambda x: x.rolling(window).std()
    )
    return df


def add_rsi(df, window=14):
    def rsi_for_group(close_prices):
        delta = close_prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    df["rsi_14"] = df.groupby("ticker")["Close"].transform(rsi_for_group)
    return df


def build_features(df):
    df = add_moving_averages(df)
    df = add_volatility(df)
    df = add_rsi(df)
    return df


if __name__ == "__main__":
    df, engine = load_from_sqlite()
    df = build_features(df)

    print("New columns:", [c for c in df.columns if c not in
          ["Date", "Close", "High", "Low", "Open", "Volume", "ticker", "sector", "daily_return", "is_outlier"]])

    # Sanity check on one ticker
    sample = df[df["ticker"] == "RELIANCE.NS"][["Date", "Close", "ma_7", "ma_30", "volatility_30d", "rsi_14"]]
    print("\nRELIANCE.NS sample (last 10 rows):\n", sample.tail(10))

    # Save back to SQLite with features included
    df.to_sql("stocks_with_features", engine, if_exists="replace", index=False)
    print(f"\nSaved {len(df)} rows to table 'stocks_with_features'")