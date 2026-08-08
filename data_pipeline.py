import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine

TICKERS = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "SUNPHARMA.NS": "Pharma",
    "MARUTI.NS": "Auto",
    "ITC.NS": "FMCG",
}


import time

def fetch_raw_data(start="2022-01-01", end="2026-07-28"):
    all_data = []
    for ticker, sector in TICKERS.items():
        df = None
        for attempt in range(3):
            df = yf.download(ticker, start=start, end=end, progress=False)
            if not df.empty:
                break
            print(f"  Retry {attempt + 1}/3 for {ticker}...")
            time.sleep(2)

        if df is None or df.empty:
            print(f"WARNING: skipping {ticker} — no data after 3 attempts")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only the columns we actually use — drops 'Adj Close' if present,
        # and guarantees identical column sets across every ticker before concat
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        df.index.name = "Date"          # force consistent index name regardless of yfinance version
        df["ticker"] = ticker
        df["sector"] = sector
        all_data.append(df)

    data = pd.concat(all_data)
    data.reset_index(inplace=True)      # now guaranteed to produce a 'Date' column
    return data


def clean_data(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["ticker", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["daily_return"] = df.groupby("ticker")["Close"].pct_change()

    q1 = df.groupby("ticker")["daily_return"].transform(lambda x: x.quantile(0.25))
    q3 = df.groupby("ticker")["daily_return"].transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 3 * iqr
    upper = q3 + 3 * iqr
    df["is_outlier"] = (df["daily_return"] < lower) | (df["daily_return"] > upper)

    return df


def load_to_sqlite(df, db_path="financial_data.db", table_name="stocks"):
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    return engine


if __name__ == "__main__":
    raw = fetch_raw_data()
    print("Missing values per column:\n", raw.isnull().sum())

    cleaned = clean_data(raw)
    print("\nOutliers found per ticker:\n", cleaned.groupby("ticker")["is_outlier"].sum())
    print("\nSample outlier rows:\n", cleaned[cleaned["is_outlier"]][["Date", "ticker", "Close", "daily_return"]].head(10))

    cleaned.to_csv("cleaned_stock_data.csv", index=False)
    load_to_sqlite(cleaned)
    print(f"\nSaved {len(cleaned)} rows to financial_data.db (table: stocks)")