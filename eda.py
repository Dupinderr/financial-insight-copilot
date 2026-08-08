import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine

sns.set_style("whitegrid")


def load_data(db_path="financial_data.db", table_name="stocks_with_features"):
    engine = create_engine(f"sqlite:///{db_path}")
    df = pd.read_sql(table_name, engine, parse_dates=["Date"])
    return df


def plot_price_trends(df):
    plt.figure(figsize=(12, 6))
    for ticker in df["ticker"].unique():
        subset = df[df["ticker"] == ticker]
        plt.plot(subset["Date"], subset["Close"], label=ticker)
    plt.title("Closing Price Trends by Stock")
    plt.xlabel("Date")
    plt.ylabel("Close Price (INR)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_price_trends.png")
    plt.close()


def plot_sector_volatility(df):
    sector_vol = df.groupby("sector")["daily_return"].std().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sector_vol.plot(kind="bar", color="steelblue")
    plt.title("Overall Volatility by Sector (std of daily returns)")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig("plot_sector_volatility.png")
    plt.close()
    print("\nSector volatility ranking:\n", sector_vol)


def plot_correlation_heatmap(df):
    pivot = df.pivot_table(index="Date", columns="ticker", values="Close")
    corr = pivot.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Price Correlation Between Stocks")
    plt.tight_layout()
    plt.savefig("plot_correlation_heatmap.png")
    plt.close()
    return corr


def monthly_seasonality(df):
    df["month"] = df["Date"].dt.month
    monthly_avg_return = df.groupby("month")["daily_return"].mean()
    plt.figure(figsize=(10, 5))
    monthly_avg_return.plot(kind="bar", color="darkorange")
    plt.title("Average Daily Return by Month (All Stocks Combined)")
    plt.xlabel("Month")
    plt.ylabel("Avg Daily Return")
    plt.tight_layout()
    plt.savefig("plot_monthly_seasonality.png")
    plt.close()
    return monthly_avg_return


if __name__ == "__main__":
    df = load_data()

    plot_price_trends(df)
    plot_sector_volatility(df)
    corr = plot_correlation_heatmap(df)
    monthly_ret = monthly_seasonality(df)

    print("\nCorrelation matrix:\n", corr.round(2))
    print("\nMonthly avg return:\n", monthly_ret)
    print("\nSaved 4 plots: plot_price_trends.png, plot_sector_volatility.png, plot_correlation_heatmap.png, plot_monthly_seasonality.png")