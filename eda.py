# Exploratory Data Analysis (Hardened for NSE / Custom CSVs)

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf


# ------------------------------------------------------------
# ADF TEST
# ------------------------------------------------------------
def adf_test(series, name):
    series = series.dropna()

    stat, pvalue, lags, _, _, _ = adfuller(series)

    print(f"\n{name}")
    print(f"  ADF p-value      : {pvalue:.6f}")
    print(f"  Test Statistic  : {stat:.4f}")
    print(f"  Lags Used       : {lags}")

    if pvalue < 0.05:
        print("  ✔ Stationary")
    else:
        print("  ✘ Not Stationary")

    return pvalue < 0.05


# ------------------------------------------------------------
# MAIN EDA FUNCTION
# ------------------------------------------------------------
def analyze_stock(filepath):

    print("\n" + "=" * 60)
    print(f" ANALYZING : {filepath}")
    print("=" * 60)

    # ---------------- LOAD DATA ----------------
    data = pd.read_csv(filepath, index_col=0)

    # ---------------- CLEAN DATE INDEX ----------------
    data.index = data.index.astype(str).str.strip()

    data.index = pd.to_datetime(
        data.index,
        format="%Y-%m-%d",     # change only if your CSV uses another format
        errors="coerce"
    )

    data = data[~data.index.isna()]

    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Index is not DatetimeIndex")

    # ---------------- CLEAN COLUMNS ----------------
    data.columns = data.columns.str.strip()

    if "Close" not in data.columns:
        raise KeyError("Required column 'Close' not found")

    # ---------------- CLEAN CLOSE COLUMN ----------------
    data["Close"] = (
        data["Close"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
    )

    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna(subset=["Close"])

    if not np.issubdtype(data["Close"].dtype, np.number):
        raise TypeError("Close column is not numeric after cleaning")

    # ---------------- BASIC INFO ----------------
    print(f"\nData Shape      : {data.shape}")
    print(f"Date Range     : {data.index.min().date()} → {data.index.max().date()}")
    print(f"Missing Values : {data.isnull().sum().sum()}")

    # ---------------- RETURNS ----------------
    data["Close_Return"] = data["Close"].pct_change()
    data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))

    # ---------------- STATIONARITY ----------------
    print("\n" + "-" * 60)
    print(" STATIONARITY TESTS (ADF)")
    print("-" * 60)

    is_stationary_price = adf_test(data["Close"], "Price")
    is_stationary_return = adf_test(data["Log_Return"], "Log Returns")

    # ---------------- PLOTS ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(data.index, data["Close"], linewidth=1)
    axes[0, 0].set_title("Price Movement")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(data.index, data["Log_Return"], linewidth=1)
    axes[0, 1].set_title("Log Returns")
    axes[0, 1].grid(True, alpha=0.3)

    plot_acf(data["Log_Return"].dropna(), lags=30, ax=axes[1, 0])
    axes[1, 0].set_title("ACF of Log Returns")

    axes[1, 1].hist(data["Log_Return"].dropna(), bins=50, alpha=0.7)
    axes[1, 1].set_title("Return Distribution")
    axes[1, 1].axvline(0, linestyle="--", alpha=0.6)
    axes[1, 1].grid(True, alpha=0.3)

    path = Path(filepath)
    plots_dir = Path("plots")

    plt.tight_layout()
    filename = plots_dir / path.name.replace("_raw.csv", "_eda.png")
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()

    print(f"\nPlot Saved: {filename}")

    # ---------------- SUMMARY ----------------
    print("\n" + "─" * 60)
    print(" SUMMARY STATISTICS")
    print("─" * 60)

    print(f"Mean Return : {data['Log_Return'].mean():.4%}")
    print(f"Std Return  : {data['Log_Return'].std():.4%}")
    print(f"Skewness   : {data['Log_Return'].skew():.4f}")
    print(f"Kurtosis   : {data['Log_Return'].kurtosis():.4f}")

    if is_stationary_return:
        print("\n✔ Log Returns are STATIONARY → Suitable for LSTM")
    else:
        print("\n✘ Log Returns NOT Stationary → Differencing recommended")

    return data

# ------------------------------------------------------------
# EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":

    import glob

    files = glob.glob("data/*_raw.csv")

    for filepath in sorted(files)[:1]:
        analyze_stock(filepath)

    print("\n" + "=" * 60)
    print("✓ EDA Complete")
    print("✓ Next step: Run feature_engineering.py")
    print("=" * 60)
