import os

import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt

def download_nse_data(symbols,period='3y'):
    """Download NSE stock data from yfinance"""

    print(f"\n{'='*60}")
    print(f"Downloading NSE data | period {period}")
    print(f"\n{'=' * 60}")

    # os.mkdir("data", exist_ok=True)

    for symbol in symbols:
        try:
            print(f"Downloading {symbol}...", end=' ' , flush=True)

            data = yf.download(
                symbol,
                period=period,
                progress=False,
                timeout=10
            )

            # Quality chcecks
            if len(data) < 500:
                print(f"X Skip ((only {len(data)} rows)")
                continue

            # Save
            filename = f'data/{symbol.replace(".", "_")}_raw.csv'
            data.to_csv(filename)

            print(f"ok {len(data)} rows | {data.index[0].data()} to {data.index[-1].date()}")

        except Exception as e :
            print(f"Error: {str(e)[:50]}")

    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    # Stocks to focus on :
    # primary_stocks = ["RELIANCE.NS","INFY.NS","TCS.NS","LT.NS"]
    primary_stocks = ["COCHINSHIP.NS","COFORGE.NS"]

    download_nse_data(primary_stocks,period='3y')

    # Print summary
    print("\n Data Downloaded sccessfully")
    print("next step : Run eda.py for stationary testing")