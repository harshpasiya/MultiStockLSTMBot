import os
import yfinance as yf
import pandas as pd

def download_nse_data(symbols, period="3y"):
    """Download NSE stock data from yfinance"""

    os.makedirs("data", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading NSE data | period = {period}")
    print(f"{'='*60}\n")

    for symbol in symbols:
        try:
            print(f"Downloading {symbol}...", end=" ", flush=True)

            df = yf.download(
                symbol,
                period=period,
                progress=False,
                timeout=10,
                auto_adjust=True
            )

            # Quality check
            if df.empty or len(df) < 500:
                print(f"X Skip (only {len(df)} rows)")
                continue

            df = df.dropna()

            filename = f"data/{symbol.replace('.', '_')}_raw.csv"
            df.to_csv(filename)

            print(
                f"ok {len(df)} rows | "
                f"{df.index[0].date()} to {df.index[-1].date()}"
            )

        except Exception as e:
            print(f"Error: {str(e)}")

    print(f"\n{'='*60}")
    print("Data download completed")
    print(f"{'='*60}\n")


if __name__ == "__main__":

    primary_stocks = [
        "ADANIPORTS.NS", "COCHINSHIP.NS", "LT.NS", "IRCON.NS", "RVNL.NS",
        "ADANIPOWER.NS", "TATAPOWER.NS", "NTPC.NS", "COALINDIA.NS", "ONGC.NS",
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "SAIL.NS", "NATIONALUM.NS",
        "SBIN.NS", "BANKBARODA.NS", "PNB.NS", "AXISBANK.NS", "ICICIBANK.NS",
        "BEL.NS", "HAL.NS", "MAZDOCK.NS", "DEEPAKNTR.NS", "AARTIIND.NS",
        "CONCOR.NS", "GMRINFRA.NS", "ADANIENSOL.NS", "JSWENERGY.NS", "NHPC.NS"
    ]

    download_nse_data(primary_stocks, period="5y")
    print("Next step: Run feature_engineering.py")
