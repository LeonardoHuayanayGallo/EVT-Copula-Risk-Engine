"""
data_loader.py
Downloads historical daily price data using yfinance (latest version).
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os


def download_returns(
    tickers: list,
    start: str = "2018-01-01",
    end: str   = "2024-12-31",
    save_dir: str = "data"
) -> pd.DataFrame:

    os.makedirs(save_dir, exist_ok=True)
    print(f"[DATA] Downloading price data for: {tickers}")

    all_prices = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                ignore_tz=True
            )
            if df.empty:
                print(f"[DATA] {ticker}: no data returned — skipping")
                continue

            # Handle both single and multi-level columns
            if isinstance(df.columns, pd.MultiIndex):
                close = df[("Close", ticker)]
            else:
                close = df["Close"]

            all_prices[ticker] = close
            print(f"[DATA] {ticker}: {len(df)} observations downloaded")

        except Exception as e:
            print(f"[DATA] {ticker} failed: {e}")

    if not all_prices:
        raise ValueError(
            "No data downloaded. Check internet connection."
        )

    prices = pd.DataFrame(all_prices)
    prices = prices.ffill().dropna()

    log_returns = np.log(prices / prices.shift(1)).dropna()

    print(f"\n[DATA] Total observations: {len(log_returns)}")
    print(f"[DATA] Date range: {log_returns.index[0].date()} "
          f"to {log_returns.index[-1].date()}")

    save_path = os.path.join(save_dir, "log_returns.csv")
    log_returns.to_csv(save_path)
    print(f"[DATA] Returns saved to {save_path}")

    return log_returns


def get_portfolio_returns(
    log_returns: pd.DataFrame,
    weights: list = None
) -> pd.Series:

    if weights is None:
        weights = [1 / len(log_returns.columns)] * len(log_returns.columns)

    weights = np.array(weights)
    portfolio = log_returns.values @ weights
    return pd.Series(portfolio, index=log_returns.index, name="Portfolio")