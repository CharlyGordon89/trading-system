from __future__ import annotations
import pandas as pd
import yfinance as yf
from typing import List

def download_assets(
    tickers: List[str],
    start_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns:
    ['symbol', 'date', 'close', 'volume']
    """
    if not tickers:
        return pd.DataFrame(columns=["symbol", "date", "close", "volume"]).set_index("date")

    data = yf.download(
        tickers=tickers,
        start=start_date,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    # Normalize to tidy format
    frames = []
    # yfinance structure differs for 1 vs many tickers; handle both
    if isinstance(data.columns, pd.MultiIndex):
        for sym in tickers:
            if sym not in data.columns.get_level_values(0):
                continue
            df = data[sym][["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})
            df = df.reset_index().rename(columns={"Date": "date"})
            df["symbol"] = sym
            frames.append(df)
    else:
        df = data[["Close", "Volume"]].rename(columns={"Close": "close", "Volume": "volume"})
        df = df.reset_index().rename(columns={"Date": "date"})
        df["symbol"] = tickers[0]
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["symbol", "date", "close", "volume"]).set_index("date")

    tidy = pd.concat(frames, ignore_index=True)
    tidy["date"] = pd.to_datetime(tidy["date"]).dt.tz_localize(None)
    return tidy.sort_values(["symbol", "date"]).set_index("date")
