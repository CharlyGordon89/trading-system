from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

def _rolling_quantile(arr: np.ndarray, q: float) -> float:
    return float(np.quantile(arr, q))

def _rolling_cvar(arr: np.ndarray, q: float) -> float:
    var_q = np.quantile(arr, q)
    tail = arr[arr <= var_q]
    if tail.size == 0:
        return np.nan
    return float(tail.mean())

def compute_risk_metrics(
    assets: pd.DataFrame,
    vol_window: int = 20,
    var_window: int = 60,
    cvar_alpha: float = 0.05,
    liq_window: int = 20,
    ret_col: str = "ret",
) -> pd.DataFrame:
    """
    assets: index=date, columns=[symbol, close, volume]
    Returns a DataFrame (index=date) with columns:
      symbol, close, volume, ret, vol_<N>, var_<alpha>, cvar_<alpha>, liq_score
    Notes:
      - VaR/CVaR are on returns (lower tail). Values will be <= 0 for loss risk.
      - Liquidity score is a rolling average of volume (simple MVP).
    """
    if assets.empty:
        return assets.copy()

    df = assets.copy()
    # 1) Returns
    df[ret_col] = (
    df.groupby("symbol", observed=True)["close"]
      .transform(lambda s: s.pct_change(fill_method=None))
    )

    # 2) Volatility (rolling std of returns)
    df[f"vol_{vol_window}"] = (
        df.groupby("symbol", observed=True)[ret_col]
          .transform(lambda s: s.rolling(vol_window, min_periods=max(2, vol_window//2)).std())
    )

    # 3) Historical VaR & CVaR (lower tail)
    df[f"var_{int((1-cvar_alpha)*100)}"] = (
        df.groupby("symbol", observed=True)[ret_col]
          .transform(lambda s: s.rolling(var_window, min_periods=var_window)
                    .apply(lambda a: _rolling_quantile(a, cvar_alpha), raw=True))
    )
    df[f"cvar_{int((1-cvar_alpha)*100)}"] = (
        df.groupby("symbol", observed=True)[ret_col]
          .transform(lambda s: s.rolling(var_window, min_periods=var_window)
                    .apply(lambda a: _rolling_cvar(a, cvar_alpha), raw=True))
    )

    # 4) Liquidity score (rolling mean volume)
    df["liq_score"] = (
        df.groupby("symbol", observed=True)["volume"]
          .transform(lambda s: s.rolling(liq_window, min_periods=max(2, liq_window//2)).mean())
    )

    return df
