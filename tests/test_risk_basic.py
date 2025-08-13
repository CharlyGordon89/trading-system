import numpy as np
import pandas as pd
from src.risk_model import compute_risk_metrics

def test_cvar_not_greater_than_var():
    # synthetic one-symbol dataset with a negative tail
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    close = pd.Series(100.0, index=dates).copy()
    # create returns: mostly small, with some negative shocks
    rng = np.random.default_rng(0)
    rets = np.clip(rng.normal(0.0005, 0.01, size=80), -0.06, 0.05)
    rets[10] = -0.08
    rets[30] = -0.05
    rets[55] = -0.07
    # build close from returns
    for i in range(1, len(close)):
        close.iloc[i] = close.iloc[i-1] * (1 + rets[i])

    df = pd.DataFrame({
        "symbol": ["AAA"] * len(dates),
        "close": close.values,
        "volume": np.linspace(1_000, 2_000, len(dates))
    }, index=dates)

    risk = compute_risk_metrics(df, vol_window=10, var_window=30, cvar_alpha=0.05, liq_window=5)

    # check columns exist
    assert "ret" in risk.columns
    assert any(col.startswith("vol_") for col in risk.columns)
    var_col = [c for c in risk.columns if c.startswith("var_")][0]
    cvar_col = [c for c in risk.columns if c.startswith("cvar_")][0]

    # take last non-null window
    tail = risk[[var_col, cvar_col]].dropna().iloc[-1]
    # CVaR (mean of worst 5%) should be <= VaR (5% quantile) for lower tail returns
    assert tail[cvar_col] <= tail[var_col] + 1e-12
