from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Tuple
from src.optimizer.bl import optimize_allocation

def _returns_matrix(risk_df: pd.DataFrame, ret_col: str = "ret") -> pd.DataFrame:
    """Pivot to (date x symbol) daily returns matrix."""
    df = risk_df.dropna(subset=[ret_col])
    mat = df.pivot_table(index=df.index, columns="symbol", values=ret_col)
    # fill remaining NaNs with 0 for stability (MVP)
    return mat.fillna(0.0).sort_index()

def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Trading month-ends in the series."""
    return idx.to_series().groupby([idx.year, idx.month]).tail(1).index

def _benchmark_returns(symbol: str, start: str, end: str) -> pd.Series:
    px = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    rets = px.pct_change().dropna()
    rets.index = pd.to_datetime(rets.index).tz_localize(None)
    return rets

def run_backtest(
    risk_df: pd.DataFrame,
    optimizer_cfg: Dict,
    backtest_cfg: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Monthly rebalancing backtest using Black–Litterman optimizer each rebalance date
    with past lookback window only. Weights held static between rebalances.
    Returns:
      perf_df: per-day portfolio performance (value, return, drawdown) and benchmark equity
      w_hist:  weights at each rebalance date
    """
    ret_col = optimizer_cfg.get("ret_col", "ret")
    lookback = int(backtest_cfg.get("lookback", optimizer_cfg.get("lookback", 252)))
    initial_capital = float(backtest_cfg.get("initial_capital", 500000))
    tc_bps = float(backtest_cfg.get("tc_bps", 5.0))
    rebalance = backtest_cfg.get("rebalance", "M")
    bench_sym = backtest_cfg.get("benchmark", "SPY")

    mat = _returns_matrix(risk_df, ret_col=ret_col)
    if mat.empty or len(mat) < lookback + 20:
        raise ValueError("Not enough data to run the backtest (need lookback + ~1 month).")

    # Rebalance dates: month-ends after we have lookback history
    mends = _month_ends(mat.index)
    start_date = mat.index[0]
    valid_mends = mends[mends >= (start_date + pd.tseries.offsets.BDay(lookback))]
    if len(valid_mends) < 2:
        raise ValueError("Not enough rebalance points to run the backtest.")

    portfolio_value = initial_capital
    current_w = pd.Series(0.0, index=mat.columns)
    w_hist = []

    # Daily performance series to fill
    pnl = pd.Series(index=mat.index, dtype=float)
    equity = pd.Series(index=mat.index, dtype=float)

    # Iterate over monthly periods
    for i in range(len(valid_mends) - 1):
        rb_date = valid_mends[i]
        next_rb = valid_mends[i + 1]

        # Subset history for lookback
        hist = risk_df[risk_df.index <= rb_date]
        hist = hist[hist.index > (rb_date - pd.tseries.offsets.BDay(lookback + 5))]  # small cushion
        # Optimize weights using only past data up to rb_date
        new_w_df = optimize_allocation(hist, optimizer_cfg)
        # Align to universe we actually have returns for
        new_w = pd.Series(0.0, index=mat.columns)
        new_w.loc[new_w_df["symbol"].values] = new_w_df["weight"].values

        # Transaction cost on turnover
        turnover = float(np.abs(new_w - current_w).sum())
        cost = (tc_bps / 10000.0) * turnover * portfolio_value
        portfolio_value -= cost
        current_w = new_w.copy()

        w_hist.append(pd.DataFrame({"symbol": current_w.index, "weight": current_w.values, "date": rb_date}))

        # Apply weights for the next period (rb_date, next_rb]
        period_rets = mat.loc[(mat.index > rb_date) & (mat.index <= next_rb)]
        for dt, row in period_rets.iterrows():
            port_ret = float(np.nansum(current_w.values * row.values))
            portfolio_value *= (1.0 + port_ret)
            pnl.loc[dt] = port_ret
            equity.loc[dt] = portfolio_value

    # Trim NaNs
    pnl = pnl.dropna()
    equity = equity.dropna()

    # Drawdown
    roll_max = equity.cummax()
    dd = (equity / roll_max) - 1.0

    perf_df = pd.DataFrame({
        "portfolio_return": pnl,
        "portfolio_equity": equity,
        "drawdown": dd,
    })

    # Benchmark
    bench = _benchmark_returns(bench_sym, start=str(perf_df.index.min().date()), end=str(perf_df.index.max().date()))
    bench = bench.reindex(perf_df.index).fillna(0.0)
    bench_equity = (1.0 + bench).cumprod() * initial_capital
    perf_df["benchmark_return"] = bench
    perf_df["benchmark_equity"] = bench_equity

    w_hist_df = pd.concat(w_hist, ignore_index=True) if w_hist else pd.DataFrame(columns=["date", "symbol", "weight"])
    return perf_df, w_hist_df
