import numpy as np
import pandas as pd
from src.backtest import run_backtest

def _toy_risk_df():
    # 2 symbols, ~400 business days
    dates = pd.date_range("2023-01-02", periods=400, freq="B")
    rng = np.random.default_rng(0)
    frames = []
    for sym, mu, sigma in [("AAA", 0.0003, 0.01), ("BBB", 0.0002, 0.012)]:
        r = rng.normal(mu, sigma, len(dates))
        close = 100 * (1 + pd.Series(r, index=dates)).cumprod()
        frames.append(pd.DataFrame({"symbol": sym, "close": close.values, "volume": 1_000, "ret": r}, index=dates))
    return pd.concat(frames).sort_index()

def test_backtest_runs_and_outputs_perf():
    risk_df = _toy_risk_df()
    opt_cfg = {"ret_col": "ret", "lookback": 120, "gamma": 2.5, "tau": 0.05, "long_only": True}
    bt_cfg = {"rebalance": "M", "initial_capital": 100000, "tc_bps": 5, "lookback": 120, "benchmark": "SPY"}
    perf, w_hist = run_backtest(risk_df, optimizer_cfg=opt_cfg, backtest_cfg=bt_cfg)

    assert {"portfolio_return", "portfolio_equity", "drawdown", "benchmark_equity"} <= set(perf.columns)
    assert len(perf) > 100
    assert not w_hist.empty
