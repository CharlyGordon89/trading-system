import numpy as np
import pandas as pd
from src.optimizer.bl import optimize_allocation

def _toy_risk_df():
    # 3 symbols, 300 days of tiny random returns
    dates = pd.date_range("2023-01-01", periods=300, freq="B")
    rng = np.random.default_rng(42)
    rets = {
        "AAA": rng.normal(0.0003, 0.01, len(dates)),
        "BBB": rng.normal(0.0002, 0.009, len(dates)),
        "CCC": rng.normal(0.0004, 0.012, len(dates)),
    }
    frames = []
    for sym, r in rets.items():
        frames.append(pd.DataFrame({"symbol": sym, "ret": r, "close": 100.0, "volume": 1_000}, index=dates))
    return pd.concat(frames).sort_index()

def test_weights_sum_to_one_and_nonnegative():
    df = _toy_risk_df()
    cfg = {
        "lookback": 120,
        "gamma": 2.5,
        "tau": 0.05,
        "long_only": True,
        "market_weights": "equal",
        "views": [
            {"type": "relative", "long": "AAA", "short": "BBB", "annualized_delta": 0.02, "uncertainty": 0.10}
        ],
        "ret_col": "ret",
    }
    w = optimize_allocation(df, cfg)
    assert abs(w["weight"].sum() - 1.0) < 1e-6
    assert (w["weight"] >= -1e-8).all()
    # sanity: at least one positive weight
    assert (w["weight"] > 0).any()


def test_optimizer_fallback_on_singular_matrix():
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    base = rng.standard_normal(len(dates))
    rets = {"AAA": base, "BBB": base, "CCC": rng.standard_normal(len(dates))}
    frames = [pd.DataFrame({"symbol": s, "ret": r, "close": 100.0, "volume": 1_000}, index=dates) for s, r in rets.items()]
    df = pd.concat(frames).sort_index()

    cfg = {
        "ret_col": "ret",
        "lookback": 90,
        "gamma": 2.5,
        "tau": 0.05,
        "long_only": True,
        "shrinkage": 0.10,
        "enable_equal_weight_fallback": True,
    }
    w = optimize_allocation(df, cfg)
    assert abs(w["weight"].sum() - 1.0) < 1e-6
    assert (w["weight"] >= 0).all()
    # If fallback was used, weights should be near equal; allow tolerance
    target = np.ones(len(w)) / len(w)
    assert (np.abs(w["weight"].to_numpy() - target) <= 0.2).all()
