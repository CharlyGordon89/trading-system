from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import List, Dict, Tuple

# ---------- Moments preparation from risk dataframe ----------

def prepare_moments(
    risk_df: pd.DataFrame,
    ret_col: str = "ret",
    lookback: int = 252
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    risk_df: index=date, columns include ['symbol', ret_col]
    Returns: (symbols, mean_returns (daily), cov_matrix (daily))
    """
    if risk_df.empty:
        return [], np.array([]), np.array([[]])

    # pivot to (date x symbol) matrix of returns
    df = risk_df.dropna(subset=[ret_col])
    mat = df.pivot_table(index=df.index, columns="symbol", values=ret_col).dropna(how="all")

    if mat.empty:
        return [], np.array([]), np.array([[]])

    window = min(len(mat), lookback)
    mat = mat.tail(window).dropna(axis=1, how="all")
    mat = mat.fillna(0.0)  # MVP: simple fill; can improve later

    symbols = mat.columns.tolist()
    mu = mat.mean(axis=0).to_numpy()              # daily mean returns
    cov = np.cov(mat.to_numpy(), rowvar=False)    # daily covariance
    return symbols, mu, cov

# ---------- Black–Litterman core ----------

def implied_equilibrium_returns(cov: np.ndarray, w_mkt: np.ndarray, risk_aversion: float) -> np.ndarray:
    # π = δ Σ w_mkt
    return risk_aversion * cov.dot(w_mkt)

def black_litterman_posterior(
    pi: np.ndarray,
    cov: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    tau: float,
    omega: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    pi: prior (implied equilibrium) returns (n,)
    cov: prior covariance (n x n)
    P:   views loading matrix (k x n)
    Q:   views vector (k,)
    tau: scalar
    omega: views uncertainty (k x k)
    """
    if P.size == 0:  # no views ⇒ posterior == prior
        return pi, cov

    tauSigma_inv = np.linalg.inv(tau * cov)
    middle = P.T @ np.linalg.inv(omega) @ P
    post_cov_part = np.linalg.inv(tauSigma_inv + middle)
    mu_bl = post_cov_part @ (tauSigma_inv @ pi + P.T @ np.linalg.inv(omega) @ Q)
    # Common BL posterior covariance form:
    Sigma_bl = cov + post_cov_part
    return mu_bl, Sigma_bl

# ---------- Views helpers ----------

def build_views(symbols: List[str], views_cfg: List[Dict] | None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Supports:
      - relative: long - short = delta (annualized_delta optional)
      - absolute: symbol expected return = mu (annualized_mu optional)
    Uncertainty interpreted as annualized stdev; converted to daily.
    """
    if not views_cfg:
        return np.zeros((0, len(symbols))), np.zeros((0,)), np.eye(0)

    idx = {s: i for i, s in enumerate(symbols)}
    P_rows, Q_vals, omegas = [], [], []

    for v in views_cfg:
        vtype = v.get("type", "relative")
        ann = 252.0

        if vtype == "relative":
            i, j = idx.get(v.get("long")), idx.get(v.get("short"))
            if i is None or j is None:
                continue
            row = np.zeros(len(symbols)); row[i] = 1.0; row[j] = -1.0
            delta = v.get("annualized_delta", v.get("delta", 0.0))
            q = float(delta) / ann if "annualized_delta" in v else float(delta)
            unc = float(v.get("uncertainty", 0.10)) / np.sqrt(ann)
        else:  # absolute
            i = idx.get(v.get("symbol"))
            if i is None:
                continue
            row = np.zeros(len(symbols)); row[i] = 1.0
            mu = v.get("annualized_mu", v.get("mu", 0.0))
            q = float(mu) / ann if "annualized_mu" in v else float(mu)
            unc = float(v.get("uncertainty", 0.10)) / np.sqrt(ann)

        P_rows.append(row)
        Q_vals.append(q)
        omegas.append(unc ** 2)

    if not P_rows:
        return np.zeros((0, len(symbols))), np.zeros((0,)), np.eye(0)

    P = np.vstack(P_rows)
    Q = np.array(Q_vals)
    omega = np.diag(omegas)
    return P, Q, omega

# ---------- Mean–variance optimizer ----------

def mean_variance_opt(mu: np.ndarray, cov: np.ndarray, gamma: float = 2.5, long_only: bool = True) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - (gamma / 2.0) * cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        raise RuntimeError("Optimization failed to produce a solution.")
    # Clean & normalize (guard against tiny negatives due to solver tolerance)
    w_clipped = np.clip(w.value, 0, None)
    s = w_clipped.sum()
    return (w_clipped / s) if s > 0 else np.ones(n) / n

# ---------- High-level entry ----------

def optimize_allocation(
    risk_df: pd.DataFrame,
    opt_cfg: Dict
) -> pd.DataFrame:
    """
    Returns a DataFrame: [symbol, weight]
    """
    symbols, mu_sample, cov = prepare_moments(
        risk_df,
        ret_col=opt_cfg.get("ret_col", "ret"),
        lookback=opt_cfg.get("lookback", 252),
    )
    if not symbols:
        return pd.DataFrame(columns=["symbol", "weight"])

    # MVP market weights proxy
    mkt_type = opt_cfg.get("market_weights", "equal")
    if mkt_type == "equal":
        w_mkt = np.ones(len(symbols)) / len(symbols)
    else:
        w_mkt = np.ones(len(symbols)) / len(symbols)  # extend later

    gamma = float(opt_cfg.get("gamma", 2.5))
    tau = float(opt_cfg.get("tau", 0.05))

    pi = implied_equilibrium_returns(cov, w_mkt, gamma)
    P, Q, omega = build_views(symbols, opt_cfg.get("views"))

    if P.size > 0:
        mu_bl, Sigma_bl = black_litterman_posterior(pi, cov, P, Q, tau, omega)
    else:
        mu_bl, Sigma_bl = pi, cov

    w = mean_variance_opt(mu_bl, Sigma_bl, gamma=gamma, long_only=opt_cfg.get("long_only", True))
    return pd.DataFrame({"symbol": symbols, "weight": w})
