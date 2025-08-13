from __future__ import annotations
import numpy as np
import pandas as pd
import cvxpy as cp
from typing import List, Dict, Tuple

# ============================================================
# Helpers: PSD guard and safe covariance handling
# ============================================================

def _make_psd(cov: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """
    Ensure covariance matrix is positive semidefinite by clipping
    tiny negative eigenvalues caused by numerical noise.
    """
    if cov.size == 0:
        return cov
    cov = (cov + cov.T) / 2.0  # symmetrize
    vals, vecs = np.linalg.eigh(cov)
    vals_clipped = np.clip(vals, a_min=eps, a_max=None)
    cov_psd = (vecs @ np.diag(vals_clipped) @ vecs.T)
    # Re-symmetrize to reduce rounding error
    return (cov_psd + cov_psd.T) / 2.0

# ============================================================
# Moments preparation from risk dataframe
# ============================================================

def prepare_moments(
    risk_df: pd.DataFrame,
    ret_col: str = "ret",
    lookback: int = 252,
    shrinkage: float = 0.10,  # identity-target shrinkage (0.05–0.15 recommended)
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    risk_df: index=date, columns include ['symbol', ret_col]
    Returns: (symbols, mean_returns (daily), cov_matrix (daily))
    """
    if risk_df is None or risk_df.empty:
        return [], np.array([]), np.array([[]])

    # Pivot to (date x symbol) returns matrix
    df = risk_df.dropna(subset=[ret_col])
    mat = df.pivot_table(index=df.index, columns="symbol", values=ret_col).dropna(how="all")
    if mat.empty:
        return [], np.array([]), np.array([[]])

    # Restrict to lookback window, keep columns with data
    window = min(len(mat), lookback)
    mat = (mat.tail(window)
              .dropna(axis=1, how="all")
              .fillna(0.0))  # MVP: simple fill; can enhance later
    if mat.shape[1] == 0:
        return [], np.array([]), np.array([[]])

    symbols = mat.columns.tolist()
    mu = mat.mean(axis=0).to_numpy()                  # daily mean returns
    raw = mat.to_numpy()
    cov = np.cov(raw, rowvar=False)                   # daily covariance

    # Identity-target shrinkage for conditioning
    shrinkage = float(max(0.0, min(1.0, shrinkage)))
    I = np.eye(cov.shape[0])
    cov = (1.0 - shrinkage) * cov + shrinkage * I

    # Final PSD guard
    cov = _make_psd(cov)
    return symbols, mu, cov

# ============================================================
# Black–Litterman core
# ============================================================

def implied_equilibrium_returns(cov: np.ndarray, w_mkt: np.ndarray, risk_aversion: float) -> np.ndarray:
    # π = δ Σ w_mkt
    return float(risk_aversion) * cov.dot(w_mkt)

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
    omega: views uncertainty (k x k), typically diagonal
    """
    if P.size == 0:  # no views ⇒ posterior == prior
        return pi, cov

    # Compute BL posterior using common closed form:
    # μ_BL = A^{-1} b  with
    #   A = (τΣ)^{-1} + Pᵀ Ω^{-1} P
    #   b = (τΣ)^{-1} π + Pᵀ Ω^{-1} Q
    # Σ_BL ≈ Σ + A^{-1}
    tau = float(tau)
    Sigma = _make_psd(cov)

    # Inverse of τΣ (after PSD guard)
    inv_tauSigma = np.linalg.inv(tau * Sigma)

    # Ω^{-1} (Ω is diagonal under our construction)
    inv_omega = np.linalg.inv(omega)

    A = inv_tauSigma + P.T @ inv_omega @ P
    b = inv_tauSigma @ pi + P.T @ inv_omega @ Q

    # Solve A x = b for posterior mean
    mu_bl = np.linalg.solve(A, b)

    # Posterior covariance (common form)
    post_cov_part = np.linalg.inv(A)
    Sigma_bl = Sigma + post_cov_part
    Sigma_bl = _make_psd(Sigma_bl)
    return mu_bl, Sigma_bl

# ============================================================
# Views helpers
# ============================================================

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
    ann = 252.0

    for v in views_cfg:
        vtype = v.get("type", "relative")
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

# ============================================================
# Mean–variance optimizer with safe fallback
# ============================================================

def mean_variance_opt(
    mu: np.ndarray,
    cov: np.ndarray,
    gamma: float = 2.5,
    long_only: bool = True,
    enable_fallback: bool = True,
) -> np.ndarray:
    """
    Solve max_w  μᵀw − (γ/2) wᵀΣw
      s.t. ∑ w = 1, and optionally w ≥ 0

    On failure (or infeasibility), return equal weights if enable_fallback=True.
    """
    n = len(mu)
    if n == 0:
        return np.array([])

    # Final guards
    cov = _make_psd(cov)
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)

    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - (float(gamma) / 2.0) * cp.quad_form(w, cov))
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)

    try:
        prob = cp.Problem(objective, constraints)
        # Try a few solvers commonly available; stop at first success
        for solver in (cp.OSQP, cp.ECOS, cp.SCS):
            try:
                prob.solve(solver=solver, verbose=False)
                if w.value is not None and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    break
            except Exception:
                continue

        if w.value is None:
            raise RuntimeError(f"Optimization failed (status={prob.status})")

        w_clipped = np.clip(w.value, 0, None) if long_only else w.value
        s = np.sum(w_clipped)
        if not np.isfinite(s) or s <= 0:
            raise RuntimeError("Non-finite or non-positive weight sum.")
        return w_clipped / s

    except Exception as e:
        if enable_fallback:
            print(f"[WARN] Optimizer failed ({e}); falling back to equal weights.")
            return np.ones(n) / n
        raise

# ============================================================
# High-level entry
# ============================================================

def optimize_allocation(
    risk_df: pd.DataFrame,
    opt_cfg: Dict
) -> pd.DataFrame:
    """
    Returns a DataFrame: [symbol, weight]
    Config keys used:
      ret_col (str), lookback (int), shrinkage (float), gamma (float),
      tau (float), long_only (bool), enable_equal_weight_fallback (bool),
      market_weights ("equal" supported), views (list of dicts).
    """
    # Defaults
    ret_col = opt_cfg.get("ret_col", "ret")
    lookback = int(opt_cfg.get("lookback", 252))
    shrinkage = float(opt_cfg.get("shrinkage", 0.10))
    gamma = float(opt_cfg.get("gamma", 2.5))
    tau = float(opt_cfg.get("tau", 0.05))
    long_only = bool(opt_cfg.get("long_only", True))
    enable_fb = bool(opt_cfg.get("enable_equal_weight_fallback", True))

    # Moments
    symbols, mu_sample, cov = prepare_moments(
        risk_df,
        ret_col=ret_col,
        lookback=lookback,
        shrinkage=shrinkage
    )
    if not symbols:
        return pd.DataFrame(columns=["symbol", "weight"])

    # Market weights proxy (MVP: equal)
    mkt_type = opt_cfg.get("market_weights", "equal")
    if mkt_type == "equal":
        w_mkt = np.ones(len(symbols)) / len(symbols)
    else:
        # Extend with cap-weight or volume-weight later
        w_mkt = np.ones(len(symbols)) / len(symbols)

    # Prior (equilibrium) returns
    pi = implied_equilibrium_returns(cov, w_mkt, gamma)

    # Views → posterior
    P, Q, omega = build_views(symbols, opt_cfg.get("views"))
    if P.size > 0:
        mu_bl, Sigma_bl = black_litterman_posterior(pi, cov, P, Q, tau, omega)
        Sigma_bl = _make_psd(Sigma_bl)
    else:
        mu_bl, Sigma_bl = pi, cov

    # Optimize
    w = mean_variance_opt(
        mu_bl, Sigma_bl,
        gamma=gamma,
        long_only=long_only,
        enable_fallback=enable_fb
    )
    return pd.DataFrame({"symbol": symbols, "weight": w})
