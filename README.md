# Adaptive Multi-Asset Portfolio Optimization System (MVP)

A small, production-lean codebase that ingests multi-asset market & macro data, computes risk metrics (volatility, VaR/CVaR, liquidity), allocates with a minimal **Black–Litterman** optimizer, and runs a **monthly-rebalanced backtest** with a simple dashboard (equity vs. benchmark + drawdown).

Built to be **working first**, **easy to extend**, and **well-tested**.

---

## Highlights

- **Data Fusion (Sprint 1):** Equities/Bonds/Crypto from Yahoo Finance; optional macro from FRED; tidy Parquet outputs.
- **Risk (Sprint 2):** Rolling returns & volatility; historical **VaR/CVaR(95%)**; simple liquidity score.
- **Optimizer (Sprint 3):** Minimal **Black–Litterman** with relative/absolute views; long-only mean–variance weights via CVXPY.
- **Backtest & Dashboard (Sprint 4):** Monthly rebalancing, turnover costs (bps), SPY benchmark, equity/drawdown plots with defensive fallbacks.

---

## Repo Structure

```
trading-system/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example                 # optional FRED key
├── pytest.ini                   # makes src/ importable in tests
├── config.yaml
├── data/                        # generated artifacts (parquet/png)
├── src/
│   ├── main.py                  # end-to-end runner (ingest→risk→opt→bt→plots)
│   ├── utils_io.py
│   ├── data_loader.py
│   ├── risk_model.py
│   ├── backtest.py
│   ├── dashboard.py
│   ├── optimizer/
│   │   ├── __init__.py
│   │   └── bl.py
│   └── sources/
│       ├── __init__.py
│       ├── yahoo.py
│       └── fred.py
└── tests/
    ├── test_merge_shapes.py
    ├── test_risk_basic.py
    └── test_optimizer_bl.py
```

---

## Quickstart

```bash
# 1) create env
python -m venv .venv && source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) optional: FRED macro (CPI, yield curve)
cp .env.example .env   # add FRED_API_KEY if you have it

# 4) run tests (offline, synthetic)
pytest -q

# 5) run pipeline (downloads from Yahoo; writes parquet & png)
python -m src.main
```

**Outputs (by default):**
- Data:  
  `data/assets.parquet`, `data/macro.parquet`, `data/merged.parquet`, `data/risk.parquet`, `data/weights_latest.parquet`, `data/backtest_results.parquet`
- Plots:  
  `data/perf_equity_vs_benchmark.png`, `data/rolling_drawdown.png`

> Plots are **guaranteed** to be created: if the backtest can’t run (e.g., not enough data), placeholder plots are saved so your dashboard step always works.

---

## Configuration

`config.yaml` (MVP defaults):

```yaml
# Assets
equities: [VOO, QQQ]
bonds: [TLT]
crypto: [BTC-USD]

macro: [CPIAUCSL, T10Y3M]   # optional (requires FRED_API_KEY)
start_date: "2018-01-01"
interval: "1d"

# Outputs
data_dir: "data"
assets_parquet: "data/assets.parquet"
macro_parquet: "data/macro.parquet"
merged_parquet: "data/merged.parquet"
risk_parquet: "data/risk.parquet"
backtest_parquet: "data/backtest_results.parquet"
perf_plot_path: "data/perf_equity_vs_benchmark.png"
dd_plot_path: "data/rolling_drawdown.png"

# Risk parameters
risk:
  ret_col: "ret"
  vol_window: 20
  var_window: 60
  cvar_alpha: 0.05
  liq_window: 20

# Black–Litterman optimizer
optimizer:
  lookback: 252
  gamma: 2.5
  tau: 0.05
  long_only: true
  market_weights: equal
  views:
    - type: relative
      long: "BTC-USD"
      short: "TLT"
      annualized_delta: 0.05
      uncertainty: 0.10

# Backtest
backtest:
  rebalance: "M"
  initial_capital: 500000
  tc_bps: 5
  benchmark: "SPY"
  lookback: 252
```

**Secrets:** put `FRED_API_KEY=...` into `.env`. Macro is optional; the system runs without it.

---

## Data Sources

- **Market data:** Yahoo Finance via `yfinance` (VOO, QQQ, TLT, BTC-USD by default).
- **Macro (optional):** FRED (e.g., CPIAUCSL, T10Y3M). Missing key ⇒ macro quietly skipped.

All market data are normalized to a **tidy** frame:  
`index=date`, `columns=["symbol", "close", "volume", <macro columns…>]`.

---

## How It Works (End-to-End)

### 1) Ingestion (`src/data_loader.py`, `src/sources/*`)
1. Download daily bars (auto-adjusted close & volume) for configured tickers.
2. Optionally fetch macro series from FRED.
3. Forward-fill macro to align to market days.
4. Save **assets.parquet**, **macro.parquet**, and **merged.parquet**.

### 2) Risk Metrics (`src/risk_model.py`)
For each symbol:
- **Returns:** `ret_t = close_t / close_{t-1} - 1`.
- **Volatility:** rolling std of returns over `vol_window`.
- **Historical VaR (alpha):** empirical quantile of returns over `var_window`.  
  Example (alpha = 0.05): `VaR_95 = quantile(ret, 0.05)` (a negative number).
- **Historical CVaR:** mean of the worst `alpha` tail:  
  `CVaR_95 = mean({ret_i | ret_i <= VaR_95})` (≤ VaR_95).
- **Liquidity Score:** rolling mean of volume (`liq_window`).

Outputs to **risk.parquet**.

### 3) Black–Litterman Optimizer (`src/optimizer/bl.py`)
- Sample **moments** from recent returns (lookback):
  - `μ` = column means (daily), `Σ` = covariance.
- **Market equilibrium returns** (prior):
  - `π = δ Σ w_mkt`  
    where `δ` = risk aversion (`gamma`), `w_mkt` = market weights (MVP: equal).
- **Views** (optional):
  - **Relative:** `r_long - r_short = Δ`  
  - **Absolute:** `r_symbol = μ*`  
  Uncertainty is diagonal `Ω`; both expressed as **annualized** in config, converted to daily.
- **Posterior (BL):**
  - `μ_BL = [ (τΣ)^(-1) + Pᵀ Ω^{-1} P ]^{-1} [ (τΣ)^(-1) π + Pᵀ Ω^{-1} Q ]`
  - `Σ_BL ≈ Σ + [ (τΣ)^(-1) + Pᵀ Ω^{-1} P ]^{-1}` (common form)
- **Allocation:** mean–variance optimization (CVXPY):
  - maximize `μ_BLᵀ w − (γ/2) wᵀ Σ_BL w`
  - s.t. `∑ w = 1`, `w ≥ 0` (long-only in MVP)

Outputs **weights_latest.parquet** and prints a table.

### 4) Backtest & Dashboard (`src/backtest.py`, `src/dashboard.py`)
- **Monthly rebalancing** at month-ends.
- At each rebalance:
  - Fit optimizer using *past* `lookback` window only (no look-ahead).
  - Apply turnover **transaction cost**: `cost = (tc_bps/10000) * turnover * portfolio_value`.
- Daily portfolio returns = weighted sum of symbol returns; equity path updated.
- **Benchmark:** SPY daily returns (Yahoo) for the same period; equity compared.
- **Drawdown:** `DD_t = equity_t / max_{≤t}(equity) − 1`.
- Plots:
  - `perf_equity_vs_benchmark.png`
  - `rolling_drawdown.png`  
  (If backtest fails, **placeholder** plots are saved so the dashboard step never breaks.)

---

## Testing

All tests are **network-free** (synthetic data) and run fast:

```bash
pytest -q
```

- `test_merge_shapes.py`: schema & merge sanity.
- `test_risk_basic.py`: CVaR ≤ VaR on tail, columns present.
- `test_optimizer_bl.py`: weights sum to 1, non-negative, basic sanity.
- `test_backtest_basic.py`: backtest produces perf & weight history on toy data.

`pytest.ini` sets `pythonpath=.` so imports like `from src...` work.

---

## Troubleshooting

- **Plots missing:** We now plot with explicit axes and create **placeholder** PNGs on any failure. Check console lines:
  - `[DEBUG] Backtest rows: ...`
  - `[OK] Plots saved: data/perf_equity_vs_benchmark.png | data/rolling_drawdown.png`
- **“Not enough data to run the backtest”**: Extend history (`start_date` earlier) or lower `backtest.lookback`.
- **FRED macro empty:** Provide `FRED_API_KEY` (optional). Without it, macro is skipped gracefully.
- **CVXPY solver warnings:** SCS is used by default for simplicity. You can install OSQP or ECOS and switch.
- **Mixed Python versions:** We recommend Python ≥3.10. If you’re on 3.13 and see dependency issues, try 3.10–3.12.

---

## Design Principles

- **Keep it working & small first**: minimal files and dependencies; each module focused and testable.
- **Extensible**: swap Parquet for DB; change optimizer; add regimes; all without rewriting core.
- **Deterministic & Transparent**: straightforward equations, clear config, simple tests.
- **Defensive UX**: dashboard never silently fails (placeholder plots if needed).

---

## Extending the MVP (Suggested Phase-2 Roadmap)

- **Data/Infra:** Docker Compose (Redpanda + TimescaleDB + Grafana); Prometheus metrics; retries/caching.
- **Risk/ML:** Online CVaR (River), factor exposures, regime detection (HMM), auto-tuned `gamma`/`tau`.
- **Optimizer:** Turnover/vol targeting constraints, tax-aware rebalancing, views from signals (momentum/value).
- **Reporting:** Performance attribution, PDF reports, CI/CD, chaos tests, FIX gateway simulator.

---

## MVP Sprint Log (Commits)

- **Sprint 1**  
  `feat(data): basic multi-asset ingestion (equities, bonds, crypto, macro) → parquet`
- **Sprint 2**  
  `feat(risk): MVP CVaR + volatility + liquidity scoring`
- **Sprint 3**  
  `feat(optimizer): MVP Black–Litterman allocation engine`
- **Sprint 4**  
  `feat(backtest): monthly-rebalanced BL backtest + simple dashboard`

(Plus small fixes like `fix(dashboard): ensure plots are always produced and visible`.)

---

## Performance & Limits (MVP)

- **Assumptions:** long-only; equal market weights proxy; simple transaction costs; fill-na(0) on returns matrix for stability.
- **Backtest granularity:** daily bars, monthly rebalancing.
- **Risk metrics:** historical simulation (no parametric tails yet).
- **No production infra yet:** Parquet instead of databases; single-machine runs.

---

## Acknowledgments

- Yahoo Finance via `yfinance`
- Federal Reserve Economic Data (FRED)


