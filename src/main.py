from __future__ import annotations
import sys
import logging
import yaml
import pandas as pd
from src.data_loader import ingest
from src.risk_model import compute_risk_metrics
from src.utils_io import to_parquet
from src.optimizer.bl import optimize_allocation
from src.backtest import run_backtest
from src.dashboard import plot_equity, plot_drawdown

logger = logging.getLogger(__name__)

def _load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _save_placeholder_plots(cfg: dict, msg: str) -> None:
    perf_path = cfg.get("perf_plot_path", "data/perf_equity_vs_benchmark.png")
    dd_path = cfg.get("dd_plot_path", "data/rolling_drawdown.png")
    # minimal placeholder dataframe (so plotting code works)
    dates = pd.date_range("2000-01-01", periods=2, freq="D")
    df = pd.DataFrame(
        {
            "portfolio_equity": [1.0, 1.0],
            "benchmark_equity": [1.0, 1.0],
            "drawdown": [0.0, 0.0],
        },
        index=dates,
    )
    logger.warning("%s — writing placeholder plots.", msg)
    plot_equity(df, perf_path)
    plot_drawdown(df, dd_path)
    logger.info("Placeholder plots saved: %s | %s", perf_path, dd_path)

def main(cfg_path: str = "config.yaml") -> None:
    cfg = _load_cfg(cfg_path)
    dfs = ingest(cfg_path)
    assets, macro, merged = dfs["assets"], dfs["macro"], dfs["merged"]

    # --- Risk (MVP) ---
    rparams = cfg.get("risk", {})
    base_for_risk = merged[["symbol", "close", "volume"]].copy() if not merged.empty else merged
    risk_df = compute_risk_metrics(
        assets=base_for_risk,
        vol_window=rparams.get("vol_window", 20),
        var_window=rparams.get("var_window", 60),
        cvar_alpha=rparams.get("cvar_alpha", 0.05),
        liq_window=rparams.get("liq_window", 20),
        ret_col=rparams.get("ret_col", "ret"),
    )
    to_parquet(risk_df, cfg.get("risk_parquet", "data/risk.parquet"))

    # --- Allocation snapshot (existing) ---
    alloc_cfg = {"ret_col": rparams.get("ret_col", "ret"), **cfg.get("optimizer", {})}
    weights_df = optimize_allocation(risk_df=risk_df, opt_cfg=alloc_cfg)
    to_parquet(weights_df, cfg.get("weights_parquet", "data/weights_latest.parquet"))

    # --- Backtest + Plots (defensive) ---
    bt_cfg = cfg.get("backtest", {})
    try:
        perf_df, w_hist_df = run_backtest(risk_df, optimizer_cfg=alloc_cfg, backtest_cfg=bt_cfg)
        to_parquet(perf_df, cfg.get("backtest_parquet", "data/backtest_results.parquet"))
        # Debug info to confirm plotting inputs
        logger.debug(
            "Backtest rows: %d, date range: %s → %s",
            len(perf_df),
            perf_df.index.min(),
            perf_df.index.max(),
        )
        plot_equity(perf_df, cfg.get("perf_plot_path", "data/perf_equity_vs_benchmark.png"))
        plot_drawdown(perf_df, cfg.get("dd_plot_path", "data/rolling_drawdown.png"))
        logger.info(
            "Plots saved: %s | %s",
            cfg.get("perf_plot_path"),
            cfg.get("dd_plot_path"),
        )
    except Exception as e:
        _save_placeholder_plots(cfg, f"Backtest failed: {e}")

    # --- Console summary ---
    logger.info("Assets: %,d rows", len(assets))
    logger.info("Risk:   %,d rows", len(risk_df))
    if not weights_df.empty:
        logger.info(
            "Latest weights:\n%s",
            weights_df.sort_values("weight", ascending=False).to_string(index=False),
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg)
