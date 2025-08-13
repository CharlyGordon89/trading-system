from __future__ import annotations
import sys
import yaml
from src.data_loader import ingest
from src.risk_model import compute_risk_metrics
from src.utils_io import to_parquet
from src.optimizer.bl import optimize_allocation

def _load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

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

    # --- Allocation (MVP Black–Litterman) ---
    alloc_cfg = {"ret_col": rparams.get("ret_col", "ret"), **cfg.get("optimizer", {})}
    weights_df = optimize_allocation(risk_df=risk_df, opt_cfg=alloc_cfg)
    to_parquet(weights_df, cfg.get("weights_parquet", "data/weights_latest.parquet"))

    # --- Console summary ---
    print(f"[OK] Assets: {len(assets):,} rows, {assets['symbol'].nunique() if not assets.empty else 0} symbols")
    print(f"[OK] Macro:  {len(macro):,} rows, {macro.shape[1] if not macro.empty else 0} series")
    print(f"[OK] Merged: {len(merged):,} rows")
    print(f"[OK] Risk:   {len(risk_df):,} rows, cols={list(risk_df.columns)}")
    if not weights_df.empty:
        print(f"[OK] Weights ({len(weights_df)}):")
        print(weights_df.sort_values('weight', ascending=False).to_string(index=False))
    else:
        print("[WARN] No weights produced (insufficient data).")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg)
