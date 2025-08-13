from __future__ import annotations
import sys
import yaml
from src.data_loader import ingest
from src.risk_model import compute_risk_metrics
from src.utils_io import to_parquet

def _load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main(cfg_path: str = "config.yaml") -> None:
    cfg = _load_cfg(cfg_path)
    dfs = ingest(cfg_path)
    assets, macro, merged = dfs["assets"], dfs["macro"], dfs["merged"]

    # --- Risk (MVP) ---
    rparams = cfg.get("risk", {})
    risk_df = compute_risk_metrics(
        assets=merged[["symbol", "close", "volume"]].copy() if not merged.empty else merged,
        vol_window=rparams.get("vol_window", 20),
        var_window=rparams.get("var_window", 60),
        cvar_alpha=rparams.get("cvar_alpha", 0.05),
        liq_window=rparams.get("liq_window", 20),
        ret_col=rparams.get("ret_col", "ret"),
    )
    to_parquet(risk_df, cfg.get("risk_parquet", "data/risk.parquet"))

    print(f"[OK] Assets: {len(assets):,} rows, {assets['symbol'].nunique() if not assets.empty else 0} symbols")
    print(f"[OK] Macro:  {0 if macro is None else len(macro):,} rows, {0 if macro is None else macro.shape[1]} series")
    print(f"[OK] Merged: {len(merged):,} rows")
    print(f"[OK] Risk:   {len(risk_df):,} rows, cols={list(risk_df.columns)}")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg)
