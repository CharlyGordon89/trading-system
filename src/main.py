from __future__ import annotations
import sys
from src.data_loader import ingest

def main(cfg_path: str = "config.yaml") -> None:
    dfs = ingest(cfg_path)
    assets, macro, merged = dfs["assets"], dfs["macro"], dfs["merged"]
    print(f"[OK] Assets: {len(assets):,} rows, {assets['symbol'].nunique() if not assets.empty else 0} symbols")
    print(f"[OK] Macro:  {0 if macro is None else len(macro):,} rows, {0 if macro is None else macro.shape[1]} series")
    print(f"[OK] Merged: {len(merged):,} rows, columns={list(merged.columns)}")

if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cfg)
