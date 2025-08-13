from __future__ import annotations
import os
import pandas as pd
from typing import Dict, List
from dotenv import load_dotenv
import yaml

from src.utils_io import to_parquet
from src.sources.yahoo import download_assets
from src.sources.fred import download_macro

def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_symbol_list(cfg: Dict) -> List[str]:
    return list(dict.fromkeys((cfg.get("equities", []) + cfg.get("bonds", []) + cfg.get("crypto", []))))

def ingest(cfg_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    load_dotenv()  # enables FRED_API_KEY if provided

    cfg = load_config(cfg_path)
    start_date = cfg["start_date"]
    interval = cfg.get("interval", "1d")

    # 1) Assets
    tickers = build_symbol_list(cfg)
    assets = download_assets(tickers, start_date=start_date, interval=interval)
    # Persist individual assets parquet
    to_parquet(assets, cfg["assets_parquet"])

    # 2) Macro
    macro_series = cfg.get("macro", [])
    macro = download_macro(macro_series, start_date=start_date)
    if not macro.empty:
        # Forward-fill macro to align with daily market days
        macro = macro.sort_index().ffill()
        to_parquet(macro, cfg["macro_parquet"])

    # 3) Merge (left join assets on macro by date)
    merged = merge_assets_macro(assets, macro)
    to_parquet(merged, cfg["merged_parquet"])

    return {"assets": assets, "macro": macro, "merged": merged}

def merge_assets_macro(assets: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """
    assets: index=date, columns=[symbol, close, volume]
    macro:  index=date, columns=macro series
    """
    if assets.empty:
        return assets  # nothing to do

    out = assets.copy()
    if not macro.empty:
        out = out.join(macro, how="left").ffill()
    return out.sort_index()
