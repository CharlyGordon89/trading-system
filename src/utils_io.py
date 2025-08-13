from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def to_parquet(df: pd.DataFrame, path: str | os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    df.to_parquet(path, index=True)

def read_parquet(path: str | os.PathLike) -> pd.DataFrame:
    return pd.read_parquet(path)
