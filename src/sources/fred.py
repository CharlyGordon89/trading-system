from __future__ import annotations
import os
import pandas as pd
from typing import List, Dict
from fredapi import Fred

def download_macro(series_ids: List[str], start_date: str) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by 'date' with columns = series_ids.
    If FRED_API_KEY is missing or a series fails, it will skip that series gracefully.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        # No key — return empty frame but with proper index if possible
        return pd.DataFrame(columns=series_ids)

    fred = Fred(api_key=api_key)
    frames: Dict[str, pd.Series] = {}
    for sid in series_ids:
        try:
            s = fred.get_series(sid)
            s = s[s.index >= pd.to_datetime(start_date)]
            s.index = pd.to_datetime(s.index)
            frames[sid] = s.rename(sid)
        except Exception:
            # skip broken series
            continue

    if not frames:
        return pd.DataFrame(columns=series_ids)

    df = pd.concat(frames.values(), axis=1)
    df.index.name = "date"
    return df.sort_index()
