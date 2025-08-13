import pandas as pd
from src.data_loader import merge_assets_macro

def test_merge_assets_macro_schema():
    # minimal synthetic frames to avoid network in tests
    assets = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "close": [100.0, 101.0, 50.0],
            "volume": [1000, 1100, 2000],
        },
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"]),
    )
    macro = pd.DataFrame(
        {"CPI": [300.0, 300.1]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"])
    )
    merged = merge_assets_macro(assets, macro)
    assert set(["symbol", "close", "volume", "CPI"]).issubset(merged.columns)
    assert merged.loc["2024-01-02"].shape[0] >= 1
