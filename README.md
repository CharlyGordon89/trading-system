# Adaptive Multi-Asset Portfolio – Sprint 1 (Data Fusion Engine)

## What this does
- Downloads daily prices & volume for Equities/Bonds/Crypto via Yahoo Finance.
- Optionally downloads FRED macro series (if `FRED_API_KEY` is provided).
- Saves tidy `parquet` files: assets, macro, and merged.
- Keeps code minimal and extensible.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional; add FRED_API_KEY if you have one
python -m src.main
