from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_equity(perf_df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    perf_df[["portfolio_equity", "benchmark_equity"]].dropna().plot(ax=ax)
    ax.set_title("Equity Curve: Portfolio vs Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (currency)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_drawdown(perf_df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    perf_df["drawdown"].dropna().plot(ax=ax)
    ax.set_title("Rolling Drawdown")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
