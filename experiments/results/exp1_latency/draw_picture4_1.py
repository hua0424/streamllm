#!/usr/bin/env python3
"""
Plot TTFT vs audio duration for Experiment 1.
- System A (non-streaming baseline): linear fit as theoretical linear growth (dashed).
- System B (streaming): bucketed mean of measured TTFT (solid).
"""

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and sanitize summary data."""
    df = pd.read_csv(csv_path)
    # keep valid numeric rows only
    df = df[df["ttft_ms"].notna()].copy()
    df = df[df["audio_duration"] > 0]
    df = df[df["ttft_ms"] > 0]
    if "error" in df.columns:
        df = df[df["error"].isna()]
    return df


def build_linear_fit(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a first-order line for System A as theoretical linear growth."""
    if df.empty:
        raise ValueError("Missing non-streaming data for linear fit.")
    x = df["audio_duration"].to_numpy()
    y = df["ttft_ms"].to_numpy()
    coef = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = coef[0] * x_line + coef[1]
    return x_line, y_line


def build_streaming_trend(
    df: pd.DataFrame, bins: int = 25, min_count: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Bucket and smooth System B measurements."""
    if df.empty:
        raise ValueError("Missing streaming data for measured curve.")
    x = df["audio_duration"]
    bin_edges = np.linspace(x.min(), x.max(), bins + 1)
    cats = pd.cut(x, bins=bin_edges, include_lowest=True)
    grouped = df.groupby(cats, observed=False)["ttft_ms"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= min_count]
    if grouped.empty:
        raise ValueError("Too few samples after bucketing to plot curve.")
    midpoints = grouped.index.map(lambda interval: interval.mid)
    return midpoints.to_numpy(), grouped["mean"].to_numpy()


def plot_ttft(summary_path: Path, output_path: Path) -> None:
    df = load_data(summary_path)
    streaming = df[df["mode"] == "streaming"].copy()
    non_streaming = df[df["mode"] == "non-streaming"].copy()

    x_line, y_line = build_linear_fit(non_streaming)
    x_stream, y_stream = build_streaming_trend(streaming)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8.5, 5.2))

    # sample scatter (faded to show distribution)
    plt.scatter(
        streaming["audio_duration"],
        streaming["ttft_ms"],
        s=12,
        alpha=0.18,
        color="#1f77b4",
        label="System B samples",
    )
    plt.scatter(
        non_streaming["audio_duration"],
        non_streaming["ttft_ms"],
        s=12,
        alpha=0.12,
        color="#ff7f0e",
        label="System A samples",
    )

    # smoothed curves
    plt.plot(
        x_stream,
        y_stream,
        color="#1f77b4",
        linewidth=2.2,
        label="System B bucketed mean",
    )
    plt.plot(
        x_line,
        y_line,
        color="#ff7f0e",
        linewidth=2.0,
        linestyle="--",
        label="System A linear fit (theoretical growth)",
    )

    plt.xlabel("Audio Duration (s)")
    plt.ylabel("Latency (ms)")
    plt.title("TTFT vs Audio Duration (Experiment 1)")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    summary_file = base_dir / "exp1_summary_20251210_024430.csv"
    output_file = base_dir / "exp1_latency_ttft.png"
    plot_ttft(summary_file, output_file)

