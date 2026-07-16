#!/usr/bin/env python3
"""
Generate the SVG vector figure for Experiment 1 TTFT vs audio duration.
- System A (non-streaming baseline): linear fit as theoretical linear growth (dashed).
- System B (streaming): bucketed mean of measured TTFT (solid).
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_SUMMARY_FILE = "exp1_summary_20251210_024430.csv"
DEFAULT_OUTPUT_FILE = "exp1_latency_ttft.svg"


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
    matplotlib.rcParams["svg.fonttype"] = "path"
    # 期刊(UAIS)要求: 图内文字最终印刷尺寸 8-12pt。本图按单栏宽约 84mm(3.3in) 排版,
    # 画布 6in 宽 -> 缩放约 0.55, 源字号 15-17pt -> 印刷后约 8-9.5pt。
    matplotlib.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
    })
    fig, ax = plt.subplots(figsize=(6.0, 3.8))

    # sample scatter (faded to show distribution)
    ax.scatter(
        streaming["audio_duration"],
        streaming["ttft_ms"],
        s=12,
        alpha=0.18,
        color="#1f77b4",
        label="System B samples",
    )
    ax.scatter(
        non_streaming["audio_duration"],
        non_streaming["ttft_ms"],
        s=12,
        alpha=0.12,
        color="#ff7f0e",
        label="System A samples",
    )

    # smoothed curves
    ax.plot(
        x_stream,
        y_stream,
        color="#1f77b4",
        linewidth=2.2,
        label="System B bucketed mean",
    )
    ax.plot(
        x_line,
        y_line,
        color="#ff7f0e",
        linewidth=2.0,
        linestyle="--",
        label="System A linear fit",
    )

    ax.set_xlabel("Audio Duration (s)")
    ax.set_ylabel("Latency (ms)")
    # 固定刻度密度(横轴每20s, 纵轴每2000ms), 避免画布缩小后刻度过疏
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2000))
    # 期刊要求图内不放标题(标题写在正文 caption 中), 故不调用 set_title
    # 图例移到绘图区上方(两列), 避免遮挡数据曲线
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate the Experiment 1 TTFT figure as an SVG vector image."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=base_dir / DEFAULT_SUMMARY_FILE,
        help="Path to the Experiment 1 summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / DEFAULT_OUTPUT_FILE,
        help="Path to the output SVG file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_ttft(args.summary, args.output)

