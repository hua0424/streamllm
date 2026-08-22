#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R7 §8.1：Fig.6 重绘（历史锁存策略的 post-feed residual TTFT + P5–P95 误差带）。

原图：main.tex fig:trend，展示 System A/B 随输入时长的归档残余延迟趋势。
数据源：R1 重算口径的逐样本数据（exp1_latency 归档结果，只读）。
重绘内容：每个模式按音频时长等频分箱（默认 12 箱），折线为箱内 mean post-feed residual TTFT，
阴影带为箱内 P5–P95（回应意见3 的分布披露；原图无分布信息）。

输出：experiments/results/revision/fig/Fig6.pdf + Fig6.png + 分箱数据 CSV。
用法：
  uv run python -m experiments.scripts.plot_fig6_trend
  uv run python -m experiments.scripts.plot_fig6_trend --self-test
"""

import argparse
import datetime
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

MODE_STYLE = {"non-streaming": ("System A (non-streaming)", "#c0392b", "--"),
              "streaming": ("System B (streaming)", "#2471a3", "-")}


def load_samples(results_glob: str) -> dict:
    files = sorted(glob.glob(results_glob))
    if not files:
        raise SystemExit(f"结果未找到: {results_glob}")
    data = json.loads(Path(files[-1]).read_text(encoding="utf-8"))
    samples = {}
    for r in data["results"]:
        if r.get("error"):
            continue
        samples.setdefault(r["mode"], []).append(
            (float(r["audio_duration"]), float(r["ttft"])))
    if len(samples) < 2:
        raise SystemExit("有效模式不足两个")
    logger.info(f"输入: {files[-1]}（{ {m: len(v) for m, v in samples.items()} }）")
    return samples


def bin_stats(durations, ttfts, n_bins: int) -> list:
    """等频分箱：每箱 (x_mean, y_mean, p5, p95, n)。"""
    order = np.argsort(durations)
    d = np.asarray(durations)[order]
    t = np.asarray(ttfts)[order]
    bins = []
    edges = np.linspace(0, len(d), n_bins + 1).astype(int)
    for i in range(n_bins):
        sd, st = d[edges[i]:edges[i + 1]], t[edges[i]:edges[i + 1]]
        if len(sd) == 0:
            continue
        bins.append({"x": float(sd.mean()), "mean": float(st.mean()),
                     "p5": float(np.percentile(st, 5)), "p95": float(np.percentile(st, 95)),
                     "n": len(sd)})
    return bins


def plot(samples: dict, n_bins: int, out_pdf: Path):
    import matplotlib
    matplotlib.use("Agg")
    # IEEE PDF eXpress 要求避免 bitmapped/Type 3 字体：以 TrueType 轮廓嵌入
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    # SVG 文字转路径：文件完全自包含，不依赖目标机器上的字体
    matplotlib.rcParams["svg.fonttype"] = "path"
    matplotlib.rcParams["svg.hashsalt"] = "streamllm-fig6-post-feed-v1"
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    all_bins = {}
    for mode in ("non-streaming", "streaming"):
        if mode not in samples:
            continue
        d, t = zip(*samples[mode])
        bins = bin_stats(np.array(d), np.array(t), n_bins)
        all_bins[mode] = bins
        label, color, ls = MODE_STYLE[mode]
        x = [b["x"] for b in bins]
        ax.plot(x, [b["mean"] for b in bins], ls, color=color, label=label, linewidth=1.6)
        ax.fill_between(x, [b["p5"] for b in bins], [b["p95"] for b in bins],
                        color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Input audio duration (s)")
    ax.set_ylabel("Post-feed residual TTFT (ms)")
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.99, 0.03, "Historical latched-trigger policy", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#555555")
    ax.grid(alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Creator": "StreamLLM deterministic Fig.6 renderer",
                "CreationDate": datetime.datetime(2026, 8, 22, tzinfo=datetime.timezone.utc),
                "ModDate": datetime.datetime(2026, 8, 22, tzinfo=datetime.timezone.utc)}
    fig.savefig(out_pdf, metadata=metadata)
    svg_path = out_pdf.with_suffix(".svg")
    fig.savefig(svg_path, metadata={"Creator": metadata["Creator"],
                                    "Date": "2026-08-22"})
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                        encoding="utf-8", newline="\n")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200)
    return all_bins


def self_test() -> int:
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not cond:
            failures += 1

    rng = np.random.default_rng(42)
    dur = rng.uniform(5, 60, 240)
    samples = {"non-streaming": list(zip(dur, 100 + dur * 90 + rng.normal(0, 100, 240))),
               "streaming": list(zip(dur, 900 + dur * 2 + rng.normal(0, 50, 240)))}
    bins = bin_stats(np.array(dur), np.array([s[1] for s in samples["streaming"]]), 12)
    check("分箱数量与覆盖", len(bins) == 12 and sum(b["n"] for b in bins) == 240)
    check("分箱统计有序", all(b["p5"] <= b["mean"] <= b["p95"] for b in bins))
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "fig" / "Fig6.pdf"
        all_bins = plot(samples, 12, out)
        check("图文件", out.exists() and out.with_suffix(".png").exists())
        check("双模式分箱", set(all_bins) == {"non-streaming", "streaming"})

    print(f"\nself-test {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="R7：Fig.6 重绘（P5–P95 误差带）")
    parser.add_argument("--results", type=str,
                        default=str(PROJECT_ROOT / "experiments/results/exp1_latency/exp1_results_*.json"))
    parser.add_argument("--n-bins", type=int, default=12)
    parser.add_argument("--out", type=str, default="experiments/results/revision/fig/Fig6.pdf")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_global_log_level(args.log_level)

    if args.self_test:
        sys.exit(self_test())

    samples = load_samples(args.results)
    out = PROJECT_ROOT / args.out
    all_bins = plot(samples, args.n_bins, out)

    import csv
    csv_path = out.with_suffix(".bins.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "bin_x_duration_s", "ttft_mean_ms", "ttft_p5_ms",
                         "ttft_p95_ms", "n"])
        for mode, bins in all_bins.items():
            for b in bins:
                writer.writerow([mode, f"{b['x']:.2f}", f"{b['mean']:.1f}",
                                 f"{b['p5']:.1f}", f"{b['p95']:.1f}", b["n"]])
    logger.info(f"已保存: {out}（+ .png / .bins.csv）")


if __name__ == "__main__":
    main()
