#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
软触发置信度分布标定 → 产出 E2 阈值扫描点（换模型后必跑，见 handoff §四.2）。

背景：E2 的 DEFAULT_THRESHOLDS 是按验证机替身（prompted Qwen2.5-0.5B）的置信度
分布定的；换 TEN 7B 后分布不同，直接沿用会让 trade-off 曲线失真。
本脚本对内置完整句/不完整句集打印两类分布，并给出建议扫描点：
  取 [不完整类 P25 .. 完整类 P90] 区间内的等分位点 + 一个"永不触发"哨兵(1.1)。

运行：
    # 验证机（替身，验证脚本本身）
    HF_TOKEN= uv run python -m experiments.scripts.calibrate_trigger
    # 实验机（TEN 7B）
    HF_TOKEN= uv run python -m experiments.scripts.calibrate_trigger --config ten
产出 JSON 的 suggested_thresholds 直接传给 run_exp2_tradeoff 的 --thresholds。
"""

import argparse
import json
from pathlib import Path
from statistics import quantiles

from src.dialogue.trigger import LLMSoftTrigger, QWEN05B_DEV_CONFIG, TEN_CONFIG
from src.utils.logging_utils import get_logger, set_global_log_level
from src.config import RESULTS_DIR

logger = get_logger(__name__)

COMPLETE = [
    "What's the weather like in Beijing today?",
    "Book me a flight to Shanghai tomorrow morning.",
    "Tell me about the Great Wall of China.",
    "How much does the museum ticket cost?",
    "I'd like to cancel my hotel reservation for Friday.",
    "Find me a cheap restaurant in the city centre.",
    "I need a train to Cambridge arriving by noon.",
    "Can you recommend a museum near the river?",
]
INCOMPLETE = [
    "I want to", "Could you tell me about the", "So um, what I was thinking is",
    "Book me a flight to", "What's the", "I need a hotel in the",
    "Find me a train to Cambridge on Sunday, and", "The restaurant should serve",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["dev", "ten"], default="dev",
                    help="dev=替身 Qwen0.5B（验证机）；ten=TEN 7B（实验机）")
    ap.add_argument("--n-points", type=int, default=6, help="建议扫描点数（另加 1.1 哨兵）")
    ap.add_argument("--out", type=str, default=str(Path(RESULTS_DIR) / "trigger_calibration.json"))
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    cfg = TEN_CONFIG if args.config == "ten" else QWEN05B_DEV_CONFIG
    logger.info(f"标定 soft-trigger：{cfg.model_name}")
    trig = LLMSoftTrigger(cfg)

    comp = sorted(trig.confidence(t) for t in COMPLETE)
    incomp = sorted(trig.confidence(t) for t in INCOMPLETE)
    q = lambda xs, k: quantiles(xs, n=100)[k - 1]   # P_k
    logger.info(f"complete  : min={comp[0]:.4f} P50={q(comp,50):.4f} P90={q(comp,90):.4f} max={comp[-1]:.4f}")
    logger.info(f"incomplete: min={incomp[0]:.4f} P25={q(incomp,25):.4f} P50={q(incomp,50):.4f} max={incomp[-1]:.4f}")

    # AUC（成对正确率）——可分性体检；过低说明该模型/prompt 不适合当软触发
    pairs = [(c, i) for c in comp for i in incomp]
    auc = sum(1 for c, i in pairs if c > i) / len(pairs)
    logger.info(f"AUC~ = {auc:.2f}")

    lo, hi = q(incomp, 25), max(q(comp, 90), q(incomp, 25) + 1e-4)
    pts = [round(lo + (hi - lo) * k / (args.n_points - 1), 4) for k in range(args.n_points)]
    suggested = sorted(set(pts)) + [1.1]    # 1.1 = 永不触发哨兵（保守极限基线）
    logger.info(f"建议 --thresholds: {' '.join(str(p) for p in suggested)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": cfg.model_name, "auc_pairwise": round(auc, 3),
        "complete_confs": [round(x, 4) for x in comp],
        "incomplete_confs": [round(x, 4) for x in incomp],
        "suggested_thresholds": suggested,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"已保存: {out}")

    assert auc >= 0.65, f"可分性过低（AUC {auc:.2f}）——检查模型/prompt 再跑 E2"
    logger.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
