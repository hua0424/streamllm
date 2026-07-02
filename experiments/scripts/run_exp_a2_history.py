#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融 A2：被打断轮的三种历史处理策略（朴素截断 / 标记法 / 重写法）—— harness。

设计（experiment_design.md §5 A2 / paper2_context.md §2.3 贡献3）：
同一对话、同一打断点（mid-fragment，触发 partial），三策略各跑：
  - naive  ：截断后不作处理（历史是半句话）
  - mark   ：追加打断标记 " …"（零延迟零模型成本）
  - rewrite：Qwen3-0.6B 把半句话改写为自然收尾（不新增信息；架构上并行可隐藏，
             此处记录 rewrite_ms 验证可隐藏性：应 << 用户打断后说话时长 ~1s+）

本机产出：三策略历史文本样例 + 下一轮回复 + rewrite 延迟统计（LLM-judge 连贯性
评分留实验机，JSON 里预留 judge 字段）。

运行（项目根目录）：
    HF_TOKEN= uv run python -m experiments.scripts.run_exp_a2_history
"""

import argparse
import json
from pathlib import Path

from src.dialogue.orchestrator import DialogueOrchestrator
from src.dialogue.rewriter import HistoryRewriter
from src.llm.stream_llm_inference import StreamLLMInference
from src.tts.streaming_tts import MockStreamingTTS, TimingProfile
from src.utils.logging_utils import get_logger, set_global_log_level
from src.config import RESULTS_DIR

logger = get_logger(__name__)

POLICIES = ["naive", "mark", "rewrite"]

FIXTURE = [
    {"id": "fx_wall",
     "turns": ["Tell me about the Great Wall of China with several facts.",
               "Interesting, tell me more."]},
    {"id": "fx_city",
     "turns": ["Describe the Forbidden City including when it was built and by whom.",
               "Go on, what else?"]},
    {"id": "fx_panda",
     "turns": ["Give me a few facts about giant pandas.",
               "And what do they eat?"]},
]
BARGE_FRACTION = 0.45   # 偏向落在片段中间（partial=True，触发重写）


def load_dialogues(path):
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    logger.warning("未提供 --dialogues，使用内置 fixture（验证 harness，非最终数据）")
    return FIXTURE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=str, default=None)
    ap.add_argument("--fraction", type=float, default=BARGE_FRACTION)
    ap.add_argument("--out", type=str, default=str(Path(RESULTS_DIR) / "exp_a2_history.json"))
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    logger.info("=" * 66)
    logger.info("消融 A2：naive / mark / rewrite 历史处理策略 harness")
    logger.info("=" * 66)

    dialogues = load_dialogues(args.dialogues)
    llm = StreamLLMInference(model_name="Qwen/Qwen2.5-0.5B-Instruct", eval_mode=False)
    rewriter = HistoryRewriter()   # Qwen3-0.6B

    records = []
    for dlg in dialogues:
        t1, probe = dlg["turns"][0], dlg["turns"][1]
        for policy in POLICIES:
            orch = DialogueOrchestrator(
                llm, MockStreamingTTS(TimingProfile()),
                max_speculative_tokens=40,
                truncation_mode="playback",
                history_policy=policy,
                rewriter=rewriter if policy == "rewrite" else None,
            )
            r1 = orch.user_turn(t1, barge_in_fraction=args.fraction)
            r2 = orch.user_turn(probe, barge_in_fraction=None)
            rec = {
                "id": dlg["id"], "policy": policy, "fraction": args.fraction,
                "partial": r1.partial,
                "heard_text": r1.heard_text,
                "history_text": r1.history_text,
                "rewrite_ms": round(r1.metrics.rewrite_ms, 1),
                "next_reply": r2.full_assistant_text,
                "judge_coherence": None,      # 实验机 LLM-judge 填充
            }
            records.append(rec)
            logger.info(f"[{dlg['id']}/{policy}] partial={r1.partial} "
                        f"rewrite_ms={rec['rewrite_ms']}")
            logger.info(f"  history: {rec['history_text'][:90]!r}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    logger.info(f"结果已保存: {out}")

    # ---- harness 自检 ----
    logger.info("-" * 66)
    ok = True
    partial_ids = {r["id"] for r in records if r["policy"] == "naive" and r["partial"]}
    n_changed = 0
    for rid in sorted(partial_ids):
        by = {r["policy"]: r for r in records if r["id"] == rid}
        m, w = by["mark"], by["rewrite"]
        c1 = m["history_text"].endswith(" …")
        # 注意：不能跨 run 比较 rewrite vs naive 的历史（两次独立生成内容本就不同/可能相同）。
        # 同 run 内断言：rewrite 机制被调用（耗时>0）且产出非空历史；改动与否是模型行为，只统计。
        c2 = (not w["partial"]) or (len(w["history_text"].strip()) > 0)
        c3 = (not w["partial"]) or (0 < w["rewrite_ms"] < 3000)
        if w["partial"] and w["history_text"].strip() != w["heard_text"].strip():
            n_changed += 1
        for name, c in [("mark 历史含标记", c1), ("rewrite 产出有效历史", c2),
                        ("rewrite 耗时 (0,3000)ms", c3)]:
            logger.info(f"  [{'PASS' if c else 'FAIL'}] {rid}: {name}")
            ok = ok and c
    if partial_ids:
        logger.info(f"重写实际改动率: {n_changed}/{len(partial_ids)}（模型行为统计，不作断言）")
    rw = [r["rewrite_ms"] for r in records if r["policy"] == "rewrite" and r["partial"]]
    if rw:
        logger.info(f"重写延迟: mean={sum(rw)/len(rw):.0f}ms max={max(rw):.0f}ms "
                    f"(论文论点: << 用户打断后说话时长 ~1000ms+)")
    assert ok, "A2 harness 自检失败"
    logger.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
