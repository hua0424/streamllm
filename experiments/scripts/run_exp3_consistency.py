#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验三：播放感知截断的多轮一致性（E3）—— harness。

核心命题（experiment_design.md §5 E3）：同一打断下，
  B-ours(playback)  历史只含听到内容 → 下一轮不会引用"未听内容"
  B-gen (generation) 历史含未听内容   → 下一轮可能"幻觉引用"未听内容
主指标：未听到内容引用率（规则版检测器代理；LLM-judge 留实验机交叉验证）。

运行（项目根目录）：
    HF_TOKEN= uv run python -m experiments.scripts.run_exp3_consistency
    # 用真实 MultiWOZ：--dialogues path/to/multiwoz_derived.json（格式见 load_dialogues）

设计：
- 每条对话：turn1 = 会引出多部分答案的问题；turn2 = 诱导复述的追问（probe）。
- 打断在 turn1 的回复上，按播放比例 {0.25,0.5,0.75} 注入（P2）。
- 对 B-ours / B-gen 各跑一遍，检测 turn2 是否引用 turn1 的未听内容。
- 增量保存/断点续传（沿用一期实验约定），结果入 experiments/results。
本机用内置 fixture 验证 harness；实验机替换真实 MultiWOZ + LLM-judge 出最终数值。
"""

import argparse
import json
import os
from pathlib import Path

from src.dialogue.orchestrator import DialogueOrchestrator
from src.dialogue.unheard_detector import matched_cues, references_unheard
from src.llm.stream_llm_inference import StreamLLMInference
from src.tts.streaming_tts import MockStreamingTTS, TimingProfile
from src.utils.logging_utils import get_logger, set_global_log_level
from src.config import RESULTS_DIR

logger = get_logger(__name__)

BARGE_FRACTIONS = [0.25, 0.5, 0.75]
CONDITIONS = ["playback", "generation"]   # B-ours vs B-gen

# 内置极小 fixture（MultiWOZ 派生格式的占位）：turn1 引出多部分答案，turn2 诱导复述
FIXTURE = [
    {"id": "fx_wall",
     "turns": ["Tell me about the Great Wall of China with several facts.",
               "Please continue with what you were saying."]},
    {"id": "fx_city",
     "turns": ["Describe the Forbidden City including when it was built and by whom.",
               "Remind me of the details you just mentioned."]},
    {"id": "fx_palace",
     "turns": ["Give me a few facts about the Summer Palace in Beijing.",
               "Continue telling me more about it."]},
]


def load_dialogues(path: str | None):
    """
    读多轮对话。真实 MultiWOZ 派生格式：JSON 列表，每项 {"id": str, "turns": [user_text, ...]}
    （turns 为 user 侧文本序列，至少 2 轮：answer 轮 + probe 轮）。
    path 为空或不存在 → 用内置 fixture（本机 harness 验证）。
    """
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"加载对话集: {path}（{len(data)} 条）")
        return data
    logger.warning("未提供有效 --dialogues，使用内置 fixture（仅验证 harness，非最终数据）")
    return FIXTURE


def run(dialogues, out_path: Path, max_spec: int):
    llm = StreamLLMInference(model_name="Qwen/Qwen2.5-0.5B-Instruct", eval_mode=False)
    tts = MockStreamingTTS(TimingProfile())

    # 断点续传：已完成的 (id,frac,cond) 跳过
    done = set()
    records = []
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        records = prev.get("records", [])
        done = {(r["id"], r["fraction"], r["condition"]) for r in records}
        logger.info(f"断点续传：已有 {len(done)} 条记录")

    for dlg in dialogues:
        if len(dlg["turns"]) < 2:
            continue
        t1, probe = dlg["turns"][0], dlg["turns"][1]
        for frac in BARGE_FRACTIONS:
            for cond in CONDITIONS:
                key = (dlg["id"], frac, cond)
                if key in done:
                    continue
                orch = DialogueOrchestrator(llm, tts, max_speculative_tokens=max_spec,
                                            truncation_mode=cond)
                r1 = orch.user_turn(t1, barge_in_fraction=frac)
                r2 = orch.user_turn(probe, barge_in_fraction=None)
                unheard = r1.unheard_in_history_text
                referenced = references_unheard(unheard, r2.full_assistant_text)
                cues = matched_cues(unheard, r2.full_assistant_text)
                rec = {
                    "id": dlg["id"], "fraction": frac, "condition": cond,
                    "n_generated": r1.metrics.n_generated, "n_heard": r1.metrics.n_heard,
                    "n_unheard_in_history": r1.metrics.n_unheard_in_history,
                    "referenced_unheard": referenced, "matched_cues": cues,
                    "waste_rate": round(r1.metrics.waste_rate, 3),
                }
                records.append(rec)
                logger.info(f"  [{dlg['id']} f={frac} {cond}] unheard={rec['n_unheard_in_history']} "
                            f"referenced={referenced} cues={cues}")
                # 增量保存
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

    # 聚合：每 condition 的未听引用率
    logger.info("=" * 60)
    summary = {}
    for cond in CONDITIONS:
        rs = [r for r in records if r["condition"] == cond]
        if not rs:
            continue
        ref_rate = sum(r["referenced_unheard"] for r in rs) / len(rs)
        avg_unheard = sum(r["n_unheard_in_history"] for r in rs) / len(rs)
        summary[cond] = {"n": len(rs), "unheard_reference_rate": round(ref_rate, 3),
                         "avg_unheard_tokens": round(avg_unheard, 1)}
        logger.info(f"[{cond:>10}] n={len(rs)} 未听引用率={ref_rate:.1%} "
                    f"平均未听token={avg_unheard:.1f}")
    out_path.write_text(json.dumps({"summary": summary, "records": records},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已保存: {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=str, default=None, help="MultiWOZ 派生对话集 JSON；缺省用 fixture")
    ap.add_argument("--max-spec", type=int, default=40, help="推测生成 token 上限")
    ap.add_argument("--out", type=str, default=str(Path(RESULTS_DIR) / "exp3_consistency.json"))
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    logger.info("=" * 60)
    logger.info("实验三 E3：播放感知截断的多轮一致性 harness")
    logger.info("=" * 60)
    dialogues = load_dialogues(args.dialogues)
    summary = run(dialogues, Path(args.out), args.max_spec)

    # harness 自检：B-ours 未听引用率应为 0（历史无未听内容），B-gen 应 > B-ours
    if "playback" in summary and "generation" in summary:
        po = summary["playback"]["unheard_reference_rate"]
        ge = summary["generation"]["unheard_reference_rate"]
        logger.info("-" * 60)
        logger.info(f"harness 自检：B-ours={po:.1%}  B-gen={ge:.1%}  "
                    f"(期望 B-ours==0 且 B-gen>=B-ours)")
        assert po == 0.0, "B-ours 未听引用率应为 0（历史不含未听内容）"
        assert ge >= po, "B-gen 未听引用率应 >= B-ours"
        logger.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
