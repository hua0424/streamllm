#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验三：播放感知截断的多轮一致性（E3）—— harness。

核心命题（experiment_design.md §5 E3）：同一打断下，
  B-ours(playback)  历史只含听到内容；B-gen(generation) 历史含未听内容 → 下一轮可能幻觉引用。

⚠️ 指标框架（2026-07-02 审查后修正，review BUG1）：
  - **loose（片段级）**：unheard = 进历史但未听的**完整片段**。playback 下恒为空——
    这是机制的**构造性保证**（论文如此表述），loose 列对 B-ours 是正确性检查、非实验发现；
    实验测量的是 B-gen 的失败率。
  - **strict（严格 ground-truth，P1）**：unheard 另含被打断片段内**未播尾部**（按播放
    采样比例切分）。playback 下 strict 引用率可 >0 —— 它量化的正是"片段级截断粒度"
    （D-008 选 A / §八 取舍）的量化误差，是 E3 的诚实补充列。

打断注入（P2）：播放比例 {0.25,0.5,0.75}（mid-fragment，可触发 partial）+ "boundary"
（吸附片段边界的干净截断对照）。

运行（项目根目录）：
    HF_TOKEN= uv run python -m experiments.scripts.run_exp3_consistency
    # 真实 MultiWOZ：--dialogues path.json（[{"id":str,"turns":[user,...>=3轮]},...]）
    # 实验机：--model Qwen/Qwen2-7B-Instruct（或 .env 设 P2_LLM_MODEL_NAME）
本机规则版检测器验证 harness；实验机加 LLM-judge 交叉验证 + 7B 出正式数值。
"""

import argparse
import json
from pathlib import Path

from src.dialogue.orchestrator import DialogueOrchestrator
from src.dialogue.unheard_detector import matched_cues, references_unheard
from src.llm.stream_llm_inference import StreamLLMInference
from src.tts.streaming_tts import MockStreamingTTS, TimingProfile
from src.utils.logging_utils import get_logger, set_global_log_level
from src.config import RESULTS_DIR, P2_LLM_MODEL_NAME

logger = get_logger(__name__)

# P2：三个 mid-fragment 比例 + 片段边界对照
BARGE_SPECS = [0.25, 0.5, 0.75, "boundary"]
CONDITIONS = ["playback", "generation"]   # B-ours vs B-gen

# 内置极小 fixture（MultiWOZ 派生格式占位）：turn1 引出多部分答案，turn2/3 诱导复述（≥3 轮，§4）
FIXTURE = [
    {"id": "fx_wall",
     "turns": ["Tell me about the Great Wall of China with several facts.",
               "Please continue with what you were saying.",
               "So what was the last thing you told me about it?"]},
    {"id": "fx_city",
     "turns": ["Describe the Forbidden City including when it was built and by whom.",
               "Remind me of the details you just mentioned.",
               "And who did you say built it?"]},
    {"id": "fx_palace",
     "turns": ["Give me a few facts about the Summer Palace in Beijing.",
               "Continue telling me more about it.",
               "Repeat the main facts you gave me."]},
]


def load_dialogues(path: str | None):
    """真实 MultiWOZ 派生格式：JSON 列表 [{"id": str, "turns": [user_text, ...]}]（≥3 轮）。"""
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        logger.info(f"加载对话集: {path}（{len(data)} 条）")
        return data
    logger.warning("未提供有效 --dialogues，使用内置 fixture（仅验证 harness，非最终数据）")
    return FIXTURE


def run(dialogues, out_path: Path, max_spec: int, model_name: str):
    llm = StreamLLMInference(model_name=model_name, eval_mode=False)
    tts = MockStreamingTTS(TimingProfile())

    done = set()
    records = []
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        records = prev.get("records", [])
        done = {(r["id"], str(r["fraction"]), r["condition"]) for r in records}
        logger.info(f"断点续传：已有 {len(done)} 条记录")

    for dlg in dialogues:
        if len(dlg["turns"]) < 2:
            continue
        t1, probes = dlg["turns"][0], dlg["turns"][1:]
        for frac in BARGE_SPECS:
            for cond in CONDITIONS:
                key = (dlg["id"], str(frac), cond)
                if key in done:
                    continue
                orch = DialogueOrchestrator(llm, tts, max_speculative_tokens=max_spec,
                                            truncation_mode=cond)
                snap = frac == "boundary"
                r1 = orch.user_turn(t1, barge_in_fraction=0.5 if snap else frac,
                                    barge_in_snap_boundary=snap)
                # 多轮 probe（§4 ≥3 轮）：任一后续轮引用即计
                probe_hits, probe_hits_strict, replies = [], [], []
                for p in probes:
                    rp = orch.user_turn(p, barge_in_fraction=None)
                    replies.append(rp.full_assistant_text)
                    probe_hits.append(references_unheard(r1.unheard_in_history_text,
                                                         rp.full_assistant_text))
                    probe_hits_strict.append(references_unheard(r1.strict_unheard_in_history_text,
                                                                rp.full_assistant_text))
                rec = {
                    "id": dlg["id"], "fraction": frac, "condition": cond,
                    "partial": r1.partial,
                    "n_generated": r1.metrics.n_generated, "n_heard": r1.metrics.n_heard,
                    "n_unheard_in_history": r1.metrics.n_unheard_in_history,
                    "strict_unheard_chars": len(r1.strict_unheard_in_history_text.strip()),
                    "referenced_unheard": any(probe_hits),                 # loose（片段级）
                    "referenced_unheard_strict": any(probe_hits_strict),   # strict（P1 GT）
                    "matched_cues": matched_cues(r1.unheard_in_history_text, " ".join(replies)),
                    "matched_cues_strict": matched_cues(r1.strict_unheard_in_history_text,
                                                        " ".join(replies)),
                    "waste_rate": round(r1.metrics.waste_rate, 3),
                    "kv_reuse_rate": round(r1.metrics.kv_reuse_rate, 3),
                    "timeline": r1.timeline_records,        # §6 反向映射落盘
                    "timestamps": r1.metrics.timestamps,    # §6 时间戳落盘
                }
                records.append(rec)
                logger.info(f"  [{dlg['id']} f={frac} {cond}] unheard={rec['n_unheard_in_history']} "
                            f"strict_chars={rec['strict_unheard_chars']} "
                            f"ref={rec['referenced_unheard']} ref_strict={rec['referenced_unheard_strict']}")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2),
                                    encoding="utf-8")

    # 聚合：每 condition 的 loose / strict 未听引用率
    logger.info("=" * 72)
    summary = {}
    for cond in CONDITIONS:
        rs = [r for r in records if r["condition"] == cond]
        if not rs:
            continue
        loose = sum(r["referenced_unheard"] for r in rs) / len(rs)
        strict = sum(r["referenced_unheard_strict"] for r in rs) / len(rs)
        avg_unheard = sum(r["n_unheard_in_history"] for r in rs) / len(rs)
        summary[cond] = {"n": len(rs),
                         "unheard_reference_rate_loose": round(loose, 3),
                         "unheard_reference_rate_strict": round(strict, 3),
                         "avg_unheard_tokens": round(avg_unheard, 1)}
        logger.info(f"[{cond:>10}] n={len(rs)} loose引用率={loose:.1%} "
                    f"strict引用率={strict:.1%} 平均未听token={avg_unheard:.1f}")
    out_path.write_text(json.dumps({"summary": summary, "records": records},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已保存: {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=str, default=None, help="MultiWOZ 派生对话集 JSON；缺省用 fixture")
    ap.add_argument("--model", type=str, default=P2_LLM_MODEL_NAME, help="主 LLM（实验机传 7B）")
    ap.add_argument("--max-spec", type=int, default=40, help="推测生成 token 上限")
    ap.add_argument("--out", type=str, default=str(Path(RESULTS_DIR) / "exp3_consistency.json"))
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    logger.info("=" * 72)
    logger.info("实验三 E3：播放感知截断的多轮一致性 harness（loose+strict 双列）")
    logger.info("=" * 72)
    dialogues = load_dialogues(args.dialogues)
    summary = run(dialogues, Path(args.out), args.max_spec, args.model)

    # harness 自检：
    #  - loose：playback 恒 0 是**构造性保证**（机制正确性检查，非实验发现）；B-gen >= B-ours
    #  - strict：只报告不断言方向（playback 的 strict>0 是量化误差发现，属正常结果）
    if "playback" in summary and "generation" in summary:
        po = summary["playback"]["unheard_reference_rate_loose"]
        ge = summary["generation"]["unheard_reference_rate_loose"]
        ps = summary["playback"]["unheard_reference_rate_strict"]
        logger.info("-" * 72)
        logger.info(f"自检：loose B-ours={po:.1%}(构造性=0) B-gen={ge:.1%}；"
                    f"strict B-ours={ps:.1%}(片段粒度量化误差，报告项)")
        assert po == 0.0, "loose 下 B-ours 应构造性为 0（机制正确性被破坏）"
        assert ge >= po, "B-gen loose 引用率应 >= B-ours"
        logger.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
