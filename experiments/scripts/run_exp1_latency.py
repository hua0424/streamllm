#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验一：端到端延迟对比（E1）—— System A（非流式基线） vs System B-ours。

指标（experiment_design.md §5 E1）：
  - TTFT_text：用户说完 → LLM 首 token（本机真实测量，核心对比）
  - mouth-to-ear：说完 → 听到首块音频（**建模值**：B=首个句子片段就绪+TTS首块延迟
    （断句攒首片段的时间已计入）；A=全部生成+完整合成(音频时长×RTF)。
    正式数值实验机 real CosyVoice2 测）
  - barge-in 响应延迟：见 run_exp_a1_kvreuse（反查+crop，0.12-0.19ms 近常数）

System A：等完整 user 文本 → 一次性全量 prefill → 生成（一期非流式路径）→
          等全部文本生成完 → 非流式合成 → 播放
System B：增量 prefill + 软触发推测（E2 机制，取中位阈值）→ 流式断句 → 流式合成

确定性文本段驱动（P1）；真实音频→ASR 链路留实验机（一期已证 TTFT 与语音长度无关）。

运行（项目根目录）：
    HF_TOKEN= uv run python -m experiments.scripts.run_exp1_latency
"""

import argparse
import json
import time
from pathlib import Path

from src.dialogue.orchestrator import DialogueOrchestrator
from src.dialogue.trigger import LLMSoftTrigger
from src.llm.stream_llm_inference import StreamLLMInference
from src.tts.streaming_tts import MockStreamingTTS, TimingProfile
from src.utils.logging_utils import get_logger, set_global_log_level
from src.config import RESULTS_DIR, P2_LLM_MODEL_NAME

logger = get_logger(__name__)

# 与 E2 同 fixture（段边界=停顿）
FIXTURE = [
    {"id": "fx1", "segments": ["Tell me about the Great Wall", " of China and its history."]},
    {"id": "fx2", "segments": ["What's the weather like", " in Beijing this weekend?"]},
    {"id": "fx3", "segments": ["Recommend a restaurant", " near the Forbidden City for dinner."]},
    {"id": "fx4", "segments": ["How do I get to the airport", " from downtown by subway?"]},
    {"id": "fx5", "segments": ["I need a hotel", " for two nights", " near the city center."]},
]
SYNTH_RTF = 0.3   # 非流式合成耗时 = 音频时长 × RTF（占位；实验机实测 CosyVoice2 替换）


# 与 B-ours（orchestrator 默认）完全一致的 system prompt —— 两系统生成内容才可比（review BUG2-①）
SYSTEM_PROMPT = "You are a helpful assistant. Reply in English."


def run_system_a(llm, profile: TimingProfile, full_text: str, max_tokens: int):
    """非流式基线：说完后才开始全量 prefill → 生成全部 → 完整合成（建模）。
    注：B 的 prefill 与用户说话重叠（一期流式 prefill 机制，是被测系统本身的一部分），
    A 在说完后支付全额 prefill——这是 E1 对比的语义（A=非流式基线），非不公平偏置。"""
    t0 = time.perf_counter()
    first_ms, n_tok, out = None, 0, []
    for tok in llm.once_add_and_generate(full_text, system_prompt=SYSTEM_PROMPT,
                                         max_new_tokens=max_tokens):
        if first_ms is None:
            first_ms = (time.perf_counter() - t0) * 1000
        out.append(tok)
        n_tok += 1
    gen_total_ms = (time.perf_counter() - t0) * 1000
    text = "".join(out)
    audio_s = profile.n_samples_for_text(text) / profile.sample_rate
    synth_ms = audio_s * 1000 * SYNTH_RTF
    mte_ms = gen_total_ms + synth_ms          # 说完→生成完→合成完→开始播放
    return {"ttft_ms": round(first_ms or 0, 1), "gen_total_ms": round(gen_total_ms, 1),
            "mouth_to_ear_ms": round(mte_ms, 1), "n_tokens": n_tok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=str, default=None)
    ap.add_argument("--model", type=str, default=P2_LLM_MODEL_NAME, help="主 LLM（实验机传 7B）")
    ap.add_argument("--spec-threshold", type=float, default=0.05, help="B-ours 软触发阈值（E2 拐点附近）")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--out", type=str, default=str(Path(RESULTS_DIR) / "exp1_latency.json"))
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    logger.info("=" * 72)
    logger.info("实验一 E1：System A（非流式） vs System B-ours 延迟对比")
    logger.info("=" * 72)

    if args.dialogues and Path(args.dialogues).exists():
        dialogues = json.loads(Path(args.dialogues).read_text(encoding="utf-8"))
    else:
        logger.warning("使用内置 fixture（验证 harness / 概念数值，正式数值实验机）")
        dialogues = FIXTURE

    llm = StreamLLMInference(model_name=args.model, eval_mode=False)
    trigger = LLMSoftTrigger()
    profile = TimingProfile()

    # warmup（两条路径各跑一次不记录）
    _ = run_system_a(llm, profile, "Hello there.", 8)
    warm = DialogueOrchestrator(llm, MockStreamingTTS(profile), max_speculative_tokens=8,
                                trigger=trigger, spec_threshold=args.spec_threshold)
    warm.speculative_turn(["Hi", " there."])

    records = []
    logger.info(f"{'id':>5} | {'A TTFT':>8} | {'B TTFT_eff':>10} | {'A m2e*':>9} | {'B m2e*':>8}")
    for dlg in dialogues:
        full_text = "".join(dlg["segments"])
        a = run_system_a(llm, profile, full_text, args.max_tokens)
        orch = DialogueOrchestrator(
            llm, MockStreamingTTS(profile), max_speculative_tokens=args.max_tokens,
            trigger=trigger, spec_threshold=args.spec_threshold)
        r = orch.speculative_turn(dlg["segments"])
        m = r.metrics
        b = {"ttft_ms": round(m.first_token_ms, 1),
             "mouth_to_ear_ms": round(m.mouth_to_ear_ms, 1),
             "spec_survived": m.spec_survived, "spec_wasted": m.spec_wasted_tokens}
        records.append({"id": dlg["id"], "system_a": a, "system_b": b})
        logger.info(f"{dlg['id']:>5} | {a['ttft_ms']:>7.1f}ms | {b['ttft_ms']:>9.1f}ms "
                    f"| {a['mouth_to_ear_ms']:>8.1f} | {b['mouth_to_ear_ms']:>7.1f}")

    # 聚合
    n = len(records)
    a_ttft = sum(r["system_a"]["ttft_ms"] for r in records) / n
    b_ttft = sum(r["system_b"]["ttft_ms"] for r in records) / n
    a_mte = sum(r["system_a"]["mouth_to_ear_ms"] for r in records) / n
    b_mte = sum(r["system_b"]["mouth_to_ear_ms"] for r in records) / n
    summary = {"n": n, "a_ttft_ms": round(a_ttft, 1), "b_ttft_eff_ms": round(b_ttft, 1),
               "a_mouth_to_ear_ms_modeled": round(a_mte, 1),
               "b_mouth_to_ear_ms_modeled": round(b_mte, 1),
               "ttft_improvement": round((a_ttft - b_ttft) / a_ttft, 3) if a_ttft else None,
               "spec_threshold": args.spec_threshold, "synth_rtf_placeholder": SYNTH_RTF}
    logger.info("-" * 72)
    logger.info(f"均值: A TTFT={a_ttft:.1f}ms  B TTFT_eff={b_ttft:.1f}ms "
                f"(改善 {summary['ttft_improvement']:.0%})")
    logger.info(f"建模 mouth-to-ear: A={a_mte:.1f}ms  B={b_mte:.1f}ms（*正式数值实验机 real TTS）")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "records": records},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已保存: {out}")

    assert b_ttft < a_ttft, "B-ours TTFT_eff 应低于 System A"
    assert b_mte < a_mte, "B-ours 建模 mouth-to-ear 应低于 System A"
    logger.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
