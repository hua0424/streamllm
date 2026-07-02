#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验二：推测浪费率 vs TTFT trade-off 曲线（E2，论文核心图）—— harness。

机制（paper2_context.md §3.5 / experiment_design.md §5 E2）：
软触发输出连续置信度，推测阈值越低（激进）→ 越早推测 → 说完时 token 越可能已就绪
（TTFT_eff→0），但假停顿处的作废浪费越多；阈值越高（保守）→ 零浪费但 TTFT 全额。
扫描阈值得到 (spec_waste_rate, TTFT_eff) 前沿曲线。

运行（项目根目录）：
    HF_TOKEN= uv run python -m experiments.scripts.run_exp2_tradeoff
    # 真实数据：--dialogues path.json，格式 [{"id":str,"segments":[str,...]},...]

本机：0.5B 主 LLM + prompted Qwen2.5-0.5B 软触发替身（D-011），验证 harness 与出概念曲线；
实验机：7B + TEN 7B 出正式数值（阈值扫描区间需按 TEN 置信度分布重标）。
"""

import argparse
import json
from pathlib import Path

from src.dialogue.orchestrator import DialogueOrchestrator
from src.dialogue.trigger import LLMSoftTrigger
from src.llm.stream_llm_inference import StreamLLMInference
from src.tts.streaming_tts import MockStreamingTTS, TimingProfile
from src.utils.logging_utils import get_logger, set_global_log_level
from src.config import RESULTS_DIR, P2_LLM_MODEL_NAME

logger = get_logger(__name__)

# 扫描区间按开发替身的置信度分布标定（run_trigger_test 实测 ~0.02-0.50）；
# 末位 1.1 = 永不触发（保守极限，TTFT 全额基线）。实验机换 TEN 后重标。
DEFAULT_THRESHOLDS = [0.02, 0.05, 0.08, 0.12, 0.20, 0.40, 1.1]

# fixture：多段 utterance，段边界即"停顿"。部分首段句法上近似完整（诱发假停顿浪费）。
FIXTURE = [
    {"id": "fx1", "segments": ["Tell me about the Great Wall", " of China and its history."]},
    {"id": "fx2", "segments": ["What's the weather like", " in Beijing this weekend?"]},
    {"id": "fx3", "segments": ["Recommend a restaurant", " near the Forbidden City for dinner."]},
    {"id": "fx4", "segments": ["How do I get to the airport", " from downtown by subway?"]},
    {"id": "fx5", "segments": ["Tell me a fun fact about pandas.", " Also, where can I see them?"]},
    {"id": "fx6", "segments": ["I need a hotel", " for two nights", " near the city center."]},
]


def load_dialogues(path):
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        logger.info(f"加载对话集: {path}（{len(data)} 条）")
        return data
    logger.warning("未提供 --dialogues，使用内置 fixture（验证 harness / 概念曲线，非最终数据）")
    return FIXTURE


def run(dialogues, thresholds, out_path: Path, max_spec: int, spec_chunk: int,
        model_name: str = P2_LLM_MODEL_NAME):
    llm = StreamLLMInference(model_name=model_name, eval_mode=False)
    trigger = LLMSoftTrigger()   # 开发替身；实验机换 TEN_CONFIG

    records = []
    done = set()
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        records = prev.get("records", [])
        done = {(r["id"], r["threshold"]) for r in records}
        logger.info(f"断点续传：已有 {len(done)} 条")

    for th in thresholds:
        for dlg in dialogues:
            key = (dlg["id"], th)
            if key in done:
                continue
            orch = DialogueOrchestrator(
                llm, MockStreamingTTS(TimingProfile()),
                max_speculative_tokens=max_spec, spec_chunk=spec_chunk,
                trigger=trigger, spec_threshold=th,
            )
            r = orch.speculative_turn(dlg["segments"])
            m = r.metrics
            records.append({
                "id": dlg["id"], "threshold": th,
                "n_generated": m.n_generated, "spec_wasted": m.spec_wasted_tokens,
                "n_speculations": m.n_speculations, "n_invalidated": m.n_invalidated,
                "survived": m.spec_survived, "ready": m.ready_tokens_at_user_end,
                "ttft_eff_ms": round(m.first_token_ms, 1),
                "confs": [round(c, 4) for c in m.trigger_confs],
            })
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({"records": records}, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    # 聚合 → 曲线数据点
    logger.info("=" * 72)
    logger.info(f"{'阈值':>6} | {'浪费率':>7} | {'TTFT_eff(ms)':>12} | {'存活率':>6} | {'均就绪tok':>8}")
    curve = []
    for th in thresholds:
        rs = [r for r in records if r["threshold"] == th]
        if not rs:
            continue
        wasted = sum(r["spec_wasted"] for r in rs)
        gen = sum(r["n_generated"] for r in rs)
        waste_rate = wasted / (wasted + gen) if (wasted + gen) else 0.0
        ttft = sum(r["ttft_eff_ms"] for r in rs) / len(rs)
        surv = sum(r["survived"] for r in rs) / len(rs)
        ready = sum(r["ready"] for r in rs) / len(rs)
        curve.append({"threshold": th, "spec_waste_rate": round(waste_rate, 4),
                      "ttft_eff_ms": round(ttft, 1), "survived_rate": round(surv, 3),
                      "avg_ready_tokens": round(ready, 1), "n": len(rs)})
        logger.info(f"{th:>6} | {waste_rate:>6.1%} | {ttft:>12.1f} | {surv:>6.0%} | {ready:>8.1f}")

    out_path.write_text(json.dumps({"curve": curve, "records": records},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已保存: {out_path}")
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=str, default=None)
    ap.add_argument("--model", type=str, default=P2_LLM_MODEL_NAME, help="主 LLM（实验机传 7B）")
    ap.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS)
    ap.add_argument("--max-spec", type=int, default=32)
    ap.add_argument("--spec-chunk", type=int, default=12)
    ap.add_argument("--out", type=str, default=str(Path(RESULTS_DIR) / "exp2_tradeoff.json"))
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    logger.info("=" * 72)
    logger.info("实验二 E2：推测浪费率 vs TTFT trade-off harness")
    logger.info("=" * 72)
    dialogues = load_dialogues(args.dialogues)
    curve = run(dialogues, args.thresholds, Path(args.out), args.max_spec, args.spec_chunk,
                model_name=args.model)

    # harness 自检：曲线两端方向正确（最激进浪费 >= 最保守浪费；最保守 TTFT >= 最激进 TTFT）
    if len(curve) >= 2:
        lo, hi = curve[0], curve[-1]
        logger.info("-" * 72)
        logger.info(f"自检：激进端 waste={lo['spec_waste_rate']:.1%}/TTFT={lo['ttft_eff_ms']:.0f}ms  "
                    f"保守端 waste={hi['spec_waste_rate']:.1%}/TTFT={hi['ttft_eff_ms']:.0f}ms")
        assert lo["spec_waste_rate"] >= hi["spec_waste_rate"], "激进端浪费应 >= 保守端"
        assert hi["ttft_eff_ms"] >= lo["ttft_eff_ms"], "保守端 TTFT 应 >= 激进端"
        logger.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
