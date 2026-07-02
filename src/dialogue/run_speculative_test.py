# src/dialogue/run_speculative_test.py
"""
推测-作废状态机 smoke（FakeTrigger 注入预设置信度，与模型行为解耦）。

运行（项目根目录）：
    HF_TOKEN= uv run python -m src.dialogue.run_speculative_test

场景：
  S1 假停顿：seg1 触发推测 → seg2 到来作废（浪费>0）→ seg2 再触发 → 存活复用
  S2 保守触发：seg1 不触发、seg2（末段）触发 → 存活、零浪费
  S3 从不触发：现场生成，TTFT_eff > 0，零浪费
  S4 多轮连续：S1 后再来一轮，KV 一致、正常生成
"""

from src.dialogue.orchestrator import DialogueOrchestrator
from src.llm.stream_llm_inference import StreamLLMInference
from src.tts.streaming_tts import MockStreamingTTS, TimingProfile
from src.config import P2_LLM_MODEL_NAME
from src.utils.check_utils import make_check
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)


class FakeTrigger:
    """按预设序列吐置信度（超出后重复最后一个）。"""
    def __init__(self, confs):
        self.confs = list(confs)
        self.i = 0

    def confidence(self, _text: str) -> float:
        c = self.confs[min(self.i, len(self.confs) - 1)]
        self.i += 1
        return c


_check = make_check(logger)


SEGS = ["Book me a flight to Beijing.",
        " Actually, make it Shanghai instead, tomorrow morning please."]


def make_orch(llm, trigger):
    return DialogueOrchestrator(
        llm, MockStreamingTTS(TimingProfile()),
        max_speculative_tokens=32, spec_chunk=12,
        trigger=trigger, spec_threshold=0.5,
    )


def main():
    set_global_log_level("INFO")
    logger.info("=" * 62)
    logger.info("推测-作废状态机 smoke（FakeTrigger）")
    logger.info("=" * 62)
    llm = StreamLLMInference(model_name=P2_LLM_MODEL_NAME, eval_mode=False)

    # ---- S1 假停顿：两段都高置信 → 触发→作废→再触发→存活 ----
    logger.info("S1 假停顿（0.9, 0.9）")
    orch = make_orch(llm, FakeTrigger([0.9, 0.9]))
    r = orch.speculative_turn(SEGS)
    m = r.metrics
    logger.info(f"  最终回复: {r.full_assistant_text[:70]!r}")
    _check("触发了 2 次推测", m.n_speculations == 2)
    _check("作废了 1 次", m.n_invalidated == 1)
    _check("有推测浪费（wasted>0）", m.spec_wasted_tokens > 0)
    _check("末段推测存活", m.spec_survived)
    _check("说完瞬间已有就绪 token", m.ready_tokens_at_user_end > 0)
    _check("spec_waste_rate ∈ (0,1)", 0 < m.spec_waste_rate < 1)
    _check("KV 三长度一致", orch.assert_kv_consistent())
    s1_orch = orch

    # ---- S2 保守：只在末段触发 → 存活、零浪费 ----
    logger.info("S2 保守触发（0.1, 0.9）")
    orch = make_orch(llm, FakeTrigger([0.1, 0.9]))
    r = orch.speculative_turn(SEGS)
    m = r.metrics
    _check("只触发 1 次", m.n_speculations == 1)
    _check("零作废", m.n_invalidated == 0)
    _check("零浪费", m.spec_wasted_tokens == 0)
    _check("存活", m.spec_survived)
    _check("KV 一致", orch.assert_kv_consistent())

    # ---- S3 从不触发：现场生成 ----
    logger.info("S3 从不触发（0.1, 0.1）")
    orch = make_orch(llm, FakeTrigger([0.1, 0.1]))
    r = orch.speculative_turn(SEGS)
    m = r.metrics
    _check("零推测", m.n_speculations == 0)
    _check("未存活（现场生成）", not m.spec_survived)
    _check("TTFT_eff > 0（说完才开始生成）", m.first_token_ms > 0)
    _check("零浪费", m.spec_wasted_tokens == 0)
    _check("生成了回复", m.n_generated > 0)
    _check("KV 一致", orch.assert_kv_consistent())

    # ---- S4 多轮：S1 的 orchestrator 再来一轮（带打断混合） ----
    logger.info("S4 多轮连续（S1 之后再一轮，0.5 打断）")
    r2 = s1_orch.speculative_turn([" What about the return flight", " on Sunday evening?"],
                                  barge_in_fraction=0.5)
    _check("第二轮生成了回复", r2.metrics.n_generated > 0)
    _check("第二轮打断生效", r2.interrupted)
    _check("KV 仍一致", s1_orch.assert_kv_consistent())
    logger.info(f"  第二轮听到: {r2.heard_text[:60]!r}")

    logger.info("=" * 62)
    logger.info("ALL PASS ✓  —— 推测-作废状态机 + 打断截断 组合正确")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
