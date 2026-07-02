# src/dialogue/run_conditions_demo.py
"""
E3 对照骨架 demo：同一对话+打断，B-ours(playback) vs B-gen(generation) 两种截断模式对比。

运行（项目根目录）：
    HF_TOKEN= uv run python -m src.dialogue.run_conditions_demo

论文核心命题的可测量化：
  同样在 T1 中途打断后，
  - B-ours(playback)   : 历史 = 用户听到的 → n_unheard_in_history == 0（历史干净）
  - B-gen (generation) : 历史 = 全部生成的 → n_unheard_in_history > 0（历史含用户没听到的内容
                          = 后续轮次可能"幻觉引用"的来源）
这就是 E3 要量化的差异来源（未听到内容引用率的土壤）。
"""

from src.dialogue.orchestrator import DialogueOrchestrator
from src.llm.stream_llm_inference import StreamLLMInference
from src.tts.streaming_tts import MockStreamingTTS, TimingProfile
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)


def _check(name, cond):
    logger.info(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def run_one(mode: str):
    llm = StreamLLMInference(model_name="Qwen/Qwen2.5-0.5B-Instruct", eval_mode=False)
    tts = MockStreamingTTS(TimingProfile())
    orch = DialogueOrchestrator(llm, tts, max_speculative_tokens=40, truncation_mode=mode)
    # 固定对话与打断点，两模式可比
    r1 = orch.user_turn("Tell me about the Great Wall of China.", barge_in_fraction=0.5)
    r2 = orch.user_turn("What did you just say about it?", barge_in_fraction=None)
    return r1, r2


def main():
    set_global_log_level("INFO")
    logger.info("=" * 66)
    logger.info("E3 对照骨架：B-ours(playback) vs B-gen(generation)")
    logger.info("=" * 66)

    logger.info("-" * 66 + "\n[B-ours / playback]")
    o1, o2 = run_one("playback")
    logger.info(f"  T1 听到=历史: {o1.history_text!r}")
    logger.info(f"  T1 未听却进历史: {o1.unheard_in_history_text!r}  (n={o1.metrics.n_unheard_in_history})")
    _check("B-ours: 历史无未听内容", o1.metrics.n_unheard_in_history == 0)
    _check("B-ours: 有浪费(生成>听到)", o1.metrics.n_wasted > 0)

    logger.info("-" * 66 + "\n[B-gen / generation]")
    g1, g2 = run_one("generation")
    logger.info(f"  T1 历史(全部生成): {g1.history_text!r}")
    logger.info(f"  T1 未听却进历史: {g1.unheard_in_history_text!r}  (n={g1.metrics.n_unheard_in_history})")
    _check("B-gen: 历史含未听内容(幻觉土壤)", g1.metrics.n_unheard_in_history > 0)
    _check("B-gen: 无浪费(全部进历史)", g1.metrics.n_wasted == 0)

    logger.info("-" * 66)
    logger.info("对照结论：同一打断下，B-ours 历史只含听到内容(unheard_in_history=0)，")
    logger.info("而 B-gen 历史含用户没听到的内容(unheard_in_history>0)——E3 差异来源已量化。")
    logger.info("=" * 66)
    logger.info("ALL PASS ✓")
    logger.info("=" * 66)


if __name__ == "__main__":
    main()
