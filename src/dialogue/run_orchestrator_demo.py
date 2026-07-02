# src/dialogue/run_orchestrator_demo.py
"""
编排闭环端到端 demo（Mock TTS + SimulatedPlayer + 确定性打断）。

运行（项目根目录）：
    HF_TOKEN= uv run python -m src.dialogue.run_orchestrator_demo

跑一段多轮对话，验证：
  T1 中途打断(0.5)：只把听到的 token 留进历史，KV 三长度一致
  T2 完整听完：全部进历史，续轮基于 T1"听到的历史"连贯
  T3 尾部打断(0.85)
  每轮后 KV(seq/mask/DynamicCache) 一致；heard 是 full 的 token 前缀
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


def _report(r):
    logger.info(f"  完整生成 : {r.full_assistant_text!r}")
    logger.info(f"  听到(历史): {r.heard_text!r}")
    if r.interrupted:
        logger.info(f"  作废(未听): {r.discarded_text!r}  (partial={r.partial})")


def main():
    set_global_log_level("INFO")
    logger.info("=" * 64)
    logger.info("编排闭环 demo：Mock TTS + 播放器 + 确定性打断")
    logger.info("=" * 64)

    llm = StreamLLMInference(model_name="Qwen/Qwen2.5-0.5B-Instruct", eval_mode=False)
    tts = MockStreamingTTS(TimingProfile())   # 占位 profile，上实验机换真实 benchmark
    orch = DialogueOrchestrator(llm, tts, max_speculative_tokens=40)

    turns = [
        ("Tell me about the Great Wall of China.", 0.5),    # T1 中途打断
        ("What about the Forbidden City?", None),           # T2 完整听完
        ("And the Summer Palace?", 0.85),                   # T3 尾部打断
    ]

    for user_text, frac in turns:
        logger.info("-" * 64)
        r = orch.user_turn(user_text, barge_in_fraction=frac)
        _report(r)
        # 不变量校验
        _check("KV 三长度一致", orch.assert_kv_consistent())
        _check("生成了 token", r.n_generated_tokens > 0)
        if r.interrupted:
            _check("听到 token 数 <= 生成 token 数", r.n_heard_tokens <= r.n_generated_tokens)
            _check("heard 是 full 的字符前缀", r.full_assistant_text.startswith(r.heard_text))
        else:
            _check("完整听完：heard == full", r.heard_text == r.full_assistant_text)
        _check("有片段产出", len(r.fragments) > 0)

    logger.info("=" * 64)
    logger.info("ALL PASS ✓  —— 完整编排闭环跑通：打断只留下'听到的内容'，多轮连贯")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
