# src/dialogue/run_trigger_test.py
"""
软触发 smoke：完整句 vs 不完整句 的置信度排序（开发替身 Qwen2.5-0.5B）。

运行（项目根目录）：
    HF_TOKEN= uv run python -m src.dialogue.run_trigger_test

验证（软触发不是论文贡献，只需方向正确、可分性存在）：
  - 完整句均值置信度 > 不完整句均值置信度
  - 存在一个阈值能把两组大致分开（分组 AUC 式检查）
"""

from src.dialogue.trigger import LLMSoftTrigger
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

COMPLETE = [
    "What's the weather like in Beijing today?",
    "Book me a flight to Shanghai tomorrow morning.",
    "Tell me about the Great Wall of China.",
    "How much does the museum ticket cost?",
    "I'd like to cancel my hotel reservation for Friday.",
]
INCOMPLETE = [
    "I want to",
    "Could you tell me about the",
    "So um, what I was thinking is",
    "Book me a flight to",
    "What's the",
]


def _check(name, cond):
    logger.info(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def main():
    set_global_log_level("INFO")
    logger.info("=" * 60)
    logger.info("软触发 smoke（开发替身 prompted Qwen2.5-0.5B）")
    logger.info("=" * 60)

    trig = LLMSoftTrigger()

    comp = [(t, trig.confidence(t)) for t in COMPLETE]
    incomp = [(t, trig.confidence(t)) for t in INCOMPLETE]
    for t, c in comp:
        logger.info(f"  [complete   {c:.3f}] {t!r}")
    for t, c in incomp:
        logger.info(f"  [incomplete {c:.3f}] {t!r}")

    mean_c = sum(c for _, c in comp) / len(comp)
    mean_i = sum(c for _, c in incomp) / len(incomp)
    logger.info(f"均值: complete={mean_c:.3f}  incomplete={mean_i:.3f}")
    _check("完整句均值置信度 > 不完整句", mean_c > mean_i)

    # 可分性：成对比较正确率（AUC 的无参估计）
    pairs = [(c, i) for _, c in comp for _, i in incomp]
    auc = sum(1 for c, i in pairs if c > i) / len(pairs)
    logger.info(f"成对正确率(AUC~)={auc:.2f}")
    _check("AUC~ >= 0.7（可分性足够支撑两阈值机制）", auc >= 0.7)

    logger.info("=" * 60)
    logger.info("ALL PASS ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
