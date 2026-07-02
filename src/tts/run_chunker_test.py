# src/tts/run_chunker_test.py
"""
sentence_chunker smoke：LLM token 流 → 断句 + token 区间映射，验证映射正确。

运行（项目根目录）：
    HF_TOKEN= uv run python -m src.tts.run_chunker_test

验证：
  A 片段 token 区间连续覆盖 [0, last_end)（无缝、递增）
  B 非空白字符守恒：拼接片段文本的非空白字符数 == 覆盖 token 解码文本的非空白字符数
  C 每个片段 token 区间非空、合法
"""

from src.llm.stream_llm_inference import StreamLLMInference
from src.tts.sentence_chunker import chunk_llm_tokens, _nws
from src.config import P2_LLM_MODEL_NAME
from src.utils.check_utils import make_check
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)


_check = make_check(logger)


def main():
    set_global_log_level("INFO")
    logger.info("=" * 60)
    logger.info("sentence_chunker smoke test（英文 + nltk）")
    logger.info("=" * 60)

    llm = StreamLLMInference(model_name=P2_LLM_MODEL_NAME, eval_mode=False)
    kv = llm.cache_prompt("Introduce Beijing and Shanghai in three short sentences.",
                          is_end=True, system_prompt="You are a helpful assistant. Reply in English.")
    acc = llm.to_accum_cache(kv)

    frags = list(chunk_llm_tokens(
        llm.generate_accumulating(acc, max_new_tokens=60),
        language="en", tokenizer="nltk",
    ))
    n = len(acc.assistant_token_ids)
    all_ids = list(acc.assistant_token_ids)
    logger.info(f"共生成 {n} token，断成 {len(frags)} 个片段：")
    for f in frags:
        logger.info(f"  tokens[{f.token_start:>2},{f.token_end:>2})  {f.text!r}")

    _check("有片段产出", len(frags) > 0)

    # A 连续覆盖
    _check("首片段从 token 0 开始", frags[0].token_start == 0)
    contiguous = all(frags[i].token_start == frags[i - 1].token_end for i in range(1, len(frags)))
    _check("片段 token 区间无缝衔接", contiguous)
    increasing = all(f.token_end > f.token_start for f in frags)
    _check("每片段区间非空且递增", increasing)
    last_end = frags[-1].token_end
    _check(f"末片段 token_end({last_end}) <= 总生成数({n})", last_end <= n)

    # B 非空白字符守恒（覆盖到的 token 范围内）
    concat_frag_nws = sum(_nws(f.text) for f in frags)
    covered_text = llm.tokenizer.decode(all_ids[:last_end], skip_special_tokens=True)
    _check(f"非空白字符守恒（片段 {concat_frag_nws} == 覆盖解码 {_nws(covered_text)}）",
           concat_frag_nws == _nws(covered_text))

    logger.info("=" * 60)
    logger.info("ALL PASS ✓  —— 句子片段已正确映射回 assistant token 区间")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
