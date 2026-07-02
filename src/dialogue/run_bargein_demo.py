# src/dialogue/run_bargein_demo.py
"""
打断 → 反查 → KV 截断 最小端到端 demo（不接 TTS，用模拟播放位置）。

把二期两块核心拼起来验证：
  PlaybackTimeline（反向映射）  +  StreamLLMInference 的 AccumKVCache/crop_to_token

运行（项目根目录）：
    HF_TOKEN= uv run python -m src.dialogue.run_bargein_demo

链路：
  1. 用户一轮 → 累积生成 assistant token
  2. 把 token 分组成 fragment（模拟 stream2sentence），attach 模拟音频（每 token 固定采样数）
  3. 模拟播放到 60%（片段中间）→ barge_in() 反查听到边界
  4. 桥接：keep_seq_len = assistant_start + crop_token_end → crop_to_token()
  5. 交叉校验：裁剪后 KV 里的 assistant token 恰是"听到"的前缀
  6. 重建 role + 新用户输入 → 续生成，证明多轮 KV 合法可用

核心命题的可视化：generated N token，但只播到 M(<N) → 只把 M 个进历史，N-M 作废。
"""

from src.dialogue.timeline import PlaybackTimeline
from src.llm.stream_llm_inference import StreamLLMInference
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

TOKENS_PER_FRAGMENT = 4      # 模拟 stream2sentence：每 4 个 token 成一个片段
SAMPLES_PER_TOKEN = 1600     # 模拟 TTS：每 token 约 0.1s @16k


def _check(name, cond):
    logger.info(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def main():
    set_global_log_level("INFO")
    logger.info("=" * 60)
    logger.info("打断 → 反查 → KV 截断 端到端 demo")
    logger.info("=" * 60)

    llm = StreamLLMInference(model_name="Qwen/Qwen2.5-0.5B-Instruct", eval_mode=False)

    # ---- 1. 用户一轮，累积生成 ----
    kv = llm.cache_prompt("用一句话介绍北京。", is_end=True)
    acc = llm.to_accum_cache(kv)
    a_start = acc.assistant_start

    gen = []  # [(text, rel_idx)]
    for t, i in llm.generate_accumulating(acc, max_new_tokens=16):
        gen.append((t, i))
    n = len(gen)
    all_ids = list(acc.assistant_token_ids)   # 生成的全部 assistant token id
    full_text = llm.tokenizer.decode(all_ids, skip_special_tokens=True)
    logger.info(f"生成 {n} token: {full_text!r}")

    # ---- 2. 分组成 fragment + 模拟音频 ----
    tl = PlaybackTimeline(turn_id=1)
    frag_spans = []  # (rel_start, rel_end)
    cid = 0
    for start in range(0, n, TOKENS_PER_FRAGMENT):
        end = min(start + TOKENS_PER_FRAGMENT, n)
        span_ids = all_ids[start:end]
        frag_text = llm.tokenizer.decode(span_ids, skip_special_tokens=True)
        fid = tl.add_fragment(frag_text, token_start=start, token_end=end)  # token 用 assistant 相对下标
        tl.attach_chunk(fid, chunk_id=cid, n_samples=len(span_ids) * SAMPLES_PER_TOKEN)
        cid += 1
        frag_spans.append((start, end))
    total = tl.total_samples
    logger.info(f"分成 {len(frag_spans)} 个片段，模拟总音频 {total/16000:.2f}s")

    # ---- 3. 模拟播放到 60%（大概率落在片段中间）→ 打断 ----
    played = int(total * 0.6)
    tl.set_played(played)
    res = tl.barge_in()
    logger.info(f"播放到 {played/16000:.2f}s ({played}/{total}) 打断："
                f"听到片段={res.heard_fragment_ids}, 作废={res.discarded_fragment_ids}, "
                f"partial={res.partial}, crop_token_end(相对)={res.crop_token_end}")

    # ---- 4. 桥接：assistant 相对 crop 点 → 绝对 keep_seq_len → crop ----
    keep_seq_len = a_start + res.crop_token_end
    llm.crop_to_token(acc, keep_seq_len)

    # ---- 5. 交叉校验：KV 里剩的 assistant token 恰是"听到"的前缀 ----
    heard_ids = all_ids[:res.crop_token_end]
    _check("裁剪后 KV assistant token 数 == 听到 token 数",
           len(acc.assistant_token_ids) == res.crop_token_end)
    _check("裁剪后 KV assistant token == 原始前缀",
           acc.assistant_token_ids == heard_ids)
    _check("seq/mask/DynamicCache 三长度一致",
           acc.seq_length == acc.attention_mask.shape[1] == acc.past_key_values.get_seq_length())
    heard_text_llm = llm.tokenizer.decode(acc.assistant_token_ids, skip_special_tokens=True)
    logger.info(f"用户实际听到（进入历史）：{heard_text_llm!r}")
    logger.info(f"被作废（未听到，不进历史）：{full_text[len(heard_text_llm):]!r}")

    # ---- 6. 重建 role + 新用户输入 → 续生成 ----
    llm.reopen_user_role(acc)
    llm.prefill_user_text(acc, "那上海呢？")
    llm.open_assistant_role(acc)
    cont = "".join(t for t, _ in llm.generate_accumulating(acc, max_new_tokens=12))
    logger.info(f"续轮（基于'听到的历史'）生成：{cont!r}")
    _check("续轮生成了 token", len(cont) > 0)
    _check("续轮后三长度仍一致",
           acc.seq_length == acc.attention_mask.shape[1] == acc.past_key_values.get_seq_length())

    logger.info("=" * 60)
    logger.info("ALL PASS ✓  —— 打断只把'听到的内容'留进 KV/历史，两块核心结构拼接正确")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
