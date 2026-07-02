# src/llm/run_kvcrop_test.py
"""
二期 KV 累积 + 播放感知 crop + role 重建 smoke 测试（需 0.5B LLM，GPU 优先）。

运行（项目根目录）：
    uv run python -m src.llm.run_kvcrop_test

验证链路：
  S1 assistant-side KV 累积：generate_accumulating 后 seq/mask/DynamicCache 三者长度一致
  S2 播放感知 crop：crop_to_token 后三者同步截短、assistant_token_ids 裁剪正确
  S3 role 重建 + 续轮：reopen_user_role → prefill_user_text → open_assistant_role → 再生成
  S4 推测整段作废：crop 回 assistant_start（0 个 assistant token）
"""

import torch
from src.llm.stream_llm_inference import StreamLLMInference
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)


def _check(name: str, cond: bool):
    logger.info(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def _consistent(cache, tag: str):
    """seq_length == attention_mask 长度 == DynamicCache 序列长度。"""
    mask_len = cache.attention_mask.shape[1]
    kv_len = cache.past_key_values.get_seq_length()
    _check(f"{tag}: seq={cache.seq_length} == mask={mask_len} == kv={kv_len}",
           cache.seq_length == mask_len == kv_len)


def main():
    set_global_log_level("INFO")
    logger.info("=" * 60)
    logger.info("二期 KV crop / role 重建 smoke test")
    logger.info("=" * 60)

    # 验证机用 0.5B 小模型（不用 .env 里的 7B 实验模型）；device auto→cuda
    llm = StreamLLMInference(model_name="Qwen/Qwen2.5-0.5B-Instruct", eval_mode=False)
    logger.info(f"device={llm.device}, role_switch={llm._role_switch_to_user!r}")

    # ---- 建初始 KV：system + user + generation_prompt（assistant role 打开）----
    kv = llm.cache_prompt("北京今天天气怎么样？", is_end=True)
    acc = llm.to_accum_cache(kv)
    a_start = acc.assistant_start
    _consistent(acc, "init")
    _check("assistant_start == 初始 seq_length", a_start == acc.seq_length)

    # ---- S1 assistant-side KV 累积 ----
    logger.info("S1 KV 累积")
    toks, idxs = [], []
    for t, i in llm.generate_accumulating(acc, max_new_tokens=12):
        toks.append(t); idxs.append(i)
    n = len(toks)
    logger.info(f"  生成 {n} token: {''.join(toks)!r}")
    _check("生成了 token", n > 0)
    _check("rel_idx 连续 0..n-1", idxs == list(range(n)))
    _check(f"seq_length == a_start+n ({a_start}+{n})", acc.seq_length == a_start + n)
    _check("assistant_token_ids 数量 == n", len(acc.assistant_token_ids) == n)
    _consistent(acc, "after-gen")

    # ---- S2 播放感知 crop：只保留前 k 个 assistant token ----
    logger.info("S2 播放感知 crop")
    k = min(4, n)
    llm.crop_to_token(acc, a_start + k)
    _check(f"seq_length == a_start+k ({a_start}+{k})", acc.seq_length == a_start + k)
    _check("assistant_token_ids 裁到 k", len(acc.assistant_token_ids) == k)
    _check("crop 后 next_token_logits 置 None", acc.next_token_logits is None)
    _consistent(acc, "after-crop")

    # ---- S3 role 重建 + 续轮 ----
    logger.info("S3 role 重建 + 续轮")
    llm.reopen_user_role(acc)          # 关闭 assistant、打开 user
    _consistent(acc, "after-reopen-user")
    llm.prefill_user_text(acc, "那明天呢？")
    _consistent(acc, "after-user-text")
    llm.open_assistant_role(acc)       # 关闭 user、打开新 assistant
    _consistent(acc, "after-open-assistant")
    _check("续轮 assistant_start 更新==seq_length", acc.assistant_start == acc.seq_length)
    _check("续轮 assistant_token_ids 清空", len(acc.assistant_token_ids) == 0)
    _check("open_assistant_role 后有起始 logits", acc.next_token_logits is not None)

    toks2 = [t for t, _ in llm.generate_accumulating(acc, max_new_tokens=8)]
    logger.info(f"  续轮生成: {''.join(toks2)!r}")
    _check("续轮生成了 token", len(toks2) > 0)
    _consistent(acc, "after-gen2")

    # ---- S4 推测整段作废：crop 回 assistant_start ----
    logger.info("S4 推测整段作废（回滚到 assistant 起点）")
    turn2_start = acc.assistant_start
    llm.crop_to_token(acc, turn2_start)
    _check("seq_length == 本轮 assistant_start", acc.seq_length == turn2_start)
    _check("assistant_token_ids 空", len(acc.assistant_token_ids) == 0)
    _consistent(acc, "after-rollback")

    logger.info("=" * 60)
    logger.info("ALL PASS ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
