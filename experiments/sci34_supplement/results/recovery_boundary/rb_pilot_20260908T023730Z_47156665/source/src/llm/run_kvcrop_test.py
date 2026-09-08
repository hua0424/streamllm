# src/llm/run_kvcrop_test.py
"""
二期 KV 累积 + 播放感知 crop + role 重建 smoke 测试（需 0.5B LLM，GPU 优先）。

运行（项目根目录）：
    uv run python -m src.llm.run_kvcrop_test

验证链路：
  S1 assistant-side KV 累积：完整 ledger/seq/mask/DynamicCache/内容 span 一致，EOT pending 不入 KV
  S2 播放感知 crop：crop_to_token 同步截短并恢复 role/end state
  S3 role 重建 + 续轮：唯一 EOT close → user/assistant role → 再生成
  S4 推测整段作废：crop 回 assistant_role_start，删除 assistant header 并恢复 USER_OPEN
  S5 assistant 内容 prefill 与 consumer-stop 显式状态
"""

import torch
from src.llm.stream_llm_inference import StreamLLMInference
from src.config import P2_LLM_MODEL_NAME
from src.utils.check_utils import make_check
from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)


_check = make_check(logger)


def _consistent(llm, cache, tag: str):
    """完整 ledger / seq / mask / DynamicCache / assistant span 全部一致。"""
    mask_len = cache.attention_mask.shape[1]
    kv_len = cache.past_key_values.get_seq_length()
    ok = len(cache.token_ids) == cache.seq_length == mask_len == kv_len
    try:
        llm._assert_accum_consistent(cache)
    except AssertionError:
        ok = False
    _check(
        f"{tag}: ledger={len(cache.token_ids)} == seq={cache.seq_length} "
        f"== mask={mask_len} == kv={kv_len}", ok,
    )


def _fails(label, fn):
    try:
        fn()
    except (RuntimeError, ValueError):
        _check(label, True)
    else:
        _check(label, False)


def main():
    set_global_log_level("INFO")
    logger.info("=" * 60)
    logger.info("二期 KV crop / role 重建 smoke test")
    logger.info("=" * 60)

    # 模型走 P2_LLM_MODEL_NAME（默认 0.5B；.env 覆盖后 smoke 也会用覆盖值，注意显存）
    llm = StreamLLMInference(model_name=P2_LLM_MODEL_NAME, eval_mode=False)
    logger.info(f"device={llm.device}, role_switch={llm._role_switch_to_user!r}")

    # ---- 建初始 KV：system + user + generation_prompt（assistant role 打开）----
    kv = llm.cache_prompt("北京今天天气怎么样？", is_end=True)
    acc = llm.to_accum_cache(kv)
    a_start = acc.assistant_start
    _consistent(llm, acc, "init")
    _check("assistant_start == 初始 seq_length", a_start == acc.seq_length)
    _check("初始 role=ASSISTANT_OPEN", acc.role_phase == llm.RolePhase.ASSISTANT_OPEN)
    _check("规范 user→assistant transition 已按 token ID 校验", bool(llm._user_to_assistant_ids))
    _check("规范 assistant→user transition 仅含一个 EOT",
           llm._assistant_to_user_ids.count(llm._assistant_eot_id) == 1)
    _fails("重复 open assistant fail closed", lambda: llm.open_assistant_role(acc))
    _fails("assistant phase 禁止 prefill_user_text",
           lambda: llm.prefill_user_text(acc, "非法用户文本"))

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
    _check("EOT 不在内容 ledger", llm._assistant_eot_id not in acc.assistant_token_ids)
    _check("生成终因显式记录",
           acc.generation_end_reason in (llm.GenerationEndReason.EOS,
                                         llm.GenerationEndReason.MAX_TOKENS))
    if acc.generation_end_reason == llm.GenerationEndReason.EOS:
        _check("自然 EOS 进入 pending close",
               acc.role_phase == llm.RolePhase.ASSISTANT_EOT_PENDING)
        _check("pending EOT 未进入完整 ledger",
               not acc.token_ids or acc.token_ids[-1] != llm._assistant_eot_id)
        pending_reason = acc.generation_end_reason
        llm.crop_to_token(acc, acc.seq_length)
        _check("no-op crop 保留 pending EOS 状态",
               acc.role_phase == llm.RolePhase.ASSISTANT_EOT_PENDING
               and acc.generation_end_reason == pending_reason)
    _consistent(llm, acc, "after-gen")

    # ---- S2 播放感知 crop：只保留前 k 个 assistant token ----
    logger.info("S2 播放感知 crop")
    k = min(4, n)
    llm.crop_to_token(acc, a_start + k)
    _check(f"seq_length == a_start+k ({a_start}+{k})", acc.seq_length == a_start + k)
    _check("crop 恢复 ASSISTANT_OPEN", acc.role_phase == llm.RolePhase.ASSISTANT_OPEN)
    _check("crop 显式记录 CROPPED",
           acc.generation_end_reason == llm.GenerationEndReason.CROPPED)
    _check("assistant_token_ids 裁到 k", len(acc.assistant_token_ids) == k)
    _check("crop 后 next_token_logits 置 None", acc.next_token_logits is None)
    _consistent(llm, acc, "after-crop")

    # ---- S3 role 重建 + 续轮 ----
    logger.info("S3 role 重建 + 续轮")
    before_close = acc.seq_length
    llm.reopen_user_role(acc)          # 唯一提交 EOT，关闭 assistant、打开 user
    _check("reopen 只提交一次规范 transition",
           acc.seq_length == before_close + len(llm._assistant_to_user_ids))
    _check("transition 中 EOT 唯一",
           acc.token_ids[before_close:].count(llm._assistant_eot_id) == 1)
    _check("reopen 后 role=USER_OPEN", acc.role_phase == llm.RolePhase.USER_OPEN)
    _fails("重复 reopen fail closed", lambda: llm.reopen_user_role(acc))
    _fails("user phase 禁止 prefill_assistant_text",
           lambda: llm.prefill_assistant_text(acc, "非法 assistant 文本"))
    _consistent(llm, acc, "after-reopen-user")
    llm.prefill_user_text(acc, "那明天呢？")
    _consistent(llm, acc, "after-user-text")
    llm.open_assistant_role(acc)       # 关闭 user、打开新 assistant
    _consistent(llm, acc, "after-open-assistant")
    _check("续轮 assistant_start 更新==seq_length", acc.assistant_start == acc.seq_length)
    _check("续轮 assistant_token_ids 清空", len(acc.assistant_token_ids) == 0)
    _check("open_assistant_role 后有起始 logits", acc.next_token_logits is not None)

    toks2 = [t for t, _ in llm.generate_accumulating(acc, max_new_tokens=8)]
    logger.info(f"  续轮生成: {''.join(toks2)!r}")
    _check("续轮生成了 token", len(toks2) > 0)
    _consistent(llm, acc, "after-gen2")

    # ---- S4 推测整段作废：crop 回 assistant_start ----
    logger.info("S4 推测整段作废（回滚到 assistant role header 起点）")
    turn2_start = acc.assistant_role_start
    llm.crop_to_token(acc, turn2_start)
    _check("seq_length == 本轮 assistant role 起点", acc.seq_length == turn2_start)
    _check("assistant_token_ids 空", len(acc.assistant_token_ids) == 0)
    _check("crop 到 role 起点恢复 USER_OPEN", acc.role_phase == llm.RolePhase.USER_OPEN)
    _check("rollback 终因=CROPPED",
           acc.generation_end_reason == llm.GenerationEndReason.CROPPED)
    _consistent(llm, acc, "after-rollback")
    llm.prefill_user_text(acc, "继续补充原用户请求。")
    _check("invalidation 后新 user 内容清除陈旧 CROPPED 终因",
           acc.generation_end_reason == llm.GenerationEndReason.NONE)
    _check("追加 user 内容后仍为 USER_OPEN",
           acc.role_phase == llm.RolePhase.USER_OPEN)
    _consistent(llm, acc, "after-invalidation-user-text")

    # ---- S5 assistant 文本 prefill 与消费者提前停止 ----
    logger.info("S5 assistant prefill + consumer stop")
    llm.open_assistant_role(acc)
    llm.prefill_assistant_text(acc, "已确认：")
    _check("prefill_assistant_text 写入内容 ledger", len(acc.assistant_token_ids) > 0)
    gen = llm.generate_accumulating(acc, max_new_tokens=8)
    first = next(gen, None)
    _check("消费者停止前生成至少一个内容 token", first is not None)
    gen.close()
    _check("消费者提前停止显式记录",
           acc.generation_end_reason == llm.GenerationEndReason.CONSUMER_STOP)
    _consistent(llm, acc, "after-consumer-stop")

    logger.info("=" * 60)
    logger.info("ALL PASS ✓")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
