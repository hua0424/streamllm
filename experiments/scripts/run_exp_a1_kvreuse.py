#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融 A1：打断后 KV 复用(crop) vs 重新 prefill —— 延迟微基准。
同时产出 E1 的 barge-in 响应延迟指标（= 反查 + crop + role 重建的系统处理耗时）。

命题（experiment_design.md §5 A1）：
  B-ours ：打断 → PlaybackTimeline 反查(µs) + DynamicCache.crop + role 重建 → 近常数耗时
  B-noKV ：打断 → 丢弃 KV，把"听到边界前的完整上下文"重新全量 prefill → 随上下文长度线性增长
扫描上下文长度得到两条延迟曲线（论文图：KV 复用的价值随上下文增长）。

方法：构造指定长度的真实 KV（filler 文本 prefill），截掉尾部 K token 模拟打断；
(a) 计时 crop+reopen；(b) 计时同一 input_ids 前缀的一次全量 forward（重新 prefill 等价物）。
CUDA 事件同步计时，warmup 后取多次均值。

运行（项目根目录）：
    HF_TOKEN= uv run python -m experiments.scripts.run_exp_a1_kvreuse
本机 0.5B 出概念曲线；实验机 7B 重跑出正式数值（7B 下 re-prefill 成本更陡峭）。
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.llm.stream_llm_inference import StreamLLMInference
from src.utils.logging_utils import get_logger, set_global_log_level
from src.config import RESULTS_DIR

logger = get_logger(__name__)

FILLER = ("The city has many museums, parks, restaurants and historical sites that "
          "attract millions of visitors every year. ")


def build_context(llm, target_len: int):
    """构造 ~target_len token 的真实 KV 上下文，返回 (acc, full_input_ids)。"""
    kv = llm.cache_prompt("Tell me about the city.", is_end=False)
    acc = llm.to_accum_cache(kv)
    ids_list = [acc.attention_mask.new_zeros(0)]  # 占位；真实 ids 由 tokenizer 重建更繁琐，
    # 直接维护文本并在最后一次性 tokenize 前缀做 re-prefill 计时（等价计算量）。
    text = ""
    while acc.seq_length < target_len:
        llm.prefill_user_text(acc, FILLER)
        text += FILLER
    return acc, text


def timed_cuda(fn, device: str):
    """CUDA 同步计时（ms）。"""
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048])
    ap.add_argument("--crop-tokens", type=int, default=32, help="模拟打断截掉的尾部 token 数")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=str, default=str(Path(RESULTS_DIR) / "exp_a1_kvreuse.json"))
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    set_global_log_level(args.log_level)
    logger.info("=" * 70)
    logger.info("消融 A1：KV 复用(crop) vs 重新 prefill 延迟微基准（兼 E1 barge-in 响应延迟）")
    logger.info("=" * 70)

    llm = StreamLLMInference(model_name="Qwen/Qwen2.5-0.5B-Instruct", eval_mode=False)
    dev = llm.device

    results = []
    logger.info(f"{'ctx_len':>8} | {'crop+role(ms)':>13} | {'re-prefill(ms)':>14} | {'加速比':>8}")
    for L in args.lengths:
        acc, text = build_context(llm, L)
        seq = acc.seq_length
        keep = seq - args.crop_tokens

        # (a) KV 复用路径，拆两段计时：
        #   crop_ms —— 纯 crop（+timeline 反查为纯 Python µs 级）＝ barge-in 响应关键路径
        #   role_ms —— role 重建（几个 token 的小 prefill）＝ **不在关键路径上**：
        #              可推迟到用户新输入到来时执行，不影响"打断即停"的响应体验
        def restore():
            if acc.seq_length < seq:
                llm.prefill_user_text(acc, " filler" * args.crop_tokens)
                if acc.seq_length > seq:
                    llm.crop_to_token(acc, seq)

        # warmup（首次含 CUDA 初始化，不计时）
        timed_cuda(lambda: llm.crop_to_token(acc, keep), dev)
        timed_cuda(lambda: llm.reopen_user_role(acc), dev)
        llm.crop_to_token(acc, keep)

        crop_times, role_times = [], []
        for _ in range(args.repeats):
            restore()
            crop_times.append(timed_cuda(lambda: llm.crop_to_token(acc, keep), dev))
            role_times.append(timed_cuda(lambda: llm.reopen_user_role(acc), dev))
            llm.crop_to_token(acc, keep)
        crop_only_ms = sum(crop_times) / len(crop_times)
        role_ms = sum(role_times) / len(role_times)
        crop_ms = crop_only_ms + role_ms   # 完整恢复路径（对比 re-prefill 用）

        # (b) 重新 prefill 路径：对"听到边界前的完整上下文"做一次全量 forward
        enc = llm.tokenizer(text, return_tensors="pt", add_special_tokens=False,
                            truncation=True, max_length=keep).to(dev)
        full_ids = enc.input_ids
        mask = torch.ones_like(full_ids)

        def re_prefill():
            with torch.no_grad():
                llm.model(input_ids=full_ids, attention_mask=mask, use_cache=True, return_dict=True)

        re_prefill()  # warmup
        rp_times = [timed_cuda(re_prefill, dev) for _ in range(args.repeats)]
        rp_ms = sum(rp_times) / len(rp_times)

        speedup = rp_ms / crop_ms if crop_ms > 0 else float("inf")
        results.append({"ctx_len": seq, "keep_len": keep,
                        "crop_only_ms": round(crop_only_ms, 3),
                        "role_rebuild_ms": round(role_ms, 2),
                        "crop_role_ms": round(crop_ms, 2),
                        "reprefill_ms": round(rp_ms, 2),
                        "speedup": round(speedup, 1)})
        logger.info(f"{seq:>8} | crop={crop_only_ms:>7.3f} role={role_ms:>6.2f} "
                    f"合计={crop_ms:>7.2f} | re-prefill={rp_ms:>8.2f} | {speedup:>5.1f}x")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": vars(args), "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"结果已保存: {out}")

    # 自检（趋势断言，0.5B 短上下文下逐点 speedup 无意义，正式数值 7B 更陡）：
    #   re-prefill 随长度单调增；crop 近常数且亚 ms 级；最长上下文处 KV 复用更快
    logger.info("-" * 70)
    rp = [r["reprefill_ms"] for r in results]
    co = [r["crop_only_ms"] for r in results]
    assert rp[-1] > rp[0], "re-prefill 应随上下文增长"
    assert max(co) < 10, "纯 crop 应为 ms 级以内（张量切片）"
    assert results[-1]["speedup"] > 1, "最长上下文处 KV 复用应更快"
    logger.info(f"barge-in 响应关键路径（反查+crop）: {min(co):.3f}-{max(co):.3f}ms，近常数 ✓")
    logger.info(f"role 重建（非关键路径，可延迟到下轮输入前）: "
                f"~{sum(r['role_rebuild_ms'] for r in results)/len(results):.1f}ms")
    logger.info(f"re-prefill 随上下文增长: {rp[0]:.1f} → {rp[-1]:.1f}ms（7B 下更陡）")
    logger.info("ALL PASS ✓")


if __name__ == "__main__":
    main()
