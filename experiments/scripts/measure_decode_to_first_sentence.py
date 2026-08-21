#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R6 补测：LLM 解码至首个句末标点的延迟（T_decode_to_first_sentence，TTFA 预算表分项）。

背景：TTFA 预算（计划 §7.3）= T_endpoint(E5) + TTFT(已有) + T_decode_to_first_sentence(本脚本)
+ T_TTS_first_chunk(E6)。E4 复跑（max_tokens=128，full_response 落盘）未记录逐 token 时刻，
generate() 也无逐 token 日志，该分项无法从既有产物恢复，故独立补测。

方法学要点（供审查）：
- 待测量是纯 LLM 解码段时延，与 ASR/VAD 管线解耦——管线中首 token 之后的解码在
  GPU 独占状态下进行，独立喂同一提示文本测量同一数量成立；
- 输入取 E4 落盘的 50 条 streaming 模式 transcribed_text（即 LLM 当时实际收到的用户输入），
  预填/解码走生产同款 StreamLLMInference.cache_prompt(is_end=True) + generate() 路径，
  同模型同权重同设备（默认 cuda:1，与生产一致）；max_tokens=128 与 E4 一致；
- 逐 token 时刻在 generate() 每次 yield 处记录（yield 紧接该 token 解码完成）；
- 句末判定：首个 。！？!? 之一；英文 '.' 仅在非数字夹击中计数（豁免 3.5 类小数）；
- 一次性预填与生产的增量预填在提示内容上等价（chat template 相同），
  差异只影响预填段（TTFT 已在 E4/E5 实测），不影响本脚本测量的解码段。

用法（GPU 主机）：
  uv run python -m experiments.scripts.measure_decode_to_first_sentence \
      --llm-device cuda:1 --output experiments/results/revision/r6_ttfa/decode_to_first_sentence.csv
本机自检（无模型依赖，验证检测/计时/CSV/汇总链路）：
  uv run python -m experiments.scripts.measure_decode_to_first_sentence --self-test
"""

import argparse
import csv
import glob
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

SENTENCE_END_CHARS = "。！？!?"


def detect_first_sentence_end(text: str) -> int:
    """返回首个句末标点的字符下标；无则 -1。'.' 夹在数字间（如 3.5）不算句末。"""
    for i, ch in enumerate(text):
        if ch in SENTENCE_END_CHARS:
            return i
        if ch == ".":
            prev = text[i - 1] if i > 0 else ""
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if not (prev.isdigit() and nxt.isdigit()):
                return i
    return -1


def lang_of(sample_id: str) -> str:
    prefix = sample_id.split("_")[0]
    return {"crosswoz": "zh", "aishell1": "zh", "multiwoz": "en", "librispeech": "en"}.get(prefix, "zh")


def run_one(llm, prompt_text: str, max_tokens: int) -> dict:
    """对一条提示执行 预填+解码，逐 token 计时，返回测量记录。
    对真实 StreamLLMInference 与 self-test 假 LLM 通用（协议：cache_prompt/generate）。"""
    kv = llm.cache_prompt(prompt_text, is_end=True)
    t_start = time.perf_counter()
    first_token_t = None
    acc = ""
    sent_char_idx = -1
    sent_token_idx = -1
    sent_t = None
    n_tokens = 0
    for tok in llm.generate(pre_cache=kv, max_new_tokens=max_tokens):
        now = time.perf_counter()
        if first_token_t is None:
            first_token_t = now
        acc += tok
        if sent_token_idx < 0:
            ci = detect_first_sentence_end(acc)
            if ci >= 0:
                sent_char_idx, sent_token_idx, sent_t = ci, n_tokens, now
        n_tokens += 1
    t_end = time.perf_counter()
    if first_token_t is None:
        raise RuntimeError("generate 未产出任何 token")
    decode_to_first_sentence_ms = ((sent_t if sent_t is not None else t_end) - first_token_t) * 1000
    return {
        "n_tokens": n_tokens,
        "ttft_ms": (first_token_t - t_start) * 1000,  # 独立一次性预填口径，仅供参考
        "response_chars": len(acc),
        "sentence_end_found": int(sent_token_idx >= 0),
        "first_sentence_char_idx": sent_char_idx,
        "first_sentence_token_idx": sent_token_idx if sent_token_idx >= 0 else n_tokens - 1,
        "decode_to_first_sentence_ms": decode_to_first_sentence_ms,
        "decode_total_ms": (t_end - first_token_t) * 1000,
        "tokens_per_s": n_tokens / max(t_end - t_start, 1e-9),
        "first_sentence_text": acc[:sent_char_idx + 1] if sent_char_idx >= 0 else acc,
    }


CSV_FIELDS = ["sample_id", "language", "pass_idx", "n_tokens", "ttft_ms",
              "response_chars", "sentence_end_found", "first_sentence_char_idx",
              "first_sentence_token_idx", "decode_to_first_sentence_ms",
              "decode_total_ms", "tokens_per_s", "first_sentence_text"]


def write_outputs(rows: list, output_csv: Path) -> str:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    lines = []
    scopes = [("overall", rows)] + [(f"lang:{lg}", [r for r in rows if r["language"] == lg])
                                    for lg in sorted({r["language"] for r in rows})]
    for scope, rs in scopes:
        if not rs:
            continue
        v = np.array([r["decode_to_first_sentence_ms"] for r in rs])
        rate = np.array([r["tokens_per_s"] for r in rs])
        found = sum(r["sentence_end_found"] for r in rs)
        lines.append(f"{scope}: n={len(rs)} sentence_end_found={found}/{len(rs)} "
                     f"decode_to_first_sentence mean={v.mean():.1f}ms std={v.std():.1f} "
                     f"p50={np.percentile(v, 50):.1f} p90={np.percentile(v, 90):.1f} | "
                     f"decode rate mean={rate.mean():.1f} tok/s")
    summary = "\n".join(lines)
    summary_path = output_csv.with_suffix(".summary.txt")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    return summary


def load_e4_prompts(e4_results_glob: str) -> list:
    """读取 E4 结果中 streaming 模式的逐样本 LLM 输入（transcribed_text）。"""
    files = sorted(glob.glob(e4_results_glob))
    if not files:
        raise SystemExit(f"E4 结果未找到: {e4_results_glob}")
    path = files[-1]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    prompts = []
    for r in data["results"]:
        if r.get("mode") != "streaming" or r.get("error"):
            continue
        text = (r.get("transcribed_text") or "").strip()
        if not text:
            continue
        prompts.append({"sample_id": r["sample_id"], "language": lang_of(r["sample_id"]),
                        "prompt_text": text})
    if not prompts:
        raise SystemExit(f"{path} 中无有效 streaming 样本")
    logger.info(f"E4 提示文本来源: {path}（{len(prompts)} 条 streaming 样本）")
    return prompts


class _FakeLLM:
    """self-test 用：按脚本化 token 序列 yield，token 间固定小睡眠。"""

    def __init__(self, tokens, sleep_s=0.01):
        self._tokens = tokens
        self._sleep = sleep_s

    def cache_prompt(self, text, pre_cache=None, is_end=False, **kw):
        return object()

    def generate(self, pre_cache=None, max_new_tokens=50, **kw):
        for t in self._tokens[:max_new_tokens]:
            time.sleep(self._sleep)
            yield t


def self_test() -> int:
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not cond:
            failures += 1

    # 1. 句末检测：中文标点 / 英文句号 / 小数豁免 / 无句末
    check("中文句末", detect_first_sentence_end("你好，世界。后续") == 5)
    check("英文句末", detect_first_sentence_end("Sure. Next") == 4)
    check("小数豁免", detect_first_sentence_end("价格是3.5元。好") == 7)
    check("无句末", detect_first_sentence_end("没有标点") == -1)

    # 2. 计时链路：假 LLM（含句末）
    toks = ["你好", "，", "这", "是", "测", "试", "。", "后", "续", "回", "复"]
    rec = run_one(_FakeLLM(toks, sleep_s=0.01), "提示", max_tokens=128)
    check("首句 token 定位", rec["first_sentence_token_idx"] == 6 and rec["sentence_end_found"] == 1,
          f"idx={rec['first_sentence_token_idx']}")
    check("首句文本", rec["first_sentence_text"] == "你好，这是测试。")
    check("解码至首句 < 总解码", 0 < rec["decode_to_first_sentence_ms"] < rec["decode_total_ms"])
    check("ttft 为正", rec["ttft_ms"] > 0)

    # 3. 无句末标点：回退为整段解码时间
    rec2 = run_one(_FakeLLM(["没", "有", "标", "点"], sleep_s=0.005), "提示", max_tokens=128)
    check("无句末回退", rec2["sentence_end_found"] == 0
          and abs(rec2["decode_to_first_sentence_ms"] - rec2["decode_total_ms"]) < 1e-6)

    # 4. CSV + 汇总链路
    rows = [{"sample_id": "crosswoz_s1", "language": "zh", "pass_idx": 0, **rec},
            {"sample_id": "multiwoz_s2", "language": "en", "pass_idx": 0, **rec2}]
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "sub" / "decode.csv"
        summary = write_outputs(rows, out)
        check("CSV 写出", out.exists() and len(list(csv.DictReader(open(out, encoding="utf-8")))) == 2)
        check("汇总含分组", "lang:zh" in summary and "lang:en" in summary and "overall" in summary)
        check("summary 文件", out.with_suffix(".summary.txt").exists())

    print(f"\nself-test {'4/4 组通过' if failures == 0 else f'{failures} 项失败'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="R6 补测：LLM 解码至首个句末标点延迟")
    parser.add_argument("--e4-results", type=str,
                        default=str(PROJECT_ROOT / "experiments/results/revision/r4_commit/exp1_results_*.json"),
                        help="E4 结果 JSON（glob；取最新一个）")
    parser.add_argument("--llm-model-name", type=str, default=None)
    parser.add_argument("--llm-device", type=str, default="cuda:1")
    parser.add_argument("--max-tokens", type=int, default=128, help="与 E4 一致")
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None, help="冒烟用（如 3）")
    parser.add_argument("--repeat", type=int, default=1, help="每条提示重复测量轮数")
    parser.add_argument("--output", type=str,
                        default="experiments/results/revision/r6_ttfa/decode_to_first_sentence.csv")
    parser.add_argument("--self-test", action="store_true", help="无模型自检后退出")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_global_log_level(args.log_level)

    if args.self_test:
        sys.exit(self_test())

    from src.config import LLM_MODEL_NAME
    from src.llm.stream_llm_inference import StreamLLMInference

    prompts = load_e4_prompts(args.e4_results)
    if args.max_samples:
        prompts = prompts[: args.max_samples]

    llm = StreamLLMInference(model_name=args.llm_model_name or LLM_MODEL_NAME,
                             device=args.llm_device, eval_mode=False)
    for i in range(args.warmup_rounds):
        kv = llm.cache_prompt("你好，这是一个测试。", is_end=True)
        for _ in llm.generate(pre_cache=kv, max_new_tokens=10):
            pass
        del kv
    logger.info(f"预热完成（{args.warmup_rounds} 轮）")

    rows = []
    for pass_idx in range(args.repeat):
        for i, p in enumerate(prompts):
            rec = run_one(llm, p["prompt_text"], args.max_tokens)
            rows.append({"sample_id": p["sample_id"], "language": p["language"],
                         "pass_idx": pass_idx, **rec})
            logger.info(f"[pass{pass_idx} {i + 1}/{len(prompts)}] {p['sample_id']}: "
                        f"decode_to_first_sentence={rec['decode_to_first_sentence_ms']:.1f}ms "
                        f"({rec['first_sentence_token_idx'] + 1}/{rec['n_tokens']} tokens)")

    output_csv = PROJECT_ROOT / args.output
    summary = write_outputs(rows, output_csv)
    print(f"\n{summary}")
    logger.info(f"结果已保存: {output_csv}")


if __name__ == "__main__":
    main()
