#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R4 §5.2：分词接缝不匹配率 —— 纯离线分析（只需 tokenizer，不需模型权重）。

复现 System B 的 LLM 侧实际分词路径（src/llm/stream_llm_inference.py）：
  ids_stream = tokenize(template_prefix + f1)                    # _init_kv_cache，首次
             + tokenize(f2, add_special_tokens=False) + ...    # _add_stream_prompt 逐片段
             + tokenize(generation_prompt, add_special_tokens=False)  # is_end 收尾
对照一次性分词：
  ids_oneshot = tokenize(chat_template([system, user="".join(fragments)], add_generation_prompt=True))

其中 template_prefix / generation_prompt 的推导与生产代码逐行一致
（apply_chat_template + "提取提示符" 定位后缀，stream_llm_inference.py:124-139, 168-190）。

输入：E4 结果的 committed_fragments（生产实际提交片段序列）。
输出：tokenizer_seams.csv（逐样本 len_diff / 首个分歧位置 / 受影响 token 数）+ 汇总行。

用法：
  HF_TOKEN= HF_HOME=<本地缓存> uv run python -m experiments.scripts.check_tokenizer_seams
  uv run python -m experiments.scripts.check_tokenizer_seams --self-test   # 无网络自检
"""

import argparse
import csv
import glob
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant responding in Chinese."  # cache_prompt 默认值
_INIT_USER_TEXT = "提取提示符"  # 生产代码用于定位 generation_prompt 的临时 user 内容


def build_prompt_parts(tokenizer):
    """与 StreamLLMInference.__init__/cache_prompt 逐行一致的模板拆分。"""
    temp_messages = [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": _INIT_USER_TEXT}]
    full_template = tokenizer.apply_chat_template(temp_messages, tokenize=False,
                                                  add_generation_prompt=True)
    index = full_template.find(_INIT_USER_TEXT)
    generation_prompt = full_template[index + len(_INIT_USER_TEXT):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ""}]
    full_prompt_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
    prefix = full_prompt_text.replace(generation_prompt, "")
    return prefix, generation_prompt


def stream_ids(tokenizer, fragments):
    """生产增量分词路径复现。首次调用与 _init_kv_cache 同口径（默认 add_special_tokens，
    Qwen2 不添加特殊 token，与 False 等价），后续片段与 _add_stream_prompt 同（False）。"""
    prefix, generation_prompt = build_prompt_parts(tokenizer)
    ids = list(tokenizer(prefix + fragments[0]).input_ids)
    for frag in fragments[1:]:
        ids += list(tokenizer(frag, add_special_tokens=False).input_ids)
    ids += list(tokenizer(generation_prompt, add_special_tokens=False).input_ids)
    return ids


def oneshot_ids(tokenizer, fragments):
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "".join(fragments)}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return list(tokenizer(text).input_ids)


def compare_ids(ids_stream: list, ids_oneshot: list) -> dict:
    """比较两条 token 序列（difflib 对齐，不用逐位 zip——分歧块后序列会重新对齐）。

    返回：首个分歧位置（-1=完全一致）、分歧块数（=接缝点数）、两侧各自的分歧 token 数、
    长度差。实测分歧形态为 BPE 跨缝合并（'.'+'Is'→'.Is'），分歧后序列重新对齐。
    """
    import difflib
    sm = difflib.SequenceMatcher(a=ids_stream, b=ids_oneshot, autojunk=False)
    first_diff = -1
    n_blocks = 0
    s_tokens = 0
    o_tokens = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        n_blocks += 1
        s_tokens += i2 - i1
        o_tokens += j2 - j1
        if first_diff < 0:
            first_diff = i1
    return {"len_stream": len(ids_stream), "len_oneshot": len(ids_oneshot),
            "len_diff": len(ids_stream) - len(ids_oneshot),
            "first_diff_pos": first_diff, "seam_mismatch": int(n_blocks > 0),
            "n_diff_blocks": n_blocks, "stream_diff_tokens": s_tokens,
            "oneshot_diff_tokens": o_tokens}


CSV_FIELDS = ["sample_id", "language", "n_fragments", "len_stream", "len_oneshot",
              "len_diff", "first_diff_pos", "seam_mismatch", "n_diff_blocks",
              "stream_diff_tokens", "oneshot_diff_tokens", "decoded_text_identical"]


def summarize(rows: list) -> str:
    n = len(rows)
    mm = [r for r in rows if r["seam_mismatch"]]
    lines = [f"samples={n} mismatch={len(mm)} ({len(mm) / max(n, 1) * 100:.1f}%)"]
    if mm:
        blocks = [r["n_diff_blocks"] for r in mm]
        toks = [r["oneshot_diff_tokens"] for r in mm]
        lines.append(f"diff_blocks(接缝点): mean={np.mean(blocks):.2f} median={np.median(blocks):.1f} "
                     f"max={max(blocks)}")
        lines.append(f"oneshot_diff_tokens: mean={np.mean(toks):.2f} median={np.median(toks):.1f} "
                     f"max={max(toks)}")
    ident = sum(1 for r in rows if r.get("decoded_text_identical", 1))
    lines.append(f"decoded_text_identical: {ident}/{n}")
    lines.append(f"len_diff: mean={np.mean([r['len_diff'] for r in rows]):.3f}")
    return "\n".join(lines)


def lang_of(sample_id: str) -> str:
    return {"crosswoz": "zh", "aishell1": "zh", "multiwoz": "en",
            "librispeech": "en"}.get(sample_id.split("_")[0], "zh")


def self_test() -> int:
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not cond:
            failures += 1

    check("完全一致", compare_ids([1, 2, 3], [1, 2, 3])["seam_mismatch"] == 0)
    r = compare_ids([1, 2, 3, 4, 5], [1, 2, 9, 5])
    check("中部分歧", r["first_diff_pos"] == 2 and r["n_diff_blocks"] == 1
          and r["len_diff"] == 1, str(r))
    r2 = compare_ids([1, 2], [1, 2, 3])
    check("前缀一致长度不同", r2["first_diff_pos"] == 2 and r2["oneshot_diff_tokens"] == 1)
    # BPE 跨缝合并形态：2 token 换 1 token，分歧后重新对齐（zip 口径会错报 3+）
    r3 = compare_ids([10, 20, 30, 40, 50], [10, 99, 30, 40, 50])
    check("合并后重对齐", r3["first_diff_pos"] == 1 and r3["n_diff_blocks"] == 1
          and r3["stream_diff_tokens"] == 1 and r3["oneshot_diff_tokens"] == 1, str(r3))
    r4 = compare_ids([1, 2, 3, 4, 5, 6], [1, 9, 3, 4, 8, 6])
    check("两个分歧块", r4["n_diff_blocks"] == 2 and r4["stream_diff_tokens"] == 2)
    rows = [{"sample_id": "a", "language": "zh", "n_fragments": 2,
             "decoded_text_identical": 1, **compare_ids([1, 2, 3], [1, 2, 3])},
            {"sample_id": "b", "language": "zh", "n_fragments": 3,
             "decoded_text_identical": 1, **compare_ids([1, 2, 3], [1, 9, 3])}]
    s = summarize(rows)
    check("汇总", "mismatch=1 (50.0%)" in s and "diff_blocks" in s
          and "decoded_text_identical: 2/2" in s, s)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "seams.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        check("CSV 写出", len(list(csv.DictReader(open(out, encoding="utf-8")))) == 2)

    print(f"\nself-test {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="R4 §5.2 分词接缝不匹配率（离线）")
    parser.add_argument("--e4-results", type=str,
                        default=str(PROJECT_ROOT / "experiments/results/revision/r4_commit/exp1_results_*.json"))
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="默认取 src.config LLM_MODEL_NAME（Qwen/Qwen2-7B-Instruct）")
    parser.add_argument("--output", type=str,
                        default="experiments/results/revision/r4_commit/tokenizer_seams.csv")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_global_log_level(args.log_level)

    if args.self_test:
        sys.exit(self_test())

    from src.config import LLM_MODEL_NAME
    from transformers import AutoTokenizer

    files = sorted(glob.glob(args.e4_results))
    if not files:
        raise SystemExit(f"E4 结果未找到: {args.e4_results}")
    path = files[-1]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = []
    for r in data["results"]:
        if r.get("mode") != "streaming" or r.get("error"):
            continue
        frags = r.get("committed_fragments")
        if not frags or not any(f.strip() for f in frags):
            raise SystemExit(f"样本 {r.get('sample_id')} 缺少 committed_fragments（{path}）")
        samples.append({"sample_id": r["sample_id"], "fragments": frags})
    logger.info(f"输入: {path}（{len(samples)} 条 streaming 样本）")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or LLM_MODEL_NAME)
    logger.info(f"tokenizer 加载完成: {args.tokenizer or LLM_MODEL_NAME}")

    rows = []
    for i, s in enumerate(samples):
        ids_s = stream_ids(tokenizer, s["fragments"])
        ids_o = oneshot_ids(tokenizer, s["fragments"])
        cmp_ = compare_ids(ids_s, ids_o)
        # 语义影响核验：两条 token 序列解码回的文本必须一致（分歧仅为 BPE 重切分）
        text_identical = int(tokenizer.decode(ids_s) == tokenizer.decode(ids_o))
        rows.append({"sample_id": s["sample_id"], "language": lang_of(s["sample_id"]),
                     "n_fragments": len(s["fragments"]),
                     "decoded_text_identical": text_identical, **cmp_})
        if cmp_["seam_mismatch"]:
            logger.info(f"  接缝分歧 {s['sample_id']}: first_diff={cmp_['first_diff_pos']} "
                        f"blocks={cmp_['n_diff_blocks']} len_diff={cmp_['len_diff']} "
                        f"text_identical={text_identical}")

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    out.with_suffix(".summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\n{summary}")
    logger.info(f"已保存: {out}")


if __name__ == "__main__":
    main()
