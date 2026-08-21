#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R6 补测：LLM 解码至首个句末标点的延迟（T_decode_to_first_sentence，TTFA 预算表分项）。

背景：TTFA 预算（计划 §7.3）= T_endpoint(E5) + TTFT(已有) + T_decode_to_first_sentence(本脚本)
+ T_TTS_first_chunk(E6)。E4 复跑（max_tokens=128，full_response 落盘）未记录逐 token 时刻，
generate() 也无逐 token 日志，该分项无法从既有产物恢复，故独立补测。

方法学要点（2026-08-21 审查意见 P0 修正后：fragment replay 口径）：
- 待测量是纯 LLM 解码段时延，与 ASR/VAD 管线解耦——管线中首 token 之后的解码在
  GPU 独占状态下进行，独立重放同一提示序列测量同一数量成立；
- 输入重放 **E4 落盘的 `committed_fragments` 片段序列**，复现生产调用序列
  （run_exp_latency.py:731-734, 760-762）：
  对每个片段依次 cache_prompt(fragment, pre_cache=kv, is_end=False)，
  最后 cache_prompt("", pre_cache=kv, is_end=True) 加生成提示，再 generate()。
  不用 transcribed_text——那是 " ".join() 的重构，插入空格且丢失片段边界，
  与生产增量预填的 token 序列不保证一致（审查 P0）；
- 同模型（Qwen/Qwen2-7B-Instruct）同设备（默认 cuda:1）同 max_tokens=128（与 E4 一致）；
- 逐 token 时刻在 generate() 每次 yield 处记录（yield 紧接该 token 解码完成）；
  本项 = 首个句末标点 token 的 yield 时刻 − 首个 token 的 yield 时刻，不含预填时间；
- 句末判定：首个 。！？!? 之一；英文 '.' 仅在非数字夹击中计数（豁免 3.5 类小数）；
  无句末标点的回复回退为整段解码时间并置 sentence_end_found=0。

审计与验收（2026-08-21 审查意见 P1）：
- 输入强制校验：恰好 --expected-samples 条（默认 50，--max-samples 冒烟模式豁免）、
  sample_id 唯一、E4 config 的 llm_model/max_tokens/llm_device 与本次一致；任一不满足即退出；
- 逐样本异常捕获写入 error 列，测量行逐条追加 checkpoint JSONL（进程中断不丢已完成测量）；
- 正式输出含 RUNINFO（命令、commit、输入文件+哈希、清单哈希、生成参数、起止耗时、重放模式）；
- 失败样本不进汇总；正式验收要求 error 全空且样本 ID 集合完整。

用法（GPU 主机）：
  uv run python -m experiments.scripts.measure_decode_to_first_sentence \
      --llm-device cuda:1 --output experiments/results/revision/r6_ttfa/decode_to_first_sentence.csv
本机自检（无模型依赖）：
  uv run python -m experiments.scripts.measure_decode_to_first_sentence --self-test
"""

import argparse
import csv
import glob
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_utils import get_logger, set_global_log_level

logger = get_logger(__name__)

SENTENCE_END_CHARS = "。！？!?"
# 生产 generate() 的默认采样参数（run_exp_latency 未显式覆盖），记录进 RUNINFO 供审计
GEN_PARAMS = {"temperature": 0.1, "top_p": 0.9, "repetition_penalty": 1.1}
REPLAY_MODE = "fragment_replay"  # committed_fragments 增量重放（审查 P0 修正）


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


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def run_one(llm, fragments: list, max_tokens: int) -> dict:
    """按生产调用序列重放增量预填，再逐 token 计时解码，返回测量记录。

    对真实 StreamLLMInference 与 self-test 假 LLM 通用（协议：cache_prompt/generate）。
    """
    if not fragments:
        raise ValueError("committed_fragments 为空，无法重放生产调用序列")
    kv = None
    for frag in fragments:
        kv = llm.cache_prompt(frag, pre_cache=kv, is_end=False)
    kv = llm.cache_prompt("", pre_cache=kv, is_end=True)  # 生产收尾：空片段 + is_end=True
    prompt_tokens = int(kv.pre_input_ids.shape[-1])  # 生成前完整 prompt 长度（审计用）

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
        "fragment_count": len(fragments),
        "fragments_sha256": _sha256_text(json.dumps(fragments, ensure_ascii=False)),
        "prompt_tokens": prompt_tokens,
        "n_tokens": n_tokens,
        "first_token_latency_ms": (first_token_t - t_start) * 1000,  # 重放口径首 token，非预算项 TTFT
        "response_chars": len(acc),
        "sentence_end_found": int(sent_token_idx >= 0),
        "first_sentence_char_idx": sent_char_idx,
        "first_sentence_token_idx": sent_token_idx if sent_token_idx >= 0 else n_tokens - 1,
        "decode_to_first_sentence_ms": decode_to_first_sentence_ms,
        "decode_total_ms": (t_end - first_token_t) * 1000,
        "tokens_per_s": n_tokens / max(t_end - t_start, 1e-9),
        "first_sentence_text": acc[:sent_char_idx + 1] if sent_char_idx >= 0 else acc,
        "error": "",
    }


CSV_FIELDS = ["sample_id", "language", "pass_idx", "fragment_count", "fragments_sha256",
              "prompt_tokens", "n_tokens", "first_token_latency_ms",
              "response_chars", "sentence_end_found", "first_sentence_char_idx",
              "first_sentence_token_idx", "decode_to_first_sentence_ms",
              "decode_total_ms", "tokens_per_s", "first_sentence_text", "error"]

_IDENTITY_FIELDS = ("sample_id", "language", "pass_idx")


def measure_sample(llm, sample: dict, pass_idx: int, max_tokens: int) -> dict:
    """测量一个样本；异常时返回带 error 的完整行（身份字段必须保留，复审 r2 P1）。"""
    try:
        rec = run_one(llm, sample["fragments"], max_tokens)
    except Exception as e:
        logger.error(f"样本 {sample['sample_id']} 测量失败: {e}")
        rec = {k: "" for k in CSV_FIELDS if k not in _IDENTITY_FIELDS}
        rec.update({
            "fragment_count": len(sample["fragments"]),
            "fragments_sha256": _sha256_text(json.dumps(sample["fragments"], ensure_ascii=False)),
            "error": str(e),
        })
    return {"sample_id": sample["sample_id"], "language": sample["language"],
            "pass_idx": pass_idx, **rec}


def load_e4_samples(e4_results_glob: str, expected_llm_model: str, expected_max_tokens: int,
                    expected_llm_device: str, expected_count: int,
                    smoke_n: int = None) -> tuple:
    """读取 E4 结果中 streaming 模式的 committed_fragments，并做强制校验（审查 P1）。

    返回 (samples, meta)；samples 元素 {sample_id, language, fragments}。
    校验失败一律 SystemExit，不产出"看似完整"的结果。
    """
    files = sorted(glob.glob(e4_results_glob))
    if not files:
        raise SystemExit(f"E4 结果未找到: {e4_results_glob}")
    path = Path(files[-1])
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    meta = {
        "input_file": str(path),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "e4_config": data.get("config", {}),
    }

    # E4 配置校验：模型/max_tokens/LLM 设备必须与锁定方案一致
    cfg = data.get("config", {})
    problems = []
    if cfg.get("llm_model") != expected_llm_model:
        problems.append(f"llm_model={cfg.get('llm_model')} != {expected_llm_model}")
    if cfg.get("max_tokens") != expected_max_tokens:
        problems.append(f"max_tokens={cfg.get('max_tokens')} != {expected_max_tokens}")
    if cfg.get("llm_device") != expected_llm_device:
        problems.append(f"E4 llm_device={cfg.get('llm_device')} != 本次 {expected_llm_device}")
    if problems:
        raise SystemExit(f"E4 配置校验失败（{path}）：" + "；".join(problems))

    samples = []
    for r in data["results"]:
        if r.get("mode") != "streaming" or r.get("error"):
            continue
        fragments = r.get("committed_fragments")
        if not fragments or not any(f.strip() for f in fragments):
            # 审查 P0-5：无片段必须报错退出，不得静默退回 transcribed_text
            raise SystemExit(f"样本 {r.get('sample_id')} 缺少 committed_fragments（{path}）")
        samples.append({"sample_id": r["sample_id"], "language": lang_of(r["sample_id"]),
                        "fragments": fragments})

    ids = [s["sample_id"] for s in samples]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise SystemExit(f"sample_id 不唯一: {dup[:5]}（{path}）")
    if smoke_n is None and len(samples) != expected_count:
        raise SystemExit(f"正式模式要求恰好 {expected_count} 条 streaming 样本，实际 {len(samples)}（{path}）。"
                         f"若为冒烟请显式使用 --max-samples")
    if smoke_n is not None:
        samples = samples[:smoke_n]
    meta["sample_count"] = len(samples)
    meta["sample_ids_sha256"] = _sha256_text(json.dumps(sorted(ids), ensure_ascii=False))
    logger.info(f"E4 输入: {path}（{len(samples)} 条 streaming 样本，fragment replay）")
    return samples, meta


def write_outputs(rows: list, output_csv: Path) -> str:
    """写 CSV 与汇总（error 行不进汇总）。"""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    ok = [r for r in rows if not r["error"]]
    n_err = len(rows) - len(ok)
    lines = []
    scopes = [("overall", ok)] + [(f"lang:{lg}", [r for r in ok if r["language"] == lg])
                                  for lg in sorted({r["language"] for r in ok})]
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
    header = f"rows={len(rows)} ok={len(ok)} error={n_err}"
    summary = header + "\n" + "\n".join(lines)
    summary_path = output_csv.with_suffix(".summary.txt")
    summary_path.write_text(summary + "\n", encoding="utf-8")
    return summary


def write_runinfo(output_csv: Path, argv: list, meta: dict, args, t0: datetime, t1: datetime,
                  n_rows: int, n_err: int) -> Path:
    """写 RUNINFO（审查 P1：命令/commit/输入哈希/参数/计时/重放模式）。"""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         cwd=str(PROJECT_ROOT)).decode().strip()
    except Exception:
        commit = "unknown"
    runinfo = output_csv.with_suffix(".runinfo.md")
    runinfo.write_text(f"""# RUNINFO — decode_to_first_sentence 补测

- 命令: `{' '.join(argv)}`
- git commit: {commit}
- 起止: {t0.isoformat(timespec='seconds')} → {t1.isoformat(timespec='seconds')}（耗时 {(t1 - t0).total_seconds():.0f}s）
- 输入文件: {meta['input_file']}
  - sha256: {meta['input_sha256']}
  - E4 config: llm_model={meta['e4_config'].get('llm_model')}, max_tokens={meta['e4_config'].get('max_tokens')}, llm_device={meta['e4_config'].get('llm_device')}
- 样本: {meta['sample_count']} 条（streaming 模式），sample_ids sha256: {meta['sample_ids_sha256']}
- 重放模式: {REPLAY_MODE}（committed_fragments 增量 cache_prompt + 空片段 is_end=True 收尾）
- 参数: llm_model={args.llm_model_name or '(config 默认)'}, llm_device={args.llm_device},
  max_tokens={args.max_tokens}, warmup={args.warmup_rounds}, repeat={args.repeat},
  generate 采样参数 {GEN_PARAMS}
- 结果: rows={n_rows}, error={n_err}；CSV={output_csv.name}；summary={output_csv.with_suffix('.summary.txt').name}
- 计时口径: decode_to_first_sentence_ms = 首个句末标点 token yield 时刻 − 首个 token yield 时刻（不含预填）
""", encoding="utf-8")
    return runinfo


class _FakeKV:
    def __init__(self, n_tokens):
        self.pre_input_ids = type("FakeIds", (), {"shape": (1, n_tokens)})()


class _FakeLLM:
    """self-test 用：记录 cache_prompt 调用序列；按脚本化 token 序列 yield。"""

    def __init__(self, tokens, sleep_s=0.01):
        self._tokens = tokens
        self._sleep = sleep_s
        self.prompt_calls = []  # [(text, is_end)]

    def cache_prompt(self, text, pre_cache=None, is_end=False, **kw):
        self.prompt_calls.append((text, is_end))
        return _FakeKV(7)

    def generate(self, pre_cache=None, max_new_tokens=50, **kw):
        for t in self._tokens[:max_new_tokens]:
            time.sleep(self._sleep)
            yield t


def _fake_e4_json(path: Path, n=2, mode="streaming", with_fragments=True,
                  llm_model="Qwen/Qwen2-7B-Instruct", max_tokens=128, llm_device="cuda:1"):
    results = []
    for i in range(n):
        r = {"sample_id": f"crosswoz_s{i}", "mode": mode, "error": "",
             "transcribed_text": "重 构 文 本"}
        if with_fragments:
            r["committed_fragments"] = ["重", "构文本"]
        results.append(r)
    data = {"config": {"llm_model": llm_model, "max_tokens": max_tokens,
                       "llm_device": llm_device},
            "results": results}
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def self_test() -> int:
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not cond:
            failures += 1

    def expect_exit(name, fn):
        try:
            fn()
            check(name, False, "应 SystemExit 但未退出")
        except SystemExit:
            check(name, True)

    # 1. 句末检测：中文 / 英文 / 小数豁免 / 无句末
    check("中文句末", detect_first_sentence_end("你好，世界。后续") == 5)
    check("英文句末", detect_first_sentence_end("Sure. Next") == 4)
    check("小数豁免", detect_first_sentence_end("价格是3.5元。好") == 7)
    check("无句末", detect_first_sentence_end("没有标点") == -1)

    # 2. fragment replay 调用序列（审查 P0）：逐片段 is_end=False + 空片段 is_end=True 收尾
    frags = ["你好", "，这是", "测试。"]
    fake = _FakeLLM(["好", "。", "后", "续"], sleep_s=0.005)
    rec = run_one(fake, frags, max_tokens=128)
    check("重放调用序列", fake.prompt_calls == [("你好", False), ("，这是", False),
                                               ("测试。", False), ("", True)],
          str(fake.prompt_calls))
    check("fragment_count", rec["fragment_count"] == 3 and len(rec["fragments_sha256"]) == 64)
    check("prompt_tokens 记录", rec["prompt_tokens"] == 7)
    check("空片段拒绝", _raises(ValueError, lambda: run_one(fake, [], 128)))

    # 3. 计时链路
    check("首句 token 定位", rec["first_sentence_token_idx"] == 1 and rec["sentence_end_found"] == 1)
    check("首句文本", rec["first_sentence_text"] == "好。")
    check("解码至首句 < 总解码", 0 < rec["decode_to_first_sentence_ms"] < rec["decode_total_ms"])
    check("首 token 延迟为正", rec["first_token_latency_ms"] > 0)
    rec2 = run_one(_FakeLLM(["没", "有", "标", "点"], sleep_s=0.005), ["提示"], max_tokens=128)
    check("无句末回退", rec2["sentence_end_found"] == 0
          and abs(rec2["decode_to_first_sentence_ms"] - rec2["decode_total_ms"]) < 1e-6)

    # 3b. 异常行身份保留（复审 r2 P1）：失败行的 sample_id/language/pass_idx 不得被覆盖
    class _BoomLLM:
        def cache_prompt(self, text, pre_cache=None, is_end=False, **kw):
            raise RuntimeError("boom")

    err_row = measure_sample(_BoomLLM(),
                             {"sample_id": "crosswoz_bad1", "language": "zh",
                              "fragments": ["片", "段"]},
                             pass_idx=2, max_tokens=128)
    check("异常行身份保留", err_row["sample_id"] == "crosswoz_bad1"
          and err_row["language"] == "zh" and err_row["pass_idx"] == 2,
          f"sample_id={err_row['sample_id']!r} pass_idx={err_row['pass_idx']!r}")
    check("异常行 error 与片段审计", err_row["error"] == "boom"
          and err_row["fragment_count"] == 2 and len(err_row["fragments_sha256"]) == 64)

    # 4. 输入校验（审查 P1）：正常 / 缺片段退出 / 数量不符退出 / 重复 ID 退出 / 配置不符退出
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        good = tp / "exp1_results_good.json"
        _fake_e4_json(good, n=2)
        samples, meta = load_e4_samples(str(good), "Qwen/Qwen2-7B-Instruct", 128, "cuda:1", 2)
        check("正常加载", len(samples) == 2 and samples[0]["fragments"] == ["重", "构文本"])
        check("meta 哈希", len(meta["input_sha256"]) == 64 and len(meta["sample_ids_sha256"]) == 64)
        check("冒烟截断", len(load_e4_samples(str(good), "Qwen/Qwen2-7B-Instruct", 128, "cuda:1", 2,
                                              smoke_n=1)[0]) == 1)

        nofrag = tp / "exp1_results_nofrag.json"
        _fake_e4_json(nofrag, n=2, with_fragments=False)
        expect_exit("缺 fragments 退出",
                    lambda: load_e4_samples(str(nofrag), "Qwen/Qwen2-7B-Instruct", 128, "cuda:1", 2))
        expect_exit("数量不符退出",
                    lambda: load_e4_samples(str(good), "Qwen/Qwen2-7B-Instruct", 128, "cuda:1", 50))
        expect_exit("配置不符退出",
                    lambda: load_e4_samples(str(good), "Qwen/Qwen2-7B-Instruct", 50, "cuda:1", 2))

        dup = tp / "exp1_results_dup.json"
        _fake_e4_json(dup, n=2)
        d = json.loads(dup.read_text(encoding="utf-8"))
        d["results"][1]["sample_id"] = "crosswoz_s0"
        dup.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        expect_exit("重复 ID 退出",
                    lambda: load_e4_samples(str(dup), "Qwen/Qwen2-7B-Instruct", 128, "cuda:1", 2))

        # 5. CSV + 汇总（error 行不进汇总）+ RUNINFO
        rows = [{"sample_id": "crosswoz_s0", "language": "zh", "pass_idx": 0, **rec},
                {"sample_id": "crosswoz_s1", "language": "zh", "pass_idx": 0, **rec2},
                err_row]
        out = tp / "sub" / "decode.csv"
        summary = write_outputs(rows, out)
        check("CSV 写出", out.exists() and len(list(csv.DictReader(open(out, encoding="utf-8")))) == 3)
        csv_rows = list(csv.DictReader(open(out, encoding="utf-8")))
        check("CSV 异常行身份保留", csv_rows[2]["sample_id"] == "crosswoz_bad1"
              and csv_rows[2]["pass_idx"] == "2" and csv_rows[2]["error"] == "boom")
        check("error 行不进汇总", "n=2" in summary and "error=1" in summary)
        args_ns = argparse.Namespace(llm_model_name=None, llm_device="cuda:1", max_tokens=128,
                                     warmup_rounds=3, repeat=1)
        ri = write_runinfo(out, ["prog", "--self-test"], meta, args_ns,
                           datetime.now(), datetime.now(), 3, 1)
        content = ri.read_text(encoding="utf-8")
        check("RUNINFO 关键字段", all(k in content for k in
              ("git commit", "sha256", REPLAY_MODE, "max_tokens", "error=1")))

    print(f"\nself-test {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 1 if failures else 0


def _raises(exc_type, fn) -> bool:
    try:
        fn()
        return False
    except exc_type:
        return True


def main():
    parser = argparse.ArgumentParser(description="R6 补测：LLM 解码至首个句末标点延迟（fragment replay）")
    parser.add_argument("--e4-results", type=str,
                        default=str(PROJECT_ROOT / "experiments/results/revision/r4_commit/exp1_results_*.json"),
                        help="E4 结果 JSON（glob；取最新一个）")
    parser.add_argument("--llm-model-name", type=str, default=None)
    parser.add_argument("--llm-device", type=str, default="cuda:1")
    parser.add_argument("--max-tokens", type=int, default=128, help="与 E4 一致（校验 E4 config）")
    parser.add_argument("--expected-samples", type=int, default=50,
                        help="正式模式要求的 streaming 样本数（默认 50）")
    parser.add_argument("--warmup-rounds", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="冒烟模式截断（如 3）；正式模式不得使用")
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

    model_name = args.llm_model_name or LLM_MODEL_NAME
    samples, meta = load_e4_samples(args.e4_results, model_name, args.max_tokens,
                                    args.llm_device, args.expected_samples, args.max_samples)

    t0 = datetime.now()
    llm = StreamLLMInference(model_name=model_name, device=args.llm_device, eval_mode=False)
    for i in range(args.warmup_rounds):
        kv = llm.cache_prompt("你好，这是一个测试。", is_end=True)
        for _ in llm.generate(pre_cache=kv, max_new_tokens=10):
            pass
        del kv
    logger.info(f"预热完成（{args.warmup_rounds} 轮）")

    output_csv = PROJECT_ROOT / args.output
    checkpoint_jsonl = output_csv.with_suffix(".checkpoint.jsonl")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_jsonl.write_text("", encoding="utf-8")  # 清空旧断点

    rows = []
    for pass_idx in range(args.repeat):
        for i, s in enumerate(samples):
            row = measure_sample(llm, s, pass_idx, args.max_tokens)
            rows.append(row)
            with open(checkpoint_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not row["error"]:
                logger.info(f"[pass{pass_idx} {i + 1}/{len(samples)}] {s['sample_id']}: "
                            f"decode_to_first_sentence={row['decode_to_first_sentence_ms']:.1f}ms "
                            f"({row['first_sentence_token_idx'] + 1}/{row['n_tokens']} tokens)")

    t1 = datetime.now()
    summary = write_outputs(rows, output_csv)
    n_err = sum(1 for r in rows if r["error"])
    runinfo = write_runinfo(output_csv, sys.argv, meta, args, t0, t1, len(rows), n_err)
    print(f"\n{summary}")
    logger.info(f"结果: {output_csv}；RUNINFO: {runinfo}")
    if n_err:
        logger.error(f"{n_err} 条样本测量失败，详见 CSV error 列与 checkpoint JSONL")
        sys.exit(1)


if __name__ == "__main__":
    main()
