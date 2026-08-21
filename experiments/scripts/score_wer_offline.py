#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线 WER/CER + TTFT 统计（R2 §3.4 / R3 §4.3 交付物：wer_real.csv / ttft_real.csv / la_vs_b 用数）。

从实验结果 JSON（含逐样本 transcribed_text / ttft / duration_group / mode / error）与
样本 JSON（processed/json/**/<sample_id>.json，含 text/language）离线计算质量与延迟统计。

口径（与 exp3 一致，两处扩展均有记录）：
- 中文：WER = wer(zh_to_word_seq(ref), zh_to_word_seq(hyp))；CER = cer(ref, hyp)
  （run_exp_quality.py:603-606 原生口径；cer 直接吃原文，不得先 zh_to_word_seq——
  空格会污染分母，2026-08-21 修正 qa_real_speech 同类问题）；
- 英文：normalize_text 去标点后**补大小写折叠**（LibriSpeech 参考全大写、Whisper 输出混合
  大小写，LibriSpeech 官方评估惯例；对 multiwoz 一并折叠保持三系统同口径，
  与 run_exp_baseline_la 内联 wer 的无折叠口径差异在 changelog 登记）；
- 空转写（babble 等场景的零提交样本）：WER/CER 记 1.0 并计入 n_empty；
- error 行不参与统计。

用法：
  uv run python -m experiments.scripts.score_wer_offline \
      --results "experiments/results/revision/r2_real_speech/*/exp1_results_*.json" \
      --out-dir experiments/results/revision/r2_real_speech
  # E3（A/B 重跑 + LA 同口径）：
  uv run python -m experiments.scripts.score_wer_offline \
      --results "experiments/results/revision/r3_baseline_la/system_ab_rerun/exp1_results_*.json" \
                 "experiments/results/revision/r3_baseline_la/la_results_*.json" \
      --out-dir experiments/results/revision/r3_baseline_la --tag la_vs_b
  # 自检：
  uv run python -m experiments.scripts.score_wer_offline --self-test
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

MODE_ORDER = {"non-streaming": 0, "la_streaming": 1, "streaming": 2}
GROUP_ORDER = {"long": 0, "very_long": 1, "extra_long": 2}


def build_sample_index(json_root: Path) -> dict:
    """sample_id -> (text, language)"""
    index = {}
    for js in json_root.rglob("*.json"):
        try:
            meta = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = meta.get("sample_id") or js.stem
        if "text" in meta:
            index[sid] = (meta["text"], meta.get("language", ""))
    return index


def score_pair(ref: str, hyp: str, language: str):
    """返回 (wer, cer)。exp3 口径 + 英文大小写折叠。"""
    from experiments.scripts.run_exp_quality import cer, normalize_text, wer, zh_to_word_seq
    if language.lower().startswith("zh"):
        w = wer(zh_to_word_seq(ref), zh_to_word_seq(hyp))
        c = cer(ref, hyp)
    else:
        w = wer(normalize_text(ref).lower(), normalize_text(hyp).lower(), normalize=False)
        c = cer(normalize_text(ref).lower(), normalize_text(hyp).lower(), normalize=False)
    return w, c


def scope_of(sample_id: str) -> str:
    return sample_id.split("_")[0]


def collect_rows(results_globs: list) -> list:
    """读取全部结果 JSON，返回展开的行列表。"""
    rows = []
    for pattern in results_globs:
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"结果 glob 无匹配: {pattern}")
        path = files[-1]  # 每个目录取最新一份
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        n = 0
        for r in data.get("results", []):
            if r.get("error"):
                continue
            rows.append(r)
            n += 1
        logger.info(f"读取 {path}（有效 {n} 行）")
    if not rows:
        raise SystemExit("无有效结果行")
    return rows


def compute(rows: list, sample_index: dict):
    """返回 (wer_rows, ttft_rows)。缺参考文本的样本报错退出（不静默跳过）。"""
    scored = []
    for r in rows:
        if r.get("error"):
            continue  # 双重保护：collect_rows 已过滤，compute 直接调用时也排除
        sid = r["sample_id"]
        if sid not in sample_index:
            raise SystemExit(f"样本 {sid} 未在样本 JSON 索引中找到（缺参考文本，停止）")
        ref, lang = sample_index[sid]
        hyp = (r.get("transcribed_text") or "").strip()
        w, c = score_pair(ref, hyp, lang or "zh")
        scored.append({"scope": scope_of(sid), "mode": r["mode"],
                       "duration_group": r.get("duration_group", ""),
                       "ttft": r.get("ttft", 0.0), "wer": w, "cer": c,
                       "empty": int(len(hyp) == 0)})

    wer_rows = []
    scopes = sorted({s["scope"] for s in scored})
    for scope in scopes + ["ALL"]:
        sub = [s for s in scored if scope == "ALL" or s["scope"] == scope]
        for mode in sorted({s["mode"] for s in sub}, key=lambda m: MODE_ORDER.get(m, 9)):
            mr = [s for s in sub if s["mode"] == mode]
            w = np.array([s["wer"] for s in mr])
            c = np.array([s["cer"] for s in mr])
            wer_rows.append({"scope": scope, "mode": mode, "n": len(mr),
                             "n_empty": sum(s["empty"] for s in mr),
                             "wer_mean": f"{w.mean():.4f}", "wer_std": f"{w.std():.4f}",
                             "cer_mean": f"{c.mean():.4f}", "cer_std": f"{c.std():.4f}"})

    ttft_rows = []
    for scope in scopes + ["ALL"]:
        sub = [s for s in scored if scope == "ALL" or s["scope"] == scope]
        for mode in sorted({s["mode"] for s in sub}, key=lambda m: MODE_ORDER.get(m, 9)):
            mr = [s for s in sub if s["mode"] == mode]
            groups = sorted({s["duration_group"] for s in mr},
                            key=lambda g: GROUP_ORDER.get(g, 9))
            for g in groups + (["overall"] if len(groups) > 1 else []):
                gr = [s for s in mr if g == "overall" or s["duration_group"] == g]
                v = np.array([s["ttft"] for s in gr])
                ttft_rows.append({"scope": scope, "mode": mode, "duration_group": g,
                                  "n": len(gr), "ttft_mean": f"{v.mean():.1f}",
                                  "ttft_std": f"{v.std():.1f}",
                                  "ttft_p50": f"{np.percentile(v, 50):.1f}",
                                  "ttft_p95": f"{np.percentile(v, 95):.1f}"})
    return wer_rows, ttft_rows


def write_csv(rows: list, fields: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"已保存: {path}（{len(rows)} 行）")


def self_test() -> int:
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(f"[{'PASS' if cond else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
        if not cond:
            failures += 1

    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        # 造样本 JSON：英文全大写参考 + 中文参考
        (tp / "json" / "libri").mkdir(parents=True)
        (tp / "json" / "cw").mkdir(parents=True)
        (tp / "json" / "libri" / "libri_s1.json").write_text(json.dumps(
            {"sample_id": "libri_s1", "text": "HELLO WORLD", "language": "en"}), encoding="utf-8")
        (tp / "json" / "cw" / "cw_s1.json").write_text(json.dumps(
            {"sample_id": "cw_s1", "text": "你好世界", "language": "zh"}), encoding="utf-8")
        (tp / "json" / "cw" / "cw_s2.json").write_text(json.dumps(
            {"sample_id": "cw_s2", "text": "今天天气不错", "language": "zh"}), encoding="utf-8")

        idx = build_sample_index(tp / "json")
        check("样本索引", idx.get("libri_s1") == ("HELLO WORLD", "en"))

        # 英文大小写折叠：混合大小写假设 vs 全大写参考 → WER 0
        w, c = score_pair("HELLO WORLD", "Hello world.", "en")
        check("英文折叠", w == 0.0 and c == 0.0, f"wer={w} cer={c}")
        # 中文口径：替换 1 字 → CER=1/4
        w, c = score_pair("你好世界", "你好世届", "zh")
        check("中文 CER 口径", abs(c - 0.25) < 1e-9, f"cer={c}（空格污染口径会给 ~1/7）")
        # 空假设 → 1.0
        w, c = score_pair("你好世界", "", "zh")
        check("空假设", w == 1.0 and c == 1.0)

        # 结果行 → 统计
        rows = [
            {"sample_id": "libri_s1", "mode": "streaming", "duration_group": "long",
             "ttft": 1000.0, "transcribed_text": "Hello world", "error": ""},
            {"sample_id": "libri_s1", "mode": "non-streaming", "duration_group": "long",
             "ttft": 3000.0, "transcribed_text": "HELLO WORLD", "error": ""},
            {"sample_id": "cw_s1", "mode": "streaming", "duration_group": "long",
             "ttft": 1100.0, "transcribed_text": "你好世届", "error": ""},
            {"sample_id": "cw_s2", "mode": "streaming", "duration_group": "very_long",
             "ttft": 1200.0, "transcribed_text": "", "error": ""},   # 空转写
            {"sample_id": "cw_s2", "mode": "streaming", "duration_group": "very_long",
             "ttft": 9999.0, "transcribed_text": "x", "error": "boom"},  # error 行排除
        ]
        wer_rows, ttft_rows = compute(rows, idx)
        cw_stream = [r for r in wer_rows if r["scope"] == "cw" and r["mode"] == "streaming"][0]
        check("error 行排除", cw_stream["n"] == 2, f"n={cw_stream['n']}")
        check("空转写计数与 WER", cw_stream["n_empty"] == 1
              and abs(float(cw_stream["wer_mean"]) - 0.625) < 1e-9,
              f"wer_mean={cw_stream['wer_mean']}（(0.25 + 1.0)/2 = 0.625）")
        overall = [r for r in ttft_rows if r["scope"] == "ALL" and r["mode"] == "streaming"
                   and r["duration_group"] == "overall"]
        check("ttft overall 行", len(overall) == 1 and overall[0]["n"] == 3)
        # 缺参考文本必须退出
        try:
            compute([{"sample_id": "ghost_s9", "mode": "streaming", "duration_group": "long",
                      "ttft": 1.0, "transcribed_text": "x", "error": ""}], idx)
            check("缺参考退出", False, "应 SystemExit")
        except SystemExit:
            check("缺参考退出", True)
        # CSV 写出
        out = tp / "out"
        write_csv(wer_rows, list(wer_rows[0].keys()), out / "wer_real.csv")
        write_csv(ttft_rows, list(ttft_rows[0].keys()), out / "ttft_real.csv")
        check("CSV 写出", (out / "wer_real.csv").exists() and (out / "ttft_real.csv").exists())

    print(f"\nself-test {'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description="离线 WER/CER + TTFT 统计（R2/R3 交付物）")
    parser.add_argument("--results", nargs="+", default=None,
                        help="结果 JSON glob（可多个；每个目录取最新一份）")
    parser.add_argument("--json-root", type=str, default="experiments/datasets/processed/json")
    parser.add_argument("--out-dir", type=str, required=False)
    parser.add_argument("--tag", type=str, default="real",
                        help="输出文件名 tag：wer_<tag>.csv / ttft_<tag>.csv")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    set_global_log_level(args.log_level)

    if args.self_test:
        sys.exit(self_test())
    if not args.results or not args.out_dir:
        parser.error("正式模式需要 --results 与 --out-dir")

    sample_index = build_sample_index(PROJECT_ROOT / args.json_root)
    logger.info(f"样本索引: {len(sample_index)} 条")
    rows = collect_rows(args.results)
    wer_rows, ttft_rows = compute(rows, sample_index)

    out_dir = PROJECT_ROOT / args.out_dir
    write_csv(wer_rows, ["scope", "mode", "n", "n_empty", "wer_mean", "wer_std",
                         "cer_mean", "cer_std"], out_dir / f"wer_{args.tag}.csv")
    write_csv(ttft_rows, ["scope", "mode", "duration_group", "n", "ttft_mean", "ttft_std",
                          "ttft_p50", "ttft_p95"], out_dir / f"ttft_{args.tag}.csv")


if __name__ == "__main__":
    main()
