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
- W4（2026-08-21，PRE-PAPER-AUDIT P0-4）：宏平均之外补 corpus 口径——
  逐样本分别保存 WER/CER 的 S/D/I/N（`_edit_counts` DP 回溯，S+D+I 与 _levenshtein
  距离断言一致），汇总 corpus = Σ(S+D+I)/ΣN；宏平均列名不变（mean utterance 口径）。

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


def supplement_index_from_csv(index: dict, ref_csv: Path) -> dict:
    """用 qa_transcribe.corrected.csv 补充缺失样本的参考文本。

    本机无 librispeech/aishell1 的样本 JSON（在 GPU 主机）。评分参考列是
    **reference_full**（完整拼接参考；`reference` 列为 QA 展示用的截断版，
    90/150 行更短，误用会把 clean 集 CER 从 0.1077 抬高到 0.2009）。
    已有键冲突时报错退出；language 由 dataset 列映射。
    """
    lang_map = {"librispeech": "en", "aishell1": "zh"}
    n_new = 0
    with open(ref_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = row["sample_id"]
            ref = row.get("reference_full") or row["reference"]
            lang = lang_map.get(row.get("dataset", ""), "")
            if sid in index and index[sid][0] != ref:
                raise SystemExit(f"参考文本冲突: {sid}（JSON 与 {ref_csv} 不一致，停止）")
            if sid not in index:
                index[sid] = (ref, lang)
                n_new += 1
    logger.info(f"参考 CSV 补充 {n_new} 条（{ref_csv}，reference_full 列）")
    return index


def score_pair(ref: str, hyp: str, language: str):
    """返回 (wer, cer)。exp3 口径 + 英文大小写折叠 + 中文 CER 去空格。

    中文 CER 修正（2026-08-21）：结果 JSON 的 transcribed_text 是 " ".join(fragments)
    的展示重构，片段接缝空格会被 cer 计为删除错误（实测抬高 ~3.5pt/50 样本抽查）。
    中文文本无空格语义，ref/hyp 均去空格后计算（zh WER 经 zh_to_word_seq 本就去空格，不受影响）。
    """
    from experiments.scripts.run_exp_quality import cer, normalize_text, wer, zh_to_word_seq
    if language.lower().startswith("zh"):
        w = wer(zh_to_word_seq(ref), zh_to_word_seq(hyp))
        c = cer(normalize_text(ref).replace(" ", ""),
                normalize_text(hyp).replace(" ", ""), normalize=False)
    else:
        w = wer(normalize_text(ref).lower(), normalize_text(hyp).lower(), normalize=False)
        c = cer(normalize_text(ref).lower(), normalize_text(hyp).lower(), normalize=False)
    return w, c


def _unit_seqs(ref: str, hyp: str, language: str):
    """复刻 score_pair 的归一化单位序列，用于 S/D/I 计数（W4 corpus 口径）。

    与 score_pair 逐字面对应：
    - zh WER：wer(zh_to_word_seq(...)) 默认 normalize=True，但 normalize_text 幂等，
      故单位 = zh_to_word_seq(x).split()（逐字）；
    - zh CER：list(normalize_text(x).replace(" ", ""))（去空格逐字）；
    - en WER：normalize_text(x).lower().split()（词）；en CER：list(...)（含内部空格字符）。
    """
    from experiments.scripts.run_exp_quality import normalize_text, zh_to_word_seq
    if language.lower().startswith("zh"):
        return (zh_to_word_seq(ref).split(), zh_to_word_seq(hyp).split(),
                list(normalize_text(ref).replace(" ", "").strip()),
                list(normalize_text(hyp).replace(" ", "").strip()))
    return (normalize_text(ref).lower().strip().split(),
            normalize_text(hyp).lower().strip().split(),
            list(normalize_text(ref).lower().strip()),
            list(normalize_text(hyp).lower().strip()))


def _edit_counts(ref_units: list, hyp_units: list):
    """DP + 确定性回溯的 (S, D, I) 计数；S+D+I 恰等于 run_exp_quality._levenshtein 距离。

    回溯优先级：对角（match/substitute）> 删除 > 插入（固定顺序，保证可复算）。
    """
    m, n = len(ref_units), len(hyp_units)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if ref_units[i - 1] == hyp_units[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    s = d = ins = 0
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_units[i - 1] == hyp_units[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    s += 1
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            d += 1
            i -= 1
            continue
        ins += 1
        j -= 1
    return s, d, ins


def score_pair_counts(ref: str, hyp: str, language: str) -> dict:
    """返回 w/c + WER/CER 各自的 S/D/I/N 计数（与 score_pair 同单位序列）。"""
    from experiments.scripts.run_exp_quality import _levenshtein
    w, c = score_pair(ref, hyp, language)
    wr, wh, cr, ch = _unit_seqs(ref, hyp, language)
    ws, wd, wi = _edit_counts(wr, wh)
    cs, cd, ci = _edit_counts(cr, ch)
    assert ws + wd + wi == _levenshtein(wr, wh), "WER 计数与距离不一致"
    assert cs + cd + ci == _levenshtein(cr, ch), "CER 计数与距离不一致"
    return {"wer": w, "cer": c,
            "wer_s": ws, "wer_d": wd, "wer_i": wi, "wer_n": len(wr),
            "cer_s": cs, "cer_d": cd, "cer_i": ci, "cer_n": len(cr)}


def scope_of(sample_id: str) -> str:
    return sample_id.split("_")[0]


def collect_rows(results_globs: list) -> list:
    """读取全部结果 JSON，返回展开的行列表。

    按**目录**分组取最新一份（2026-08-21 修正：带引号的单 glob 会在 Python 内展开成
    多目录多文件，旧 files[-1] 只读到最后一目录；分组取最新后带不带引号行为一致）。
    """
    by_dir = {}
    for pattern in results_globs:
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"结果 glob 无匹配: {pattern}")
        for f in files:
            by_dir.setdefault(str(Path(f).parent), []).append(f)
    rows = []
    for dir_key in sorted(by_dir):
        path = Path(sorted(by_dir[dir_key])[-1])  # 每个目录取最新一份
        data = json.loads(path.read_text(encoding="utf-8"))
        n = 0
        for r in data.get("results", []):
            if r.get("error"):
                continue
            r["_dir"] = Path(path).parent.name  # 结果目录名（R2 变体口径的 scope 来源）
            rows.append(r)
            n += 1
        logger.info(f"读取 {path}（有效 {n} 行）")
    if not rows:
        raise SystemExit("无有效结果行")
    return rows


def compute(rows: list, sample_index: dict, scope_by: str = "dir"):
    """返回 (wer_rows, ttft_rows, persample_rows)。缺参考文本的样本报错退出（不静默跳过）。

    scope_by="dir"：按结果目录名分组（R2 变体口径：librispeech_clean / *_snr20 / ...）；
    scope_by="prefix"：按 sample_id 前缀分组（E3 口径：crosswoz / multiwoz）。
    """
    scored = []
    for r in rows:
        if r.get("error"):
            continue  # 双重保护：collect_rows 已过滤，compute 直接调用时也排除
        sid = r["sample_id"]
        if sid not in sample_index:
            raise SystemExit(f"样本 {sid} 未在样本 JSON 索引中找到（缺参考文本，停止）")
        ref, lang = sample_index[sid]
        hyp = (r.get("transcribed_text") or "").strip()
        sc = score_pair_counts(ref, hyp, lang or "zh")
        scope = r.get("_dir") if scope_by == "dir" else scope_of(sid)
        if not scope:
            scope = scope_of(sid)
        scored.append({"sample_id": sid, "scope": scope, "mode": r["mode"],
                       "duration_group": r.get("duration_group", ""),
                       "ttft": r.get("ttft", 0.0), **sc,
                       "empty": int(len(hyp) == 0)})

    def _corpus(mr, prefix):
        n_sum = sum(s[f"{prefix}_n"] for s in mr)
        if n_sum == 0:
            return ""
        err = sum(s[f"{prefix}_s"] + s[f"{prefix}_d"] + s[f"{prefix}_i"] for s in mr)
        return f"{err / n_sum:.4f}"

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
                             "cer_mean": f"{c.mean():.4f}", "cer_std": f"{c.std():.4f}",
                             # corpus 口径（W4）：Σ(S+D+I)/ΣN，与宏平均并存
                             "wer_corpus": _corpus(mr, "wer"),
                             "cer_corpus": _corpus(mr, "cer"),
                             "wer_S": sum(s["wer_s"] for s in mr),
                             "wer_D": sum(s["wer_d"] for s in mr),
                             "wer_I": sum(s["wer_i"] for s in mr),
                             "wer_N": sum(s["wer_n"] for s in mr),
                             "cer_S": sum(s["cer_s"] for s in mr),
                             "cer_D": sum(s["cer_d"] for s in mr),
                             "cer_I": sum(s["cer_i"] for s in mr),
                             "cer_N": sum(s["cer_n"] for s in mr)})

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
    persample_rows = [{"sample_id": s["sample_id"], "scope": s["scope"], "mode": s["mode"],
                       "duration_group": s["duration_group"],
                       "wer": f"{s['wer']:.4f}", "cer": f"{s['cer']:.4f}",
                       "wer_s": s["wer_s"], "wer_d": s["wer_d"], "wer_i": s["wer_i"],
                       "wer_n": s["wer_n"],
                       "cer_s": s["cer_s"], "cer_d": s["cer_d"], "cer_i": s["cer_i"],
                       "cer_n": s["cer_n"], "empty": s["empty"]} for s in scored]
    return wer_rows, ttft_rows, persample_rows


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
        # 中文 CER 去空格（join 接缝空格不得计入）：带接缝空格的相同文本 → CER=0
        w2, c2 = score_pair("你好世界今天", "你好 世界 今天", "zh")
        check("中文 CER 去接缝空格", c2 == 0.0, f"cer={c2}")
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
        wer_rows, ttft_rows, persample_rows = compute(rows, idx, scope_by="prefix")
        cw_stream = [r for r in wer_rows if r["scope"] == "cw" and r["mode"] == "streaming"][0]
        check("error 行排除", cw_stream["n"] == 2, f"n={cw_stream['n']}")
        check("空转写计数与 WER", cw_stream["n_empty"] == 1
              and abs(float(cw_stream["wer_mean"]) - 0.625) < 1e-9,
              f"wer_mean={cw_stream['wer_mean']}（(0.25 + 1.0)/2 = 0.625）")
        # W4 corpus 口径：cw streaming = cw_s1（你好世届 vs 你好世界：S=1,N=4）
        # + cw_s2 空转写（D=6,N=6）→ corpus wer = (1+6)/(4+6) = 0.7
        check("corpus WER 恒等式", abs(float(cw_stream["wer_corpus"]) - 0.7) < 1e-9,
              f"wer_corpus={cw_stream['wer_corpus']}（(1+6)/(4+6)=0.7）")
        check("corpus 求和列", cw_stream["wer_S"] == 1 and cw_stream["wer_D"] == 6
              and cw_stream["wer_N"] == 10 and cw_stream["cer_N"] == 10,
              str({k: cw_stream[k] for k in ("wer_S", "wer_D", "wer_N", "cer_N")}))
        # 逐样本计数字段（W4：WER/CER 分别保存 S/D/I/N）
        ps1 = [p for p in persample_rows if p["sample_id"] == "cw_s1"][0]
        check("逐样本 WER 计数", ps1["wer_s"] == 1 and ps1["wer_d"] == 0
              and ps1["wer_i"] == 0 and ps1["wer_n"] == 4, str(ps1))
        check("逐样本 CER 计数", ps1["cer_s"] == 1 and ps1["cer_n"] == 4, str(ps1))
        ps2 = [p for p in persample_rows if p["sample_id"] == "cw_s2"][0]
        check("空转写按全删除计数", ps2["wer_d"] == 6 and ps2["cer_d"] == 6, str(ps2))
        # 英文计数：the cat sat vs the dog at → S=2,N=3
        enc = score_pair_counts("the cat sat", "the dog at", "en")
        check("英文 WER 计数", enc["wer_s"] == 2 and enc["wer_n"] == 3, str(enc))
        check("英文计数与宏平均一致", abs(enc["wer"] - 2 / 3) < 1e-9, str(enc["wer"]))
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
        # collect_rows：单 glob 跨多目录 → 按目录分组取最新（2026-08-21 修正）
        for sub in ("ds_a", "ds_b"):
            (tp / "res" / sub).mkdir(parents=True)
        for ts, marker in (("20260101_000000", 111.0), ("20260102_000000", 222.0)):
            (tp / "res" / "ds_a" / f"exp1_results_{ts}.json").write_text(json.dumps({"results": [
                {"sample_id": "cw_s1", "mode": "streaming", "error": "",
                 "transcribed_text": "你好世界", "duration_group": "long", "ttft": marker}]}),
                encoding="utf-8")
        (tp / "res" / "ds_b" / "exp1_results_20260101_000000.json").write_text(json.dumps({"results": [
            {"sample_id": "cw_s1", "mode": "streaming", "error": "",
             "transcribed_text": "你好世界", "duration_group": "long", "ttft": 333.0}]}),
            encoding="utf-8")
        got = collect_rows([str(tp / "res" / "*" / "exp1_results_*.json")])
        check("单 glob 多目录收全", len(got) == 2 and {r["_dir"] for r in got} == {"ds_a", "ds_b"})
        check("每目录取最新", sorted(r["ttft"] for r in got) == [222.0, 333.0],
              str([r["ttft"] for r in got]))
        # dir 口径：_dir 标签决定分组（R2 变体逐条件拆分）
        rows_dir = [dict(r, _dir="libri_clean") for r in rows]
        wer_rows_d, _, _ = compute(rows_dir, idx, scope_by="dir")
        check("dir 口径分组", all(r["scope"] in ("libri_clean", "ALL") for r in wer_rows_d))
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
    parser.add_argument("--ref-csv", type=str, default=None,
                        help="可选：qa_transcribe.corrected.csv 路径，补充 JSON 索引缺失样本的参考文本")
    parser.add_argument("--out-dir", type=str, required=False)
    parser.add_argument("--tag", type=str, default="real",
                        help="输出文件名 tag：wer_<tag>.csv / ttft_<tag>.csv")
    parser.add_argument("--scope-by", type=str, default="dir", choices=["dir", "prefix"],
                        help="dir=按结果目录名分组（R2 变体）；prefix=按 sample_id 前缀（E3）")
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
    if args.ref_csv:
        sample_index = supplement_index_from_csv(sample_index, PROJECT_ROOT / args.ref_csv)
    rows = collect_rows(args.results)
    wer_rows, ttft_rows, persample_rows = compute(rows, sample_index, args.scope_by)

    out_dir = PROJECT_ROOT / args.out_dir
    write_csv(wer_rows, ["scope", "mode", "n", "n_empty", "wer_mean", "wer_std",
                         "cer_mean", "cer_std", "wer_corpus", "cer_corpus",
                         "wer_S", "wer_D", "wer_I", "wer_N",
                         "cer_S", "cer_D", "cer_I", "cer_N"], out_dir / f"wer_{args.tag}.csv")
    write_csv(persample_rows, ["sample_id", "scope", "mode", "duration_group", "wer", "cer",
                               "wer_s", "wer_d", "wer_i", "wer_n",
                               "cer_s", "cer_d", "cer_i", "cer_n", "empty"],
              out_dir / f"wer_{args.tag}_persample.csv")
    write_csv(ttft_rows, ["scope", "mode", "duration_group", "n", "ttft_mean", "ttft_std",
                          "ttft_p50", "ttft_p95"], out_dir / f"ttft_{args.tag}.csv")


if __name__ == "__main__":
    main()
