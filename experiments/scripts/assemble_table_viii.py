# -*- coding: utf-8 -*-
"""W8 阶段 2：新 Table VIII 装配（R7 统一 TTFA，唯一合法数据源）。

固定决策（2026-08-22，按审查 §2.4；标签修正按复审 review-reply §2）：
  1. 主指标 first_playable_pcm（playable 阈值 1324B = 22050Hz×16bit×30ms）；
  2. 统计范围 repeat0、n=50/模式（zh 25 + en 25）；补轮 40 条仅用于 CV，不进本表；
  3. 统计量 mean / std(ddof=1) / P50 / P90 / P95（np.percentile 线性插值，与 R1 口径一致）；
  4. 单位 ms，1 位小数；
  5. ttfa_received(first_pcm_byte) 为 QA 补充指标：|received−playable| 记入装配 QA，不进主表；
  6. tts_control 7076ms 仅作 TTS 服务延迟归因/审稿回复证据，不进 Table VIII；引用必带偏差豁免脚注；
  7. 完全排除旧 r6_ttfa/ttfa_budget.csv 估计项与跨运行分项；
  8. 分项标签边界（2026-08-22 复审修正）：第二分项实际区间 =
     pipeline_input_close − feed_end，论文标签 **t_feed_to_close_wait（喂入结束→管线输入关闭）**；
     源 summary 字段名 `t_flush_to_close` 为历史命名（易误读为 flush 执行耗时——explicit flush
     本身仅 ~0.21ms，133ms 绝大部分是 feed_end→explicit_flush_start 的等待）。论文不得将
     该分项归因为 flush 计算开销。

输入（只读）：results/revision/r7_ttfa_unified/r7_main/checkpoint_r7_main.jsonl
             （不可变归档，blob 046f1d6d@946b720）
             results/revision/r7_ttfa_unified/r7_main/ttfa_summary_r7_main.csv（运行侧 summary，
             双入口对拍用；其字段名保留历史命名，输出层映射为论文标签）
输出：results/revision/r7_ttfa_unified/table_viii/table_viii_r7.csv、TABLE_VIII_ASSEMBLED.md

用法: uv run python experiments/scripts/assemble_table_viii.py
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R7 = ROOT / "experiments/results/revision/r7_ttfa_unified"
CKPT = R7 / "r7_main" / "checkpoint_r7_main.jsonl"
SUMMARY = R7 / "r7_main" / "ttfa_summary_r7_main.csv"
OUT_DIR = R7 / "table_viii"
OUT_CSV = OUT_DIR / "table_viii_r7.csv"
OUT_MD = OUT_DIR / "TABLE_VIII_ASSEMBLED.md"

# 源字段（历史命名，与 checkpoint/summary 一致）→ 论文展示标签（边界准确）
PAPER_LABEL = {"t_flush_to_close": "t_feed_to_close_wait"}
SOURCE_COMPONENTS = [
    "t_trailing_feed_wait", "t_flush_to_close", "t_close_to_first_token",
    "t_first_token_to_text_ready", "t_text_ready_to_tts_req", "t_tts_to_playable",
]
COMPONENT_LABELS = {
    "t_trailing_feed_wait": "语音结束→喂入结束",
    "t_feed_to_close_wait": "喂入结束→管线输入关闭",
    "t_close_to_first_token": "输入关闭→首 token",
    "t_first_token_to_text_ready": "首 token→文本就绪(首句/全文)",
    "t_text_ready_to_tts_req": "文本就绪→TTS 请求",
    "t_tts_to_playable": "TTS 请求→首个可播 PCM",
}


def stats(v: np.ndarray) -> dict:
    return {"n": len(v), "mean": v.mean(), "std": v.std(ddof=1),
            "p50": np.percentile(v, 50), "p90": np.percentile(v, 90),
            "p95": np.percentile(v, 95)}


def components_of(r: dict) -> dict:
    ev = r["events"]
    tr = ev["first_sentence_boundary_ns"] or ev["generation_end_ns"]
    return {"t_trailing_feed_wait": ev["feed_end_ns"] - ev["physical_speech_end_ns"],
            "t_flush_to_close": ev["pipeline_input_close_ns"] - ev["feed_end_ns"],
            "t_close_to_first_token": ev["first_model_token_ns"] - ev["pipeline_input_close_ns"],
            "t_first_token_to_text_ready": tr - ev["first_model_token_ns"],
            "t_text_ready_to_tts_req": ev["tts_request_start_ns"] - tr,
            "t_tts_to_playable": ev["first_playable_pcm_ns"] - ev["tts_request_start_ns"]}


def main() -> int:
    lines = CKPT.read_text(encoding="utf-8").splitlines()
    recs = [json.loads(x) for x in lines[1:] if x.strip()]
    ok = [r for r in recs if r["terminal_state"] == "success" and r["repeat_idx"] == 0]
    assert len(ok) == 100, f"repeat0 success 应为 100（50×2），实际 {len(ok)}"

    qa = []
    # QA-1 逐记录六分项闭合（首尾相接恒等式）
    worst = 0.0
    for r in ok:
        ev = r["events"]
        s = sum(components_of(r).values()) / 1e6
        ttfa = (ev["first_playable_pcm_ns"] - ev["physical_speech_end_ns"]) / 1e6
        worst = max(worst, abs(s - ttfa))
    qa.append(f"QA-1 六分项逐记录闭合：100 条最大残差 {worst:.2e} ms（恒等式成立）")
    assert worst < 1e-6

    # 主统计
    table = {}
    for mode in ("streaming", "non-streaming"):
        for lang in ("zh", "en", "ALL"):
            sub = [r for r in ok if r["mode"] == mode
                   and (lang == "ALL" or r["language"] == lang)]
            assert len(sub) == (50 if lang == "ALL" else 25)
            ev_play = np.array([(r["events"]["first_playable_pcm_ns"]
                                 - r["events"]["physical_speech_end_ns"]) / 1e6 for r in sub])
            ev_recv = np.array([(r["events"]["first_pcm_byte_ns"]
                                 - r["events"]["physical_speech_end_ns"]) / 1e6 for r in sub])
            entry = {"ttfa_playable_ms": stats(ev_play),
                     "ttfa_received_ms": stats(ev_recv)}
            for c in SOURCE_COMPONENTS:
                entry[c] = stats(np.array([components_of(r)[c] / 1e6 for r in sub]))
            table[(mode, lang)] = entry

    # QA-2 received 与 playable 差（QA 补充，不进主表）
    deltas = np.array([(r["events"]["first_playable_pcm_ns"]
                        - r["events"]["first_pcm_byte_ns"]) / 1e6 for r in ok])
    qa.append(f"QA-2 received→playable 缓冲差：mean {deltas.mean():.1f} / p95 "
              f"{np.percentile(deltas, 95):.1f} / max {deltas.max():.1f} ms（received 仅作 QA 补充）")

    # QA-3 与运行侧 summary CSV 双入口对拍（summary 用源字段名，输出层映射为论文标签）
    with open(SUMMARY, encoding="utf-8") as f:
        srows = list(csv.DictReader(f))
    mism = []
    for row in srows:
        if row["metric"] not in SOURCE_COMPONENTS + ["ttfa_playable_ms", "ttfa_received_ms"]:
            continue
        e = table[(row["mode"], row["language"])][row["metric"]]
        for k in ("mean", "std", "p50", "p90", "p95"):
            if row[k] and abs(float(row[k]) - e[k]) > 0.051:
                mism.append((row["mode"], row["language"], row["metric"], k, row[k], f"{e[k]:.1f}"))
    qa.append(f"QA-3 与 ttfa_summary_r7_main.csv 双入口对拍：{len(srows)} 行全部一致"
              if not mism else f"QA-3 对拍失败 {len(mism)} 项: {mism[:5]}")
    assert not mism, mism

    # QA-4 输入 checkpoint 完整性
    ck_sha = hashlib.sha256(CKPT.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    qa.append(f"QA-4 输入 checkpoint sha256(LF)={ck_sha[:16]}…（不可变归档，control_from 同源）")
    assert ck_sha.startswith("4edcd6ec28189d00"), ck_sha

    # 扁平 CSV（metric 列使用论文标签）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["mode", "language", "metric", "n", "mean_ms", "std_ms",
                    "p50_ms", "p90_ms", "p95_ms"])
        for (mode, lang), entry in table.items():
            for metric, st in entry.items():
                out_metric = PAPER_LABEL.get(metric, metric)
                w.writerow([mode, lang, out_metric, st["n"]] +
                           [f"{st[k]:.1f}" for k in ("mean", "std", "p50", "p90", "p95")])

    # 论文表 MD
    red = {}
    for lang in ("zh", "en", "ALL"):
        b, a = table[("streaming", lang)], table[("non-streaming", lang)]
        red[lang] = {"mean": 100 * (1 - b["ttfa_playable_ms"]["mean"] / a["ttfa_playable_ms"]["mean"]),
                     "p50": 100 * (1 - b["ttfa_playable_ms"]["p50"] / a["ttfa_playable_ms"]["p50"]),
                     "ratio_mean": a["ttfa_playable_ms"]["mean"] / b["ttfa_playable_ms"]["mean"],
                     "ratio_p50": a["ttfa_playable_ms"]["p50"] / b["ttfa_playable_ms"]["p50"]}

    md = []
    md.append("# Table VIII 装配稿（R7 统一 TTFA，2026-08-22；分项标签修正版）\n")
    md.append("> **数据源（唯一合法）**：`r7_ttfa_unified/r7_main/`（repeat0，n=50/模式，zh/en 各 25）。")
    md.append("> 主指标 **first_playable_pcm**（speech_end → 首个 ≥1324B 可播 PCM）；单位 ms，1 位小数；")
    md.append("> std=ddof=1，分位数 np.percentile 线性插值。六分项首尾相接恒等闭合（QA-1）。")
    md.append("> 旧 `r6_ttfa/ttfa_budget.csv` 全部行作废，未参与本表。\n")
    md.append("> **分项标签边界（2026-08-22 复审修正）**：第二分项论文标签为 "
              "`t_feed_to_close_wait`（**喂入结束→管线输入关闭**，= pipeline_input_close − feed_end）；")
    md.append("> 源 summary 字段 `t_flush_to_close` 为历史命名。该 ~133ms 是完整等待，"
              "**不得归因为 flush 计算开销**——按复审独立复算（review-reply §2）：")
    md.append("> feed_end→explicit_flush_start ≈132.68ms、explicit flush 本身 ≈0.21ms、"
              "flush_done→input_close ≈0.12ms（合计 0.33ms 量级为 flush 段自身）。")
    md.append("> 本表保持六段闭合链、仅用准确标签（复审推荐的最简处理）。\n")

    md.append("## (a) TTFA 总量\n")
    md.append("| 系统 | 语种 | n | mean | std | P50 | P90 | P95 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for mode, name in (("streaming", "B（流式）"), ("non-streaming", "A（非流式）")):
        for lang in ("zh", "en", "ALL"):
            st = table[(mode, lang)]["ttfa_playable_ms"]
            md.append(f"| {name} | {lang} | {st['n']} | " +
                      " | ".join(f"{st[k]:.1f}" for k in ("mean", "std", "p50", "p90", "p95")) + " |")
    md.append("")
    for lang in ("ALL", "zh", "en"):
        r = red[lang]
        md.append(f"- B vs A（{lang}）：mean 降 **{r['mean']:.1f}%**（{r['ratio_mean']:.2f}×）、"
                  f"P50 降 **{r['p50']:.1f}%**（{r['ratio_p50']:.2f}×）——两种表述二选一，勿混用。")
    md.append("")

    md.append("## (b) 组件分解（mean±std，ms）\n")
    md.append("| 组件 | B zh | B en | B ALL | A zh | A en | A ALL |")
    md.append("|---|---|---|---|---|---|---|")
    for c in SOURCE_COMPONENTS:
        label = PAPER_LABEL.get(c, c)
        cells = []
        for mode in ("streaming", "non-streaming"):
            for lang in ("zh", "en", "ALL"):
                st = table[(mode, lang)][c]
                cells.append(f"{st['mean']:.1f}±{st['std']:.1f}")
        md.append(f"| {label}（{COMPONENT_LABELS[label]}） | " + " | ".join(cells) + " |")
    row_sum, row_ttfa = [], []
    for mode in ("streaming", "non-streaming"):
        for lang in ("zh", "en", "ALL"):
            cs = sum(table[(mode, lang)][c]["mean"] for c in SOURCE_COMPONENTS)
            t = table[(mode, lang)]["ttfa_playable_ms"]["mean"]
            row_sum.append(f"{cs:.1f}")
            row_ttfa.append(f"{t:.1f}")
    md.append("| **Σ组件（闭合校验）** | " + " | ".join(row_sum) + " |")
    md.append("| **TTFA（表 a 复核）** | " + " | ".join(row_ttfa) + " |")
    md.append("")

    md.append("## (c) 稳定性与截断注记\n")
    md.append("- 子集三轮 CV（10 样本×2 模式，ddof=1）：mean 7.73% / max 20.70%（`ttfa_subset_cv_r7_main.csv`）。")
    gstop = {}
    for r in ok:
        gstop[r["generation_stop_reason"]] = gstop.get(r["generation_stop_reason"], 0) + 1
    md.append(f"- 生成截断（repeat0，n=100）：{gstop.get('max_tokens', 0)} 条 max_tokens 截断 / "
              f"{gstop.get('eos', 0)} 条 eos（max_tokens=128 对 A/B 同等作用）。")
    md.append("- speaker '晓伊' 由本地服务映射为内置中文女声（非原论文音色）；Triton fallback×4 为平台固定条件——"
              "两者均入 RUNINFO 注记，论文按 review §6 声明。\n")

    md.append("## (d) TTS 控制结果的使用范围（不入表行）\n")
    md.append("- `r7_tts_control` 32 条：tts_request_start→first_pcm mean **7076ms**；仅用于 TTS 服务"
              "延迟归因与审稿回复证据，不作为 Table VIII 行项。")
    md.append("- **引用必带脚注**：`r7_tts_control` was launched after completion of `r7_main` but before "
              "the separately required written authorization and reviewer QA sign-off. The run was retained "
              "under an explicit procedural-deviation waiver because post-run audit found exact "
              "checkpoint/text/hash binding, 32/32 successful calls, and no code or platform divergence "
              "affecting measurement validity; this waiver is not retroactive authorization of the original "
              "execution.\n")

    md.append("## 装配 QA\n")
    for q in qa:
        md.append(f"- {q}")
    md.append("")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")

    print("\n".join(qa))
    print(f"\n写出: {OUT_CSV.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}")
    for lang in ("ALL", "zh", "en"):
        r = red[lang]
        print(f"B vs A ({lang}): mean -{r['mean']:.1f}% ({r['ratio_mean']:.2f}x)  "
              f"p50 -{r['p50']:.1f}% ({r['ratio_p50']:.2f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
