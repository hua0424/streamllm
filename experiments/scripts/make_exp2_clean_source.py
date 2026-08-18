# experiments/scripts/make_exp2_clean_source.py
"""
生成 exp2 消融实验的干净数据来源文件（排除异常落盘，供复查与重新统计）。

背景：论文 Table IV 旧口径来自 exp2_gains_20251214_002214.csv 的手工排除，
排除依据未落盘，导致旧数字无法精确还原。本脚本以结果 JSON 为唯一真实来源，
按 exp2_ablation_sample_list.json 中登记的显式排除规则生成：

输出（experiments/results/exp2_ablation/）：
1. exp2_gains_clean.csv        —— 498 条保留样本的逐样本三模式数据（全精度，
   列布局与原 exp2_gains CSV 一致），后续一切 Table IV 统计以此为准；
2. exp2_gains_exclusions.csv   —— 7 条被排除样本及排除原因与触发值；
3. 控制台验证报告：
   a. 保留/排除样本 ID 与原 gains CSV 的样本集合完全重合（498+7=505）；
   b. 保留样本的逐样本数值与原 gains CSV（2 位小数）一致；
   c. 由 clean 文件计算的分组均值与 recompute_stats.py 的 Table IV 口径一致。

用法：uv run python -m experiments.scripts.make_exp2_clean_source [--results-json ...] ...
所有路径参数均有默认值指向当前 exp2 归档；实际输入输出写入 clean 文件旁的
exp2_gains_clean.meta.json（R2-P1-1）。
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

DEFAULT_EXP2_JSON = "experiments/results/exp2_ablation/exp2_results_20251214_002214.json"
DEFAULT_ORIG_GAINS = "experiments/results/exp2_ablation/exp2_gains_20251214_002214.csv"
DEFAULT_SAMPLE_LIST = "experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json"
DEFAULT_OUT_CLEAN = "experiments/results/exp2_ablation/exp2_gains_clean.csv"
DEFAULT_OUT_EXCL = "experiments/results/exp2_ablation/exp2_gains_exclusions.csv"

MODES = ["baseline", "streaming_asr_only", "full_streaming"]
# 模式 -> 原 gains CSV 列名前缀
MODE_COL = {"baseline": "baseline", "streaming_asr_only": "streaming_asr",
            "full_streaming": "full_streaming"}


def parse_args():
    parser = argparse.ArgumentParser(description="生成 exp2 干净数据来源文件（排除异常落盘）")
    parser.add_argument('--results-json', type=Path, default=Path(DEFAULT_EXP2_JSON),
                        help='exp2 结果 JSON（唯一真实来源）')
    parser.add_argument('--gains-csv', type=Path, default=Path(DEFAULT_ORIG_GAINS),
                        help='原逐样本 gains CSV（用于重合验证）')
    parser.add_argument('--sample-list', type=Path, default=Path(DEFAULT_SAMPLE_LIST),
                        help='干净成对子集清单 JSON（排除规则的登记处）')
    parser.add_argument('--output-clean', type=Path, default=Path(DEFAULT_OUT_CLEAN))
    parser.add_argument('--output-exclusions', type=Path, default=Path(DEFAULT_OUT_EXCL))
    return parser.parse_args()


def main(args):
    EXP2_JSON, ORIG_GAINS, SAMPLE_LIST = args.results_json, args.gains_csv, args.sample_list
    OUT_CLEAN, OUT_EXCL = args.output_clean, args.output_exclusions
    results = json.load(open(EXP2_JSON, encoding="utf-8"))["results"]
    keep = set(json.load(open(SAMPLE_LIST, encoding="utf-8"))["sample_ids"])

    # 建索引 sample_id -> mode -> result
    by_sample = {}
    for r in results:
        by_sample.setdefault(r["sample_id"], {})[r["mode"]] = r

    clean_rows, excl_rows = [], []
    for sid in sorted(by_sample):
        entries = by_sample[sid]
        meta = entries[MODES[0]]
        base = dict(
            sample_id=sid,
            dataset=meta["dataset"],
            language=meta["language"],
            dialog_id=meta["dialog_id"],
            turn_index=meta["turn_index"],
            text_length=meta["text_length"],
            duration_group=meta["duration_group"],
            audio_duration_s=round(meta["audio_duration"], 2),
        )
        if sid in keep:
            row = dict(base)
            vals = {}
            for m in MODES:
                vals[m] = entries[m]["ttft"]
                row[MODE_COL[m] + "_ttft_ms"] = round(vals[m], 2)
            asr_gain = vals["baseline"] - vals["streaming_asr_only"]
            kv_gain = vals["streaming_asr_only"] - vals["full_streaming"]
            total_gain = vals["baseline"] - vals["full_streaming"]
            row.update(
                asr_gain_ms=round(asr_gain, 2),
                kv_gain_ms=round(kv_gain, 2),
                total_gain_ms=round(total_gain, 2),
                **{"total_gain_ratio_%": round(total_gain / vals["baseline"] * 100, 1)
                   if vals["baseline"] > 0 else ""},
            )
            clean_rows.append(row)
        else:
            # 排除原因与触发值
            reasons, trigger = [], ""
            for m in MODES:
                e = entries.get(m, {})
                if e.get("error"):
                    reasons.append("runtime_error")
                    trigger = f"{m}: {e['error'][:60]}"
                    break
            if not reasons:
                hang = [(m, entries[m]["ttft"]) for m in MODES[1:]
                        if entries.get(m, {}).get("ttft", 0) > 10000]
                if hang:
                    reasons.append("hang_outlier_ttft>10000ms")
                    trigger = "; ".join(f"{m}={v:.1f}ms" for m, v in hang)
            excl_rows.append(dict(base, exclusion_reason="|".join(reasons), trigger_value=trigger))

    assert len(clean_rows) == 498 and len(excl_rows) == 7, (len(clean_rows), len(excl_rows))

    # 写 clean 文件（列名与原 gains CSV 完全一致）
    header = ["sample_id", "dataset", "language", "dialog_id", "turn_index", "text_length",
              "duration_group", "audio_duration_s", "baseline_ttft_ms", "streaming_asr_ttft_ms",
              "full_streaming_ttft_ms", "asr_gain_ms", "kv_gain_ms", "total_gain_ms",
              "total_gain_ratio_%"]
    with open(OUT_CLEAN, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in clean_rows:
            w.writerow([row["sample_id"], row["dataset"], row["language"], row["dialog_id"],
                        row["turn_index"], row["text_length"], row["duration_group"],
                        f"{row['audio_duration_s']:.2f}",
                        f"{row['baseline_ttft_ms']:.2f}", f"{row['streaming_asr_ttft_ms']:.2f}",
                        f"{row['full_streaming_ttft_ms']:.2f}",
                        f"{row['asr_gain_ms']:.2f}", f"{row['kv_gain_ms']:.2f}",
                        f"{row['total_gain_ms']:.2f}", row["total_gain_ratio_%"]])

    with open(OUT_EXCL, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "dataset", "language", "duration_group",
                    "audio_duration_s", "exclusion_reason", "trigger_value"])
        for row in excl_rows:
            w.writerow([row["sample_id"], row["dataset"], row["language"], row["duration_group"],
                        f"{row['audio_duration_s']:.2f}", row["exclusion_reason"], row["trigger_value"]])

    # ===== 验证 a：与原 gains CSV 样本集合的关系 =====
    # 原 gains CSV 生成时已剔除运行错误样本（505-3=502 行），
    # 因此：原 gains = 保留 498 + 挂起排除 4；运行错误 3 条仅在排除清单中登记。
    orig = {r["sample_id"]: r for r in csv.DictReader(open(ORIG_GAINS, encoding="utf-8"))}
    kept_ids = {r["sample_id"] for r in clean_rows}
    hang_ids = {r["sample_id"] for r in excl_rows if "hang_outlier" in r["exclusion_reason"]}
    err_ids = {r["sample_id"] for r in excl_rows if "runtime_error" in r["exclusion_reason"]}
    assert set(orig) == kept_ids | hang_ids, "样本集合不重合"
    assert not (kept_ids & hang_ids) and not (set(orig) & err_ids)
    print(f"[verify-a] 原 gains {len(orig)} 行 = 保留 {len(kept_ids)} + 挂起排除 {len(hang_ids)} ✓ "
          f"(运行错误 {len(err_ids)} 条原 CSV 本就不含，已在排除清单登记)")

    # ===== 验证 b：保留样本逐样本数值与原 gains CSV 一致（2 位小数）=====
    mismatch = []
    for row in clean_rows:
        o = orig[row["sample_id"]]
        for col, key in [("baseline_ttft_ms", "baseline_ttft_ms"),
                         ("streaming_asr_ttft_ms", "streaming_asr_ttft_ms"),
                         ("full_streaming_ttft_ms", "full_streaming_ttft_ms")]:
            if abs(float(o[col]) - row[key]) > 0.011:
                mismatch.append((row["sample_id"], col, o[col], row[key]))
    assert not mismatch, mismatch[:5]
    print(f"[verify-b] {len(clean_rows)} 条保留样本的三模式数值与原 gains CSV 完全一致（2dp）✓")

    # ===== 验证 c：分组均值与 Table IV 新口径一致 =====
    print("[verify-c] 由 clean 文件计算的分组均值（ms）：")
    for g in ["long", "very_long", "extra_long"]:
        rows = [r for r in clean_rows if r["duration_group"] == g]
        n = len(rows)
        means = {MODE_COL[m]: sum(r[MODE_COL[m] + "_ttft_ms"] for r in rows) / n for m in MODES}
        print(f"  {g}: n={n} base={means['baseline']:.2f} asr={means['streaming_asr']:.2f} "
              f"full={means['full_streaming']:.2f} "
              f"asr_gain={means['baseline'] - means['streaming_asr']:.2f} "
              f"kv_gain={means['streaming_asr'] - means['full_streaming']:.2f}")

    # sidecar metadata：登记实际输入输出与规模（R2-P1-1）
    meta = {
        "generated_at": datetime.now().isoformat(),
        "inputs": {"results_json": str(EXP2_JSON), "orig_gains_csv": str(ORIG_GAINS),
                   "sample_list": str(SAMPLE_LIST)},
        "outputs": {"clean": str(OUT_CLEAN), "exclusions": str(OUT_EXCL)},
        "counts": {"total": len(by_sample), "kept": len(clean_rows), "excluded": len(excl_rows)},
        "group_counts": {},
    }
    for row in clean_rows:
        g = row["duration_group"]
        meta["group_counts"][g] = meta["group_counts"].get(g, 0) + 1
    OUT_CLEAN.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n输出:\n  {OUT_CLEAN}\n  {OUT_EXCL}\n  {OUT_CLEAN.with_suffix('.meta.json')}")


if __name__ == "__main__":
    main(parse_args())
