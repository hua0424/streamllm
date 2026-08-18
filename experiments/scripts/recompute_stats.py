# experiments/scripts/recompute_stats.py
"""
R1.1：离线重算 exp1/exp2/exp3 的分位数统计（对应审稿意见3）。

输入（均为已有结果 JSON，不需要 GPU）：
- exp1: experiments/results/exp1_latency/exp1_results_20251210_024430.json
- exp2: experiments/results/exp2_ablation/exp2_results_20251214_002214.json
        + exp2_ablation_sample_list.json（干净成对子集，Table IV 口径）
- exp3: experiments/results/exp3_quality/{suffix0_result,suffix1_result,prefix0suffix0}/exp3_results_*.json

过滤规则（显式、可复现）：
- 排除 error 非空的记录；exp1/exp2 按样本成对排除（同一样本任一模式出错则全模式排除）
- 流式模式 TTFT > 10000 ms 判定平台挂起，成对排除（exp1 无此类记录，规则仅作声明）

输出（experiments/results/revision/r1_stats/）：
- table3_latency_percentiles.csv   论文 Table III 数据（exp1：分组 × streaming/non-streaming）
- table4_ablation_percentiles.csv  论文 Table IV 数据（exp2：分组 × 3 模式 + 贡献分解）
- table5_context_percentiles.csv   论文 Table V 数据（exp3：3 种上下文创口的 WER/CER/ASR 时间）
- plateau_stability.txt            System B 长语音组的 P95/P99 与平台稳定性判定

分位数方法：numpy.percentile 线性插值（type 7，与 pandas 默认一致）。

用法：uv run python -m experiments.scripts.recompute_stats
"""
import json
from pathlib import Path

import numpy as np

EXP1 = Path("experiments/results/exp1_latency/exp1_results_20251210_024430.json")
EXP2 = Path("experiments/results/exp2_ablation/exp2_results_20251214_002214.json")
EXP2_LIST = Path("experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json")
EXP3_DIRS = {
    "prefix1_suffix1": Path("experiments/results/exp3_quality/suffix1_result"),
    "prefix1_suffix0_default": Path("experiments/results/exp3_quality/suffix0_result"),
    "prefix0_suffix0": Path("experiments/results/exp3_quality/prefix0suffix0"),
}
OUT_DIR = Path("experiments/results/revision/r1_stats")

HANG_THRESHOLD_MS = 10000.0
GROUP_ORDER = ["short", "medium", "long", "very_long", "extra_long"]
GROUP_LABEL = {"short": "Short", "medium": "Medium", "long": "Long",
               "very_long": "Very Long", "extra_long": "Extra Long"}


def pct_stats(values):
    a = np.asarray(values, dtype=float)
    return {
        "n": len(a), "mean": a.mean(), "std": a.std(ddof=1) if len(a) > 1 else 0.0,
        "p50": np.percentile(a, 50), "p90": np.percentile(a, 90),
        "p95": np.percentile(a, 95), "p99": np.percentile(a, 99),
        "min": a.min(), "max": a.max(),
    }


def fmt(v):
    return f"{v:.2f}"


def write_csv(path, header, rows):
    path.write_text(",".join(header) + "\n" + "".join(",".join(r) + "\n" for r in rows),
                    encoding="utf-8")
    print(f"  -> {path}  ({len(rows)} rows)")


def paired_filter(results, modes):
    """按样本成对排除：任一所选模式 error/缺 ttft/流式挂起 → 排除该样本全部模式。"""
    bad = set()
    for r in results:
        if r["mode"] not in modes:
            continue
        if r.get("error") or not r.get("ttft"):
            bad.add(r["sample_id"])
        elif r["mode"] != "baseline" and r["mode"] != "non-streaming" \
                and r["ttft"] > HANG_THRESHOLD_MS:
            bad.add(r["sample_id"])
    return [r for r in results if r["mode"] in modes and r["sample_id"] not in bad], bad


def table3():
    print("[Table III] exp1 latency percentiles")
    data = json.load(open(EXP1, encoding="utf-8"))
    results, bad = paired_filter(data["results"], ["streaming", "non-streaming"])
    print(f"  paired-excluded samples: {sorted(bad) if bad else 'none'}")
    header = ["group", "mode", "n", "mean_ms", "std_ms", "p50_ms", "p90_ms",
              "p95_ms", "p99_ms", "min_ms", "max_ms"]
    rows, sysb = [], {}
    for g in GROUP_ORDER:
        for mode in ["non-streaming", "streaming"]:
            v = [r["ttft"] for r in results if r["duration_group"] == g and r["mode"] == mode]
            if not v:
                continue
            s = pct_stats(v)
            rows.append([GROUP_LABEL[g], mode, str(s["n"])] + [fmt(s[k]) for k in
                        ["mean", "std", "p50", "p90", "p95", "p99", "min", "max"]])
            if mode == "streaming":
                sysb[g] = s
    write_csv(OUT_DIR / "table3_latency_percentiles.csv", header, rows)

    lines = ["System B (streaming) plateau stability, exp1",
             "claim: streaming TTFT tail stays bounded as duration grows; baseline tail grows ~linearly", ""]
    base = {}
    for g in GROUP_ORDER:
        v = [r["ttft"] for r in results if r["duration_group"] == g and r["mode"] == "non-streaming"]
        if v:
            base[g] = pct_stats(v)
    for g in ["long", "very_long", "extra_long"]:
        s, b = sysb[g], base[g]
        lines.append(f"{GROUP_LABEL[g]:<10} n={s['n']:<4} mean={s['mean']:8.2f}  "
                     f"P95={s['p95']:8.2f}  P99={s['p99']:8.2f}  P99/mean={s['p99']/s['mean']:.2f}  "
                     f"| baseline P99={b['p99']:8.2f}  streaming/baseline P99={s['p99']/b['p99']:.2f}")
    g_l, g_x = sysb["long"], sysb["extra_long"]
    lines += ["",
              f"streaming P99 growth Long->Extra Long: {g_x['p99']/g_l['p99']:.2f}x "
              f"(baseline: {base['extra_long']['p99']/base['long']['p99']:.2f}x)",
              f"streaming Extra Long P99 abs: {g_x['p99']:.2f} ms "
              f"(baseline: {base['extra_long']['p99']:.2f} ms)"]
    (OUT_DIR / "plateau_stability.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join("  " + l for l in lines))


def table4():
    print("[Table IV] exp2 ablation percentiles (clean paired subset)")
    data = json.load(open(EXP2, encoding="utf-8"))
    keep = set(json.load(open(EXP2_LIST, encoding="utf-8"))["sample_ids"])
    results = [r for r in data["results"] if r["sample_id"] in keep]
    header = ["group", "mode", "n", "mean_ms", "std_ms", "p50_ms", "p90_ms",
              "p95_ms", "p99_ms", "min_ms", "max_ms"]
    rows, means = [], {}
    modes = [("baseline", "baseline"), ("streaming_asr_only", "streaming_asr_only"),
             ("full_streaming", "full_streaming")]
    for g in ["long", "very_long", "extra_long"]:
        for mode, label in modes:
            v = [r["ttft"] for r in results if r["duration_group"] == g and r["mode"] == mode]
            s = pct_stats(v)
            means[(g, mode)] = s["mean"]
            rows.append([GROUP_LABEL[g], label, str(s["n"])] + [fmt(s[k]) for k in
                        ["mean", "std", "p50", "p90", "p95", "p99", "min", "max"]])
        b, a, f = means[(g, "baseline")], means[(g, "streaming_asr_only")], means[(g, "full_streaming")]
        rows.append([GROUP_LABEL[g], "asr_gain_ms (=baseline-asr_only)", "", fmt(b - a),
                     "", "", "", "", "", "", ""])
        rows.append([GROUP_LABEL[g], "kv_gain_ms (=asr_only-full)", "", fmt(a - f),
                     "", "", "", "", "", "", ""])
    write_csv(OUT_DIR / "table4_ablation_percentiles.csv", header, rows)


def table5():
    print("[Table V] exp3 context-window percentiles")
    header = ["config", "mode", "metric", "subset", "n", "mean", "std",
              "p50", "p90", "p95", "p99", "min", "max"]
    rows = []
    for cfg, d in EXP3_DIRS.items():
        p = next(d.glob("exp3_results_*.json"))
        data = json.load(open(p, encoding="utf-8"))
        res = [r for r in data["results"] if not r.get("error")]
        #  pooled（两模式合并）均值：论文 Table V "ASR time" 列的原始口径，保留以便直接核对
        v_all = [r["asr_time_ms"] for r in res]
        s = pct_stats(v_all)
        rows.append([cfg, "pooled", "asr_time_ms", "all", str(s["n"])] +
                    [fmt(s[k]) for k in ["mean", "std", "p50", "p90", "p95", "p99", "min", "max"]])
        for mode in ["streaming", "non-streaming"]:
            sub = [r for r in res if r["mode"] == mode]
            v = [r["asr_time_ms"] for r in sub]
            s = pct_stats(v)
            rows.append([cfg, mode, "asr_time_ms", "all", str(s["n"])] +
                        [fmt(s[k]) for k in ["mean", "std", "p50", "p90", "p95", "p99", "min", "max"]])
            for ds in ["multiwoz", "crosswoz"]:
                for metric in ["wer", "cer"]:
                    v = [r[metric] for r in sub if r["dataset"] == ds]
                    s = pct_stats(v)
                    rows.append([cfg, mode, metric, ds, str(s["n"])] +
                                [f"{s[k]:.4f}" for k in ["mean", "std", "p50", "p90", "p95", "p99", "min", "max"]])
    write_csv(OUT_DIR / "table5_context_percentiles.csv", header, rows)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table3()
    table4()
    table5()
    print("done.")
