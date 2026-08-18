# experiments/scripts/make_sample_lists.py
"""
生成 CISR 修改版实验的两份样本清单（一次性脚本，保留用于可复现）：

1. repeat_subset_ids.json
   - 来源：exp1 结果 JSON 的 Very Long 组（208 个唯一样本）
   - 规则：sorted() 后用 random.Random(42) 抽 50 个
   - 用途：R1.2 重复测量（50 样本 × 3 轮），R4/R5 复用同一列表

2. exp2_ablation_sample_list.json
   - 来源：exp2 结果 JSON 的 505 个唯一样本
   - 排除规则（显式、可复现，取代旧的手工 static-repair.csv）：
     a) 任一模式运行出错的样本（成对排除，3 个）
     b) 任一流式模式 TTFT > 10000 ms 的样本（判定为平台挂起/跑飞，4 个）
   - 用途：R3 LocalAgreement 基线对比（Table VII），System A/B 数字按同一清单重算

用法：uv run python -m experiments.scripts.make_sample_lists
"""
import json
import random
from pathlib import Path

EXP1 = Path("experiments/results/exp1_latency/exp1_results_20251210_024430.json")
EXP2 = Path("experiments/results/exp2_ablation/exp2_results_20251214_002214.json")
OUT_R1 = Path("experiments/results/revision/r1_stats/repeat_subset_ids.json")
OUT_R3 = Path("experiments/results/revision/r3_baseline_la/exp2_ablation_sample_list.json")

SEED = 42
HANG_THRESHOLD_MS = 10000.0  # 流式模式 TTFT 超过此值判定为平台挂起


def make_repeat_subset():
    data = json.load(open(EXP1, encoding="utf-8"))
    results = data["results"]
    vl = {}
    for r in results:
        if r["duration_group"] == "very_long" and not r.get("error"):
            vl[r["sample_id"]] = r["audio_duration"]
    ids = sorted(vl)
    picked = sorted(random.Random(SEED).sample(ids, 50))
    out = {
        "description": "R1.2 重复测量子集：exp1 Very Long 组固定抽样 50 样本；R4/R5 复用",
        "source": str(EXP1),
        "group": "very_long",
        "seed": SEED,
        "rule": "sorted(sample_ids) -> random.Random(42).sample(50)",
        "n_pool": len(ids),
        "n_pick": len(picked),
        "sample_ids": picked,
        "audio_durations": {sid: vl[sid] for sid in picked},
    }
    OUT_R1.parent.mkdir(parents=True, exist_ok=True)
    OUT_R1.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[repeat_subset] pool={len(ids)} picked={len(picked)} -> {OUT_R1}")
    print(f"  duration range: {min(vl[s] for s in picked):.1f}s - {max(vl[s] for s in picked):.1f}s")


def make_exp2_list():
    data = json.load(open(EXP2, encoding="utf-8"))
    results = data["results"]

    samples = {}  # sample_id -> meta
    err_ids, hang_ids = set(), set()
    for r in results:
        sid = r["sample_id"]
        samples.setdefault(sid, {
            "sample_id": sid,
            "dataset": r["dataset"],
            "language": r["language"],
            "duration_group": r["duration_group"],
            "audio_duration": r["audio_duration"],
        })
        if r.get("error"):
            err_ids.add(sid)
        if r["mode"] != "baseline" and r["ttft"] and r["ttft"] > HANG_THRESHOLD_MS:
            hang_ids.add(sid)

    excluded = sorted(err_ids | hang_ids)
    kept = [samples[sid] for sid in sorted(samples) if sid not in excluded]

    groups = {}
    for s in kept:
        groups.setdefault(s["duration_group"], []).append(s["sample_id"])

    out = {
        "description": "R3 LocalAgreement 基线对比样本清单：与 System A/B 重算数字使用同一干净成对子集",
        "source": str(EXP2),
        "exclusion_rule": {
            "a_runtime_error": "任一模式 error 非空的样本成对排除",
            "b_hang_outlier": f"任一流式模式 TTFT > {HANG_THRESHOLD_MS:.0f} ms 判定平台挂起，成对排除",
            "note": "取代旧的手工 static-repair.csv（其排除清单不可复现）；Table IV 数字按本清单重算后更新",
        },
        "excluded_samples": {
            "runtime_error": sorted(err_ids),
            "hang_outlier": sorted(hang_ids),
        },
        "group_counts": {g: len(ids) for g, ids in sorted(groups.items())},
        "n_total": len(kept),
        "sample_ids": [s["sample_id"] for s in kept],
        "samples": kept,
    }
    OUT_R3.parent.mkdir(parents=True, exist_ok=True)
    OUT_R3.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exp2_list] total={len(samples)} excluded={len(excluded)} kept={len(kept)} -> {OUT_R3}")
    print(f"  groups: {out['group_counts']}")
    print(f"  excluded(runtime_error): {sorted(err_ids)}")
    print(f"  excluded(hang_outlier): {sorted(hang_ids)}")

    # 预览按本清单重算的 Table IV 数字（官方版本由 R1.1 recompute_stats.py 产出）
    kept_ids = set(out["sample_ids"])
    print("\n[preview] 干净成对子集重算的 Table IV 均值（ms）：")
    for g in ["long", "very_long", "extra_long"]:
        row = {}
        for mode in ["baseline", "streaming_asr_only", "full_streaming"]:
            v = [r["ttft"] for r in results
                 if r["sample_id"] in kept_ids and r["duration_group"] == g
                 and r["mode"] == mode and not r.get("error")]
            row[mode] = sum(v) / len(v)
        dur = [samples[sid]["audio_duration"] for sid in groups.get(g, [])]
        print(f"  {g}: dur={sum(dur)/len(dur):.2f}s "
              f"base={row['baseline']:.2f} asr={row['streaming_asr_only']:.2f} "
              f"full={row['full_streaming']:.2f} "
              f"asr_gain={row['baseline']-row['streaming_asr_only']:.2f} "
              f"kv_gain={row['streaming_asr_only']-row['full_streaming']:.2f}")


if __name__ == "__main__":
    make_repeat_subset()
    make_exp2_list()
