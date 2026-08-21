"""W5：成对统计推断（paired bootstrap CI + Wilcoxon + 效应量）。

对应 PRE-PAPER-AUDIT P1-1 与 v3.1 唯一冻结比较族：
- F1 Table III（原平台）：总体 A/B 主比较；Long/Very Long/Extra Long 三比较为一个 Holm family；
- F2 Table VII（第二平台）：B vs LA 主比较、A vs B 验证性比较（分开标注，不并入校正族）；
- F3 R2：clean 两条件单独报原始 p；十二增强条件为一个 Holm family；
- F4 R5：solo 评分 B−A 配对均值差只报 bootstrap 95% CI，不做等价性检验。

冻结协议：
- bootstrap 以 sample ID 为配对重采样单位，10,000 次，seed=20260821，percentile 95% CI；
- 差值方向：延迟类 diff = A − B（正值=B 更快）；改善率 = (mean(A)−mean(B))/mean(A)
  （比值均值口径，不与逐样本改善率均值混用）；R5 diff = B − A；
- Wilcoxon：双侧、zero_method='wilcox'、correction=False、method='auto'；
  全零差 → p 记 NaN 并标 all_zero_diff；
- 效应量：rank-biserial correlation（正=前一系统更慢/更差）与 paired Cohen's dz
  （差值零方差时 dz 记 NaN 并标 zero_variance）；
- 文字规则：CI 跨 0 或校正 p 不达标不写"统计显著"；不显著≠等价。

用法：
  uv run python -m experiments.scripts.paired_inference \
      --table3 <exp1 archive json> --table3-manifest <manifest json> \
      --table7-ab <system_ab_rerun json> --table7-la <la_results json> \
      --r2-dir <r2_real_speech dir> --r5-csv <semantic_consistency.csv> \
      --out-dir experiments/results/revision/stats_inference
  uv run python -m experiments.scripts.paired_inference --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy

N_BOOT = 10_000
BOOT_SEED = 20260821
R2_AUGMENT_CONDITIONS = ("snr10", "snr15", "snr20", "babble", "speed09", "speed11")
R2_DATASETS = ("librispeech", "aishell1")
HANG_TTFT_MS = 10_000.0  # 流式 TTFT>10s 判挂起（与 table3 manifest 规则一致）


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    recs = data if isinstance(data, list) else data.get("results", list(data.values()))
    if isinstance(recs, dict):
        recs = list(recs.values())
    return recs


def extract_pairs(records: list[dict], mode_slow: str, mode_fast: str,
                  exclude_ids: set[str] | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """按 sample_id 对齐两模式 TTFT，返回 (ids, slow, fast)。

    排除：error 行、ttft 非有限正值、任一侧缺失、exclude_ids。
    挂起过滤（TTFT>10s）只适用于流式侧——非流式 extra_long 合法超 10s
    （P99≈12.3s），与 table3 manifest 规则"流式模式 TTFT>10000ms 判定挂起"一致。
    """
    exclude_ids = exclude_ids or set()
    by_id: dict[str, dict[str, float]] = {}
    for r in records:
        sid = str(r.get("sample_id", ""))
        if not sid or sid in exclude_ids or r.get("error"):
            continue
        mode = str(r.get("mode", ""))
        if mode not in (mode_slow, mode_fast):
            continue
        try:
            v = float(r["ttft"])
        except (TypeError, ValueError):
            continue
        hang_ok = mode == "non-streaming" or v <= HANG_TTFT_MS
        if not np.isfinite(v) or v <= 0 or not hang_ok:
            continue
        by_id.setdefault(sid, {})[mode] = v
    ids = sorted(sid for sid, d in by_id.items() if mode_slow in d and mode_fast in d)
    return ids, np.array([by_id[s][mode_slow] for s in ids]), np.array([by_id[s][mode_fast] for s in ids])


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT,
                     seed: int = BOOT_SEED) -> dict:
    """配对 bootstrap：diff=A−B 均值与改善率的 percentile 95% CI。"""
    n = len(a)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    imps = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ma, mb = a[idx].mean(), b[idx].mean()
        diffs[i] = ma - mb
        imps[i] = (ma - mb) / ma if ma != 0 else np.nan
    return {
        "diff_mean": float(a.mean() - b.mean()),
        "diff_ci_lo": float(np.percentile(diffs, 2.5)),
        "diff_ci_hi": float(np.percentile(diffs, 97.5)),
        "improvement": float((a.mean() - b.mean()) / a.mean()),
        "improvement_ci_lo": float(np.nanpercentile(imps, 2.5)),
        "improvement_ci_hi": float(np.nanpercentile(imps, 97.5)),
    }


def wilcoxon_effect(a: np.ndarray, b: np.ndarray) -> dict:
    """双侧 Wilcoxon（wilcox/auto/无连续性校正）+ rank-biserial + Cohen's dz。"""
    from scipy.stats import wilcoxon
    d = a - b
    if np.all(d == 0):
        return {"wilcoxon_stat": float("nan"), "p_raw": float("nan"), "note": "all_zero_diff",
                "rank_biserial": float("nan"), "cohens_dz": float("nan")}
    res = wilcoxon(d, zero_method="wilcox", correction=False,
                   alternative="two-sided", method="auto")
    # rank-biserial：去掉零差后，正秩和与负秩和之差 / 总和（正=A 侧更大）
    nz = d[d != 0]
    ranks = _rankdata_abs(nz)
    w_pos = ranks[nz > 0].sum()
    w_neg = ranks[nz < 0].sum()
    rbc = float((w_pos - w_neg) / (w_pos + w_neg))
    sd = d.std(ddof=1)
    dz = float(d.mean() / sd) if sd > 0 else float("nan")
    return {"wilcoxon_stat": float(res.statistic), "p_raw": float(res.pvalue),
            "note": "" if sd > 0 else "zero_variance",
            "rank_biserial": rbc, "cohens_dz": dz}


def _rankdata_abs(d: np.ndarray) -> np.ndarray:
    """|d| 的秩（平均秩处理并列），与 scipy.stats.rankdata 一致。"""
    from scipy.stats import rankdata
    return rankdata(np.abs(d))


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm 校正（NaN 保留原位不参与）。"""
    n = len(pvals)
    indexed = [(p, i) for i, p in enumerate(pvals) if not np.isnan(p)]
    indexed.sort(key=lambda t: t[0])
    adjusted = [float("nan")] * n
    running = 0.0
    m = len(indexed)
    for rank, (p, i) in enumerate(indexed):
        running = max(running, min(p * (m - rank), 1.0))
        adjusted[i] = running
    return adjusted


def compare(name: str, family: str, role: str, ids, a, b, unit: str = "ms") -> dict:
    bs = paired_bootstrap(a, b)
    wx = wilcoxon_effect(a, b)
    return {
        "comparison": name, "family": family, "role": role, "unit": unit,
        "n_pairs": len(ids),
        "mean_slow": f"{a.mean():.4f}", "mean_fast": f"{b.mean():.4f}",
        "diff_mean": f"{bs['diff_mean']:.4f}",
        "diff_ci95": f"[{bs['diff_ci_lo']:.4f}, {bs['diff_ci_hi']:.4f}]",
        "improvement": f"{bs['improvement']:.4f}",
        "improvement_ci95": f"[{bs['improvement_ci_lo']:.4f}, {bs['improvement_ci_hi']:.4f}]",
        "wilcoxon_stat": f"{wx['wilcoxon_stat']:.1f}" if not np.isnan(wx["wilcoxon_stat"]) else "",
        "p_raw": f"{wx['p_raw']:.3e}" if not np.isnan(wx["p_raw"]) else "",
        "rank_biserial": f"{wx['rank_biserial']:.4f}" if not np.isnan(wx["rank_biserial"]) else "",
        "cohens_dz": f"{wx['cohens_dz']:.4f}" if not np.isnan(wx["cohens_dz"]) else "",
        "note": wx["note"],
        "_p": wx["p_raw"],
    }


# ---------------------------------------------------------------- self-test

def _self_test() -> int:
    fails = []

    def check(name, cond, detail=""):
        if not cond:
            fails.append(f"{name}: {detail}")

    # 1) 全正差：n=5，Wilcoxon 双侧 exact p = 2/2^5 = 0.0625；rank-biserial=1
    a = np.array([5.0, 6, 7, 8, 9]); b = np.array([4.0, 5, 6, 7, 8])
    wx = wilcoxon_effect(a, b)
    check("wilcoxon 全正 p=0.0625", abs(wx["p_raw"] - 0.0625) < 1e-9, str(wx["p_raw"]))
    check("rank-biserial=1", abs(wx["rank_biserial"] - 1.0) < 1e-9, str(wx["rank_biserial"]))
    check("零方差 dz=NaN+note", np.isnan(wx["cohens_dz"]) and wx["note"] == "zero_variance", wx["note"])

    # 2) 全零差
    wx0 = wilcoxon_effect(a, a)
    check("全零差 NaN+note", np.isnan(wx0["p_raw"]) and wx0["note"] == "all_zero_diff")

    # 3) bootstrap 可复现 + CI 含点估计
    rng = np.random.default_rng(7)
    aa = rng.normal(1000, 200, 40); bb = aa - rng.normal(300, 100, 40)
    r1 = paired_bootstrap(aa, bb); r2 = paired_bootstrap(aa, bb)
    check("bootstrap 可复现", r1 == r2)
    check("CI 含点估计", r1["diff_ci_lo"] <= r1["diff_mean"] <= r1["diff_ci_hi"], str(r1))
    check("改善率口径", abs(r1["improvement"] - (aa.mean() - bb.mean()) / aa.mean()) < 1e-12)

    # 4) Holm 已知例：p=[0.01, 0.04, 0.03] → [0.03, 0.06, 0.06]
    adj = holm_adjust([0.01, 0.04, 0.03])
    check("Holm", np.allclose(adj, [0.03, 0.06, 0.06]), str(adj))
    adj2 = holm_adjust([float("nan"), 0.05])
    check("Holm NaN 保留", np.isnan(adj2[0]) and abs(adj2[1] - 0.05) < 1e-12, str(adj2))

    # 5) 配对抽取：error/缺侧/挂起/exclude 全排除
    recs = [
        {"sample_id": "s1", "mode": "non-streaming", "ttft": 3000.0},
        {"sample_id": "s1", "mode": "streaming", "ttft": 1000.0},
        {"sample_id": "s2", "mode": "non-streaming", "ttft": 3000.0, "error": "x"},
        {"sample_id": "s2", "mode": "streaming", "ttft": 1000.0},
        {"sample_id": "s3", "mode": "non-streaming", "ttft": 3000.0},
        {"sample_id": "s3", "mode": "streaming", "ttft": 15000.0},   # 挂起
        {"sample_id": "s4", "mode": "non-streaming", "ttft": 3000.0},
        {"sample_id": "s4", "mode": "streaming", "ttft": 1000.0},
    ]
    ids, sa, fa = extract_pairs(recs, "non-streaming", "streaming", exclude_ids={"s4"})
    check("配对过滤", ids == ["s1"], str(ids))

    if fails:
        for f in fails:
            print(f"FAIL {f}")
        return 1
    print("self-test: ALL PASS")
    return 0


# ---------------------------------------------------------------- 正式装配

def run_all(args) -> tuple[list[dict], list[dict]]:
    """返回 (rows, inputs_meta)。"""
    inputs_meta = []
    rows: list[dict] = []

    def reg(path):
        inputs_meta.append({"path": str(path), "sha256": sha256_file(str(path))})

    # F1 Table III（原平台归档 + manifest 成对排除）
    reg(args.table3); reg(args.table3_manifest)
    recs = load_records(args.table3)
    manifest = json.loads(Path(args.table3_manifest).read_text(encoding="utf-8"))
    excl = {e["sample_id"] for e in manifest.get("excluded_samples", [])}
    groups_of = {}
    for r in recs:
        groups_of.setdefault(str(r.get("sample_id", "")), str(r.get("duration_group", "")))
    for label, group in [("table3_long", "long"), ("table3_very_long", "very_long"),
                         ("table3_extra_long", "extra_long")]:
        sub = [r for r in recs if str(r.get("duration_group", "")) == group]
        ids, a, b = extract_pairs(sub, "non-streaming", "streaming", excl)
        expected = manifest["groups"][group]["final_n"]
        if len(ids) != expected:
            raise SystemExit(f"{label} 配对数 {len(ids)} != manifest final_n {expected}（停止）")
        rows.append(compare(label, "table3_groups", "Holm 族内", ids, a, b))
    ids_all = [r for r in recs if str(r.get("duration_group", "")) in
               ("long", "very_long", "extra_long")]
    ids, a, b = extract_pairs(ids_all, "non-streaming", "streaming", excl)
    rows.append(compare("table3_overall", "standalone", "主比较", ids, a, b))

    # F2 Table VII（第二平台 A/B 重跑 + LA）
    reg(args.table7_ab); reg(args.table7_la)
    ab_recs = load_records(args.table7_ab)
    la_recs = load_records(args.table7_la)
    ids_ab, a_arr, b_arr = extract_pairs(ab_recs, "non-streaming", "streaming")
    rows.append(compare("table7_a_vs_b", "standalone", "验证性比较", ids_ab, a_arr, b_arr))
    # B vs LA：LA 记录与 B 记录跨文件配对
    b_map = {str(r["sample_id"]): float(r["ttft"]) for r in ab_recs
             if r.get("mode") == "streaming" and not r.get("error")
             and 0 < float(r["ttft"]) <= HANG_TTFT_MS}
    la_map = {str(r["sample_id"]): float(r["ttft"]) for r in la_recs
              if not r.get("error") and 0 < float(r["ttft"]) <= HANG_TTFT_MS}
    ids_bl = sorted(set(b_map) & set(la_map))
    rows.append(compare("table7_b_vs_la", "standalone", "主比较", ids_bl,
                        np.array([la_map[s] for s in ids_bl]),
                        np.array([b_map[s] for s in ids_bl])))

    # F3 R2：clean 单独 + 十二增强条件 Holm 族
    r2 = Path(args.r2_dir)
    for ds in R2_DATASETS:
        f = sorted((r2 / f"{ds}_clean").glob("exp1_results_*.json"))
        if len(f) != 1:
            raise SystemExit(f"{ds}_clean 结果文件不唯一: {f}（停止）")
        reg(f[0])
        ids, a, b = extract_pairs(load_records(str(f[0])), "non-streaming", "streaming")
        rows.append(compare(f"r2_{ds}_clean", "standalone", "主比较(干净集)", ids, a, b))
    for cond in R2_AUGMENT_CONDITIONS:
        for ds in R2_DATASETS:
            f = sorted((r2 / f"{ds}_{cond}").glob("exp1_results_*.json"))
            if len(f) != 1:
                raise SystemExit(f"{ds}_{cond} 结果文件不唯一: {f}（停止）")
            reg(f[0])
            ids, a, b = extract_pairs(load_records(str(f[0])), "non-streaming", "streaming")
            rows.append(compare(f"r2_{ds}_{cond}", "r2_augmented", "Holm 族内", ids, a, b))

    # F4 R5：solo 评分 B−A，只报 CI
    reg(args.r5_csv)
    with open(args.r5_csv, encoding="utf-8-sig", newline="") as fh:
        srows = [r for r in csv.DictReader(fh)
                 if r.get("solo_score_A") and r.get("solo_score_B")]
    sa = np.array([float(r["solo_score_A"]) for r in srows])
    sb = np.array([float(r["solo_score_B"]) for r in srows])
    bs = paired_bootstrap(sb, sa)  # diff = B − A
    rows.append({
        "comparison": "r5_solo_b_minus_a", "family": "standalone",
        "role": "探索性（仅 CI，不做等价性检验）", "unit": "score",
        "n_pairs": len(srows),
        "mean_slow": f"{sb.mean():.4f}", "mean_fast": f"{sa.mean():.4f}",
        "diff_mean": f"{bs['diff_mean']:.4f}",
        "diff_ci95": f"[{bs['diff_ci_lo']:.4f}, {bs['diff_ci_hi']:.4f}]",
        "improvement": "", "improvement_ci95": "",
        "wilcoxon_stat": "", "p_raw": "", "rank_biserial": "", "cohens_dz": "",
        "note": "bootstrap_only; 差值方向 B−A", "_p": float("nan"),
    })

    # Holm 校正回填
    for fam in ("table3_groups", "r2_augmented"):
        idxs = [i for i, r in enumerate(rows) if r["family"] == fam]
        adj = holm_adjust([rows[i]["_p"] for i in idxs])
        for i, p in zip(idxs, adj):
            rows[i]["p_holm"] = "" if np.isnan(p) else f"{p:.3e}"
    for r in rows:
        r.setdefault("p_holm", "")
        del r["_p"]
    return rows, inputs_meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table3")
    ap.add_argument("--table3-manifest")
    ap.add_argument("--table7-ab")
    ap.add_argument("--table7-la")
    ap.add_argument("--r2-dir")
    ap.add_argument("--r5-csv")
    ap.add_argument("--out-dir")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    missing = [k for k in ("table3", "table3_manifest", "table7_ab", "table7_la",
                           "r2_dir", "r5_csv", "out_dir") if getattr(args, k) is None]
    if missing:
        ap.error(f"缺参数: {missing}")

    rows, inputs_meta = run_all(args)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = ["comparison", "family", "role", "unit", "n_pairs", "mean_slow", "mean_fast",
              "diff_mean", "diff_ci95", "improvement", "improvement_ci95",
              "wilcoxon_stat", "p_raw", "p_holm", "rank_biserial", "cohens_dz", "note"]
    with open(out / "paired_inference.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = ["# 成对统计推断（W5 冻结协议）", "",
          f"- bootstrap：paired、{N_BOOT} 次、seed={BOOT_SEED}、percentile 95% CI；",
          "- Wilcoxon：双侧、zero_method='wilcox'、correction=False、method='auto'；",
          "- 差值方向：延迟类 A−B（正=B 更快）；R5 为 B−A；改善率=(mean(A)−mean(B))/mean(A)；",
          f"- scipy {scipy.__version__}；", "", "## 输入 SHA-256", ""]
    for it in inputs_meta:
        md.append(f"- `{it['path']}` : `{it['sha256']}`")
    md += ["", "## 结果", "",
           "| comparison | family | n | diff mean | diff CI95 | improvement [CI] | p_raw | p_holm | rank-biserial | dz |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['comparison']} | {r['family']} | {r['n_pairs']} | {r['diff_mean']} "
                  f"| {r['diff_ci95']} | {r['improvement']} {r['improvement_ci95']} "
                  f"| {r['p_raw']} | {r['p_holm']} | {r['rank_biserial']} | {r['cohens_dz']} |")
    (out / "paired_inference.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"比较数: {len(rows)}；产物: {out}/paired_inference.csv, paired_inference.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
