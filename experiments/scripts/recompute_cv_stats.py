"""W3：E1 三轮重复测量 CV 统一口径重算（ddof=1 全分布）。

对应 PRE-PAPER-AUDIT P0-3 与 v3.1 方案：
- CV_i = std(x_i1..3, ddof=1) / mean(x_i1..3)，百分比输出；
- 样本须恰 3 轮、同模式、无 error、TTFT 有效；
- 输出逐样本明细 + mean/median/P90（线性插值）/max + CV>5% 数量与比例；
- 三个输入文件显式路径 + SHA-256 落盘；不用 glob 猜最新。

用法：
  uv run python -m experiments.scripts.recompute_cv_stats \
      --inputs r1.json r2.json r3.json --out-dir <dir>
  uv run python -m experiments.scripts.recompute_cv_stats --self-test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REQUIRED_RECORD_FIELDS = ("sample_id", "mode", "ttft")
VALID_MODES = ("streaming", "non-streaming")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


CONFIG_COMPARE_KEYS = ("asr_model", "llm_model", "asr_device", "llm_device",
                       "chunk_duration_ms", "max_tokens", "prefix_segments",
                       "suffix_segments", "recognition_threshold", "append_silence_ms",
                       "sample_list")


def load_records(path: str) -> tuple[list[dict], dict]:
    """加载 exp1_results_*.json，返回 (记录列表, config)。兼容 list / {"results": [...]}。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        recs, cfg = data, {}
    elif isinstance(data, dict):
        recs = data.get("results", list(data.values()))
        cfg = data.get("config", {}) if isinstance(data.get("config", {}), dict) else {}
    else:
        raise ValueError(f"无法识别的结果文件结构: {path}")
    if isinstance(recs, dict):
        recs = list(recs.values())
    for i, r in enumerate(recs):
        missing = [k for k in REQUIRED_RECORD_FIELDS if k not in r]
        if missing:
            raise ValueError(f"{path} 第 {i} 条记录缺字段 {missing}")
    # 文件内主键唯一（评审 W3 强化）
    keys = [(str(r["sample_id"]), str(r["mode"])) for r in recs]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"{path} 存在重复主键 (sample_id, mode)（停止）")
    return recs, cfg


def collect_cv_rows(paths: list[str]) -> list[dict]:
    """按 (sample_id, mode) 聚合三轮 TTFT，计算逐样本 CV（ddof=1）。

    冻结协议（评审 W3）：三个文件键集合完全一致、每文件每键恰一条（load 内查重）、
    关键 config 字段跨文件一致；满足后每键恰三轮有效记录才纳入。
    """
    per_file_keys = []
    cfg_refs = []
    loaded = []
    for p in paths:
        recs, cfg = load_records(p)
        loaded.append(recs)
        per_file_keys.append({(str(r["sample_id"]), str(r["mode"])) for r in recs})
        cfg_refs.append({k: cfg.get(k) for k in CONFIG_COMPARE_KEYS})
    if per_file_keys[0] != per_file_keys[1] or per_file_keys[0] != per_file_keys[2]:
        raise SystemExit("三个输入文件的 (sample_id, mode) 键集合不一致（停止）")
    if not (cfg_refs[0] == cfg_refs[1] == cfg_refs[2]):
        raise SystemExit(f"三个输入文件关键 config 字段不一致（停止）: {cfg_refs}")

    per: dict[tuple[str, str], list[float]] = {}
    for recs in loaded:
        for r in recs:
            if r.get("error"):
                continue
            try:
                ttft = float(r["ttft"])
            except (TypeError, ValueError):
                continue
            if not np.isfinite(ttft) or ttft <= 0:
                continue
            key = (str(r["sample_id"]), str(r["mode"]))
            per.setdefault(key, []).append(ttft)

    rows = []
    for (sid, mode), vals in sorted(per.items()):
        if len(vals) != 3:
            continue  # 恰 3 轮有效记录才纳入
        arr = np.asarray(vals, dtype=float)
        mean = arr.mean()
        cv = float(arr.std(ddof=1) / mean * 100.0) if mean > 0 else float("nan")
        rows.append({
            "sample_id": sid, "mode": mode,
            "ttft_r1": f"{arr[0]:.3f}", "ttft_r2": f"{arr[1]:.3f}", "ttft_r3": f"{arr[2]:.3f}",
            "ttft_mean": f"{mean:.3f}",
            "ttft_std_ddof1": f"{arr.std(ddof=1):.3f}",
            "cv_pct": f"{cv:.4f}",
        })
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """按 mode 汇总 mean/median/P90（线性插值）/max CV 与 >5% 比例。"""
    out = []
    for mode in VALID_MODES:
        cvs = np.array([float(r["cv_pct"]) for r in rows if r["mode"] == mode], dtype=float)
        if len(cvs) == 0:
            continue
        n = len(cvs)
        n_gt5 = int((cvs > 5.0).sum())
        out.append({
            "mode": mode,
            "n_samples": n,
            "cv_mean_pct": f"{cvs.mean():.4f}",
            "cv_median_pct": f"{float(np.median(cvs)):.4f}",
            "cv_p90_pct": f"{float(np.percentile(cvs, 90)):.4f}",  # numpy 默认线性插值
            "cv_max_pct": f"{cvs.max():.4f}",
            "n_cv_gt5": n_gt5,
            "pct_cv_gt5": f"{n_gt5 / n * 100:.2f}",
            "std_definition": "per-sample std over 3 rounds, ddof=1",
        })
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"无数据可写: {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_md(summary: list[dict], inputs: list[dict], path: Path) -> None:
    lines = [
        "# E1 三轮重复测量 CV 汇总（ddof=1 统一口径）",
        "",
        "- 口径：CV_i = std(3 轮, ddof=1) / mean(3 轮)，百分比；样本须恰 3 轮、同模式、无 error；",
        "- P90 为 numpy 默认线性插值；",
        "- 输入文件 SHA-256：",
    ]
    for it in inputs:
        lines.append(f"  - `{it['path']}` : `{it['sha256']}`")
    lines += ["", "| mode | n | mean CV% | median CV% | P90 CV% | max CV% | CV>5% n(%) |", "|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(
            f"| {s['mode']} | {s['n_samples']} | {s['cv_mean_pct']} | {s['cv_median_pct']} "
            f"| {s['cv_p90_pct']} | {s['cv_max_pct']} | {s['n_cv_gt5']} ({s['pct_cv_gt5']}%) |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- self-test

def _self_test() -> int:
    import tempfile

    fails = []

    def check(name: str, cond: bool, detail: str = ""):
        if not cond:
            fails.append(f"{name}: {detail}")

    # 构造三轮合成数据：streaming s1=[100,110,105]、s2 仅 2 轮有效（应排除）、
    # non-streaming s1=[200,200,200]（CV=0）、s2 含 error 轮（应排除）
    rounds = []
    for ridx, t in enumerate([0.0, 1.0, 2.0]):
        recs = [
            {"sample_id": "s1", "mode": "streaming", "ttft": [100.0, 110.0, 105.0][ridx]},
            {"sample_id": "s2", "mode": "streaming", "ttft": 500.0, "error": "boom" if ridx == 2 else ""},
            {"sample_id": "s1", "mode": "non-streaming", "ttft": 200.0},
            {"sample_id": "s2", "mode": "non-streaming", "ttft": [300.0, 330.0, 315.0][ridx]},
        ]
        rounds.append(recs)

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, recs in enumerate(rounds):
            p = Path(td) / f"r{i}.json"
            p.write_text(json.dumps({"results": recs}), encoding="utf-8")
            paths.append(str(p))
        rows = collect_cv_rows(paths)

    keys = {(r["sample_id"], r["mode"]) for r in rows}
    check("恰3轮过滤", keys == {("s1", "streaming"), ("s1", "non-streaming"), ("s2", "non-streaming")},
          str(keys))

    r_s1 = next(r for r in rows if r["sample_id"] == "s1" and r["mode"] == "streaming")
    arr = np.array([100.0, 110.0, 105.0])
    exp_cv = arr.std(ddof=1) / arr.mean() * 100
    check("CV 数值(ddof=1)", abs(float(r_s1["cv_pct"]) - exp_cv) < 1e-3,
          f"{r_s1['cv_pct']} vs {exp_cv}")
    r_ns1 = next(r for r in rows if r["sample_id"] == "s1" and r["mode"] == "non-streaming")
    check("零方差 CV=0", float(r_ns1["cv_pct"]) == 0.0, r_ns1["cv_pct"])

    summary = summarize(rows)
    st = next(s for s in summary if s["mode"] == "streaming")
    check("汇总 n=1", st["n_samples"] == 1, str(st))
    check(">5% 计数", st["n_cv_gt5"] == (1 if exp_cv > 5 else 0), str(st))
    ns = next(s for s in summary if s["mode"] == "non-streaming")
    check("non-streaming n=2", ns["n_samples"] == 2, str(ns))
    check("std 定义声明", ns["std_definition"].startswith("per-sample std"), ns["std_definition"])

    # 与 E1 真实数据一致的锚点量级自检（构造 50 样本均值应重现 5.19%）
    rng = np.random.default_rng(0)
    base = rng.normal(1000, 100, 50)
    recs3 = []
    for ridx in range(3):
        recs3.append([{"sample_id": f"s{i}", "mode": "streaming",
                       "ttft": float(base[i] * (1 + rng.normal(0, 0.045)))} for i in range(50)])
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, recs in enumerate(recs3):
            p = Path(td) / f"r{i}.json"
            p.write_text(json.dumps(recs), encoding="utf-8")
            paths.append(str(p))
        rows2 = collect_cv_rows(paths)
    check("50 样本全保留", len(rows2) == 50, str(len(rows2)))
    s2 = summarize(rows2)[0]
    check("P90>=median", float(s2["cv_p90_pct"]) >= float(s2["cv_median_pct"]), str(s2))

    # 负向（评审 W3）：文件内重复主键 / 键集合不一致 / config 不一致
    with tempfile.TemporaryDirectory() as td:
        def w(name, recs, cfg=None):
            p = Path(td) / name
            p.write_text(json.dumps({"results": recs, "config": cfg or {}}), encoding="utf-8")
            return str(p)
        base1 = [{"sample_id": "s1", "mode": "streaming", "ttft": 1.0}]
        p1 = w("a.json", base1, {"asr_model": "turbo"})
        pdup = w("b.json", base1 + base1, {"asr_model": "turbo"})
        try:
            collect_cv_rows([p1, pdup, p1])
            check("重复主键退出", False)
        except SystemExit:
            check("重复主键退出", True)
        p2 = w("c.json", [{"sample_id": "s2", "mode": "streaming", "ttft": 1.0}],
               {"asr_model": "turbo"})
        try:
            collect_cv_rows([p1, p1, p2])
            check("键集合不一致退出", False)
        except SystemExit:
            check("键集合不一致退出", True)
        p3 = w("d.json", base1, {"asr_model": "base"})
        try:
            collect_cv_rows([p1, p1, p3])
            check("config 不一致退出", False)
        except SystemExit:
            check("config 不一致退出", True)

    if fails:
        for f in fails:
            print(f"FAIL {f}")
        return 1
    print("self-test: ALL PASS")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs=3, metavar=("R1", "R2", "R3"),
                    help="三轮 exp1_results_*.json 显式路径")
    ap.add_argument("--out-dir", help="输出目录")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not args.inputs or not args.out_dir:
        ap.error("正式模式需要 --inputs R1 R2 R3 与 --out-dir")

    for p in args.inputs:
        if not Path(p).is_file():
            print(f"FATAL: 输入不存在: {p}", file=sys.stderr)
            return 1

    rows = collect_cv_rows(list(args.inputs))
    summary = summarize(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "repeat_cv_persample.csv")
    write_csv(summary, out / "repeat_cv_summary.csv")
    inputs_meta = [{"path": p, "sha256": sha256_file(p)} for p in args.inputs]
    write_md(summary, inputs_meta, out / "repeat_cv_summary.md")
    print(f"per-sample rows: {len(rows)}")
    for s in summary:
        print(f"{s['mode']}: n={s['n_samples']} mean={s['cv_mean_pct']}% median={s['cv_median_pct']}% "
              f"P90={s['cv_p90_pct']}% max={s['cv_max_pct']}% >5%: {s['n_cv_gt5']}/{s['n_samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
