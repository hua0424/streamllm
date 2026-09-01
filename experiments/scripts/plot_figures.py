"""生成论文第六章图 6-1 ~ 6-4。

数据一律从 experiments/results/ 或 experiments/sci34_supplement/results/ 的
审计结果读取，禁止硬编码实验数字（图内数值均为运行时计算/读取）。

用法（项目根目录）：
    uv run python -m experiments.scripts.plot_figures          # 中文版（学位论文）
    uv run python -m experiments.scripts.plot_figures --en     # 英文版（期刊投稿，文件名加 _en）

输出：paper2/figures/fig6_{1..4}[_en].svg + .pdf + .png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
SUPPLEMENT_RESULTS = ROOT / "experiments" / "sci34_supplement" / "results"
E3_ANALYSIS = (
    SUPPLEMENT_RESULTS
    / "e3"
    / "sci34_f11ccba_20260901_e3"
    / "analysis_metric_specific_eligibility_v1.json"
)
A1_ANALYSIS = (
    SUPPLEMENT_RESULTS
    / "a1"
    / "sci34_f11ccba_20260901_a1"
    / "analysis.json"
)
FIGDIR = ROOT / "paper2" / "figures"

EN = "--en" in sys.argv
SUFFIX = "_en" if EN else ""

# ---- 标签表：中/英一份数据两套文字 ----
_ZH = {
    "f1_xlabel": "共同目标与检测器",
    "f1_ylabel": "引用率差值（generation - playback，百分点）",
    "f1_labels": ["片段目标\n词面检测", "片段目标\nLLM 裁判", "近似目标\n词面检测", "近似目标\nLLM 裁判"],
    "f1_note": "对话聚类 bootstrap 95% CI；全部跨越零线\nplayback 条件本地完整未听文本 400/400 为空（构造检查）",
    "f2_knee": "候选工作区间\nθ∈[0.85, 0.97]",
    "f2_leg_pts": "推测工作点",
    "f2_leg_sent": "永不推测（保守极限）",
    "f2_sent": "永不推测",
    "f2_surv": "存活",
    "f2_xlabel": "推测浪费率 ρ（%）",
    "f2_ylabel": "TTFT$_{eff}$（ms）",
    "f3_x1": "TTFT（ms，实测）",
    "f3_x2": "mouth-to-ear（ms，建模值：首片段就绪 + TTS 首块，3090 实测画像）",
    "f3_leg_tts": "其中：TTS 首块合成延迟",
    "f3_note": "（TTS 首块 {tts:.0f} ms）",
    "f4_leg_pre": "重新预填充（放弃 KV 复用）",
    "f4_leg_recover": "联合 crop + 角色恢复（单次同步窗口）",
    "f4_leg_crop": "KV 裁剪操作（crop-only）",
    "f4_note": "误差线为四分位距；每点 50 次正式重复",
    "f4_xlabel": "上下文长度（token）",
    "f4_ylabel": "延迟（ms，median）",
    "f4_iqr_ylabel": "IQR（ms）",
}
_EN = {
    "f1_xlabel": "Shared target and detector",
    "f1_ylabel": "Reference-rate difference (generation − playback, pp)",
    "f1_labels": ["fragment\nlexical", "fragment\nLLM judge", "proxy\nlexical", "proxy\nLLM judge"],
    "f1_note": "Dialogue-cluster bootstrap 95% CIs; all cross zero\nLocal complete-unheard text is empty in 400/400 playback cases (construction check)",
    "f2_knee": "candidate region\nθ∈[0.85, 0.97]",
    "f2_leg_pts": "speculative operating points",
    "f2_leg_sent": "never speculate (conservative limit)",
    "f2_sent": "never speculate",
    "f2_surv": "surv. ",
    "f2_xlabel": "Speculation waste rate ρ (%)",
    "f2_ylabel": "TTFT$_{eff}$ (ms)",
    "f3_x1": "TTFT (ms, measured)",
    "f3_x2": "mouth-to-ear (ms, modeled)",
    "f3_leg_tts": "TTS first-chunk synthesis latency",
    "f3_note": "(TTS first chunk {tts:.0f} ms)",
    "f4_leg_pre": "re-prefill (no KV reuse)",
    "f4_leg_recover": "joint crop + role recovery (one synchronized window)",
    "f4_leg_crop": "KV crop operation only",
    "f4_note": "Error bars show IQR; 50 formal repeats per point",
    "f4_xlabel": "Context length (tokens)",
    "f4_ylabel": "Latency (ms, median)",
    "f4_iqr_ylabel": "IQR (ms)",
}
L = _EN if EN else _ZH

# ---- 全局风格：Okabe-Ito 彩色（色盲友好），线型/marker 保持灰度打印可区分 ----
if not EN:
    _CJK_CANDIDATES = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
    _available = {f.name for f in font_manager.fontManager.ttflist}
    _cjk = [f for f in _CJK_CANDIDATES if f in _available]
    if not _cjk:
        raise SystemExit("未找到中文字体，请安装 SimHei/微软雅黑，或用 --en 出英文版。")
    plt.rcParams["font.sans-serif"] = _cjk + ["DejaVu Sans"]
plt.rcParams.update(
    {
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "axes.axisbelow": True,
        "legend.framealpha": 0.9,
        "pdf.fonttype": 42,
    }
)

C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERMI = "#D55E00"
C_SKY = "#56B4E9"
G_DARK = "#333333"
G_MID = "#888888"


def _load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def _save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    stem = stem + SUFFIX
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 200})):
        path = FIGDIR / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kw)
        if ext == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"[saved] {FIGDIR / stem}.svg/.pdf/.png")


# ---------------------------------------------------------------- 图 6-1
def fig6_1():
    with open(E3_ANALYSIS, encoding="utf-8") as f:
        data = json.load(f)

    metric_names = ["rule_fragment", "judge_fragment", "rule_proxy", "judge_proxy"]
    expected_pairs = [297, 297, 380, 380]
    effects = []
    lows = []
    highs = []
    for name, expected in zip(metric_names, expected_pairs):
        metric = data["metrics"][name]
        bootstrap = metric["dialogue_cluster_bootstrap"]
        pairs = metric["mcnemar"]["pairs"]
        assert pairs == expected, f"{name}: expected {expected} eligible pairs, got {pairs}"
        effect = 100.0 * bootstrap["generation_minus_playback"]
        low, high = [100.0 * value for value in bootstrap["difference_95_ci"]]
        effects.append(effect)
        lows.append(low)
        highs.append(high)
    assert all(low <= 0 <= high for low, high in zip(lows, highs))
    assert data["construction_checks"]["playback_local_unheard_empty"]
    print(f"[fig6-1] fixed-trajectory effects (pp): {effects}; all cluster CIs cross zero ✓")

    y = list(range(len(metric_names)))
    colors = [C_SKY, C_ORANGE, C_BLUE, C_VERMI]
    xerr = [
        [effect - low for effect, low in zip(effects, lows)],
        [high - effect for effect, high in zip(effects, highs)],
    ]
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.axvline(0, color=G_DARK, lw=1.0, zorder=1)
    for index, (effect, color) in enumerate(zip(effects, colors)):
        ax.errorbar(
            effect,
            index,
            xerr=[[xerr[0][index]], [xerr[1][index]]],
            fmt="o",
            color=color,
            ecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.5,
            capsize=4,
            lw=1.6,
            ms=7,
            zorder=3,
        )
        ax.annotate(
            f"{effect:+.1f} pp (n={expected_pairs[index]})",
            (effect, index),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=8.5,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(L["f1_labels"])
    ax.invert_yaxis()
    ax.set_xlabel(L["f1_ylabel"])
    ax.text(
        0.01,
        -0.25,
        L["f1_note"],
        transform=ax.transAxes,
        fontsize=8.3,
        va="top",
        color=G_DARK,
    )
    fig.subplots_adjust(bottom=0.29, left=0.20)
    _save(fig, "fig6_1")


# ---------------------------------------------------------------- 图 6-2
def fig6_2():
    curve = _load("paper2_reanalysis.json")["e2"]["curve"]
    curve = sorted(curve, key=lambda p: p["threshold"])
    assert len(curve) == 9 and all(p["n"] == 100 for p in curve)
    waste = [100 * p["spec_waste_rate"] for p in curve]
    ttft = [p["ttft_eff_ms"] for p in curve]
    surv = [100 * p["survived_rate"] for p in curve]
    thr = [p["threshold"] for p in curve]
    print(f"[fig6-2] {len(curve)} formal pts, waste {waste[0]:.1f}%→{waste[-1]:.1f}%")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    knee = [w for w, t in zip(waste, thr) if 0.85 <= t <= 0.97]
    ax.axvspan(min(knee), max(knee), color=C_SKY, alpha=0.15, zorder=0)
    ax.annotate(
        L["f2_knee"],
        ((min(knee) + max(knee)) / 2, max(ttft) * 0.82),
        ha="center", fontsize=9, color=G_DARK,
    )

    real = [i for i, t in enumerate(thr) if t <= 1.0]
    sent = [i for i, t in enumerate(thr) if t > 1.0]
    ax.plot(
        [waste[i] for i in real] + [waste[i] for i in sent],
        [ttft[i] for i in real] + [ttft[i] for i in sent],
        "-", color=C_BLUE, lw=1.8, zorder=2,
    )
    ax.plot(
        [waste[i] for i in real], [ttft[i] for i in real],
        "o", color=C_BLUE, ms=6.5, zorder=3, label=L["f2_leg_pts"],
    )
    ax.plot(
        [waste[i] for i in sent], [ttft[i] for i in sent],
        "s", mfc="white", mec=C_VERMI, ms=8, mew=1.8, zorder=3,
        label=L["f2_leg_sent"],
    )
    offset_spec = {
        0.0052: (0, 10, "center"),
        0.776: (5, 8, "left"),
        0.85: (-9, 6, "right"),
        0.92: (6, 4, "left"),
        0.9688: (8, 4, "left"),
        1.1: (10, -10, "left"),
    }
    leader_spec = {
        0.1979: (22, 4.5),
        0.3906: (19, 8),
        0.5833: (16, 11.5),
    }
    for w_, t_, th_, s_ in zip(waste, ttft, thr, surv):
        label = L["f2_sent"] if th_ > 1.0 else f"θ={th_:g}"
        text = f"{label}\n{L['f2_surv']}{s_:.0f}%"
        if th_ in leader_spec:
            ax.annotate(
                text, (w_, t_), xytext=leader_spec[th_], textcoords="data",
                fontsize=7.8, ha="left", color=G_DARK,
                arrowprops=dict(arrowstyle="-", color=G_MID, lw=0.7),
            )
        else:
            dx, dy, ha = offset_spec[th_]
            ax.annotate(
                text, (w_, t_), textcoords="offset points", xytext=(dx, dy),
                fontsize=7.8, ha=ha, color=G_DARK,
            )
    ax.set_xlabel(L["f2_xlabel"])
    ax.set_ylabel(L["f2_ylabel"])
    ax.set_xlim(left=-1.5)
    ax.set_ylim(-2, 55)
    ax.legend(loc="upper right")
    _save(fig, "fig6_2")


# ---------------------------------------------------------------- 图 6-3
def fig6_3():
    s = _load("exp1_latency.json")["summary"]
    tts_first = _load("cosyvoice_profile.json")["profile"]["first_chunk_latency_ms"]
    a_ttft, b_ttft = s["a_ttft_ms"], s["b_ttft_eff_ms"]
    a_m2e, b_m2e = s["a_mouth_to_ear_ms_modeled"], s["b_mouth_to_ear_ms_modeled"]
    print(f"[fig6-3] TTFT {a_ttft}/{b_ttft}  m2e {a_m2e}/{b_m2e}  tts {tts_first}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.0, 4.6), gridspec_kw={"height_ratios": [1, 1.3]}
    )
    ylabels = ["System A", "B-ours"]

    ax1.barh(
        ylabels[::-1], [b_ttft, a_ttft],
        color=[C_BLUE, C_ORANGE], edgecolor="black", lw=0.6, height=0.55,
    )
    for y, v in zip(ylabels[::-1], [b_ttft, a_ttft]):
        ax1.annotate(
            f"{v:g} ms", (v, y), textcoords="offset points", xytext=(5, 0),
            va="center", fontsize=9.5,
        )
    ax1.set_xlim(0, a_ttft * 1.25)
    ax1.set_xlabel(L["f3_x1"])

    ax2.barh(
        "System A", a_m2e, color=C_ORANGE, edgecolor="black", lw=0.6, height=0.55,
    )
    ax2.barh(
        "B-ours", tts_first, color=C_SKY, edgecolor="black", lw=0.6,
        height=0.55, label=L["f3_leg_tts"],
    )
    ax2.barh(
        "B-ours", b_m2e - tts_first, left=tts_first, color=C_BLUE,
        edgecolor="black", lw=0.6, height=0.55,
    )
    ax2.annotate(
        f"{a_m2e:.0f} ms", (a_m2e, "System A"),
        textcoords="offset points", xytext=(5, 0), va="center", fontsize=9.5,
    )
    note = L["f3_note"].format(tts=tts_first)
    ax2.annotate(
        f"{b_m2e:.0f} ms {note}", (b_m2e, "B-ours"),
        textcoords="offset points", xytext=(5, 0), va="center", fontsize=9.5,
    )
    ax2.set_xlim(0, a_m2e * 1.42)
    ax2.set_xlabel(L["f3_x2"])
    ax2.invert_yaxis()
    ax2.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "fig6_3")


# ---------------------------------------------------------------- 图 6-4
def fig6_4():
    with open(A1_ANALYSIS, encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    ctx = [row["target_length"] for row in rows]
    crop = [row["statistics"]["crop_only_ms"]["median"] for row in rows]
    joint = [row["statistics"]["crop_role_joint_ms"]["median"] for row in rows]
    joint_q1 = [row["statistics"]["crop_role_joint_ms"]["q1"] for row in rows]
    joint_q3 = [row["statistics"]["crop_role_joint_ms"]["q3"] for row in rows]
    reprefill = [row["statistics"]["reprefill_ms"]["median"] for row in rows]
    reprefill_q1 = [row["statistics"]["reprefill_ms"]["q1"] for row in rows]
    reprefill_q3 = [row["statistics"]["reprefill_ms"]["q3"] for row in rows]
    speedup = [row["speedup_reprefill_over_joint_median"] for row in rows]
    assert all(row["statistics"]["crop_role_joint_ms"]["n"] == 50 for row in rows)
    print(f"[fig6-4] ctx {ctx}  synchronized joint speedup {speedup}")

    joint_err = [
        [median - q1 for median, q1 in zip(joint, joint_q1)],
        [q3 - median for median, q3 in zip(joint, joint_q3)],
    ]
    reprefill_err = [
        [median - q1 for median, q1 in zip(reprefill, reprefill_q1)],
        [q3 - median for median, q3 in zip(reprefill, reprefill_q3)],
    ]
    fig, (ax, ax_iqr) = plt.subplots(
        2,
        1,
        figsize=(6.6, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08},
    )
    ax.errorbar(
        ctx,
        reprefill,
        yerr=reprefill_err,
        fmt="o-",
        color=C_VERMI,
        lw=1.8,
        ms=6,
        capsize=3,
        label=L["f4_leg_pre"],
    )
    ax.errorbar(
        ctx,
        joint,
        yerr=joint_err,
        fmt="s--",
        color=C_ORANGE,
        lw=1.6,
        ms=6,
        capsize=3,
        mfc="white",
        label=L["f4_leg_recover"],
    )
    ax.plot(
        ctx, crop, "^-", color=C_BLUE, lw=1.7, ms=6, mfc="white",
        label=L["f4_leg_crop"],
    )
    for x, y, sp in zip(ctx, joint, speedup):
        ax.annotate(
            f"{sp:.1f}×", (x, y), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=8.3,
        )
    joint_iqr = [high - low for low, high in zip(joint_q1, joint_q3)]
    reprefill_iqr = [high - low for low, high in zip(reprefill_q1, reprefill_q3)]
    ax_iqr.plot(ctx, joint_iqr, "s--", color=C_ORANGE, lw=1.5, ms=5, mfc="white")
    ax_iqr.plot(ctx, reprefill_iqr, "o-", color=C_VERMI, lw=1.5, ms=5)
    ax_iqr.set_ylabel(L["f4_iqr_ylabel"], fontsize=9)
    ax_iqr.text(
        0.99,
        0.93,
        L["f4_note"],
        transform=ax_iqr.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=max(reprefill) * 1.8)
    ax.set_ylabel(L["f4_ylabel"])
    ax.legend(loc="upper left", fontsize=8.1)
    ax_iqr.set_xscale("log")
    ax_iqr.set_xticks(ctx)
    ax_iqr.set_xticklabels([str(c) for c in ctx])
    ax_iqr.set_xlabel(L["f4_xlabel"])
    _save(fig, "fig6_4")


if __name__ == "__main__":
    fig6_1()
    fig6_2()
    fig6_3()
    fig6_4()
    print(("EN" if EN else "ZH"), "全部完成 →", FIGDIR)
