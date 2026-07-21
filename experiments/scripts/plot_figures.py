"""生成论文第六章图 6-1 ~ 6-4。

数据一律从 experiments/results/*.json 读取，禁止硬编码实验数字
（图内出现的数值均为运行时计算/读取，保证图文一致可复现）。

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
FIGDIR = ROOT / "paper2" / "figures"

EN = "--en" in sys.argv
SUFFIX = "_en" if EN else ""

# ---- 标签表：中/英一份数据两套文字 ----
_ZH = {
    "f1_xlabel": "打断注入位置（播放比例）",
    "f1_ylabel": "未听内容引用率（%）",
    "f1_xticks": ["25%", "50%", "75%", "干净边界"],
    "f1_leg_rule": "B-gen（规则口径）",
    "f1_leg_judge": "B-gen（裁判口径）",
    "f1_leg_ours": "B-ours（两口径）",
    "f2_knee": "拐点区\nθ∈[0.85, 0.97]",
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
    "f4_leg_pre": "重新 prefill（放弃 KV 复用）",
    "f4_leg_reb": "角色重建（非关键路径，可延迟执行）",
    "f4_leg_crop": "反查 + 截断（关键路径）",
    "f4_note": "亚毫秒、与上下文长度无关",
    "f4_xlabel": "上下文长度（token）",
    "f4_ylabel": "延迟（ms，median）",
}
_EN = {
    "f1_xlabel": "Barge-in injection position (playback fraction)",
    "f1_ylabel": "Unheard-content reference rate (%)",
    "f1_xticks": ["25%", "50%", "75%", "clean boundary"],
    "f1_leg_rule": "B-gen (rule detector)",
    "f1_leg_judge": "B-gen (LLM judge)",
    "f1_leg_ours": "B-ours (both)",
    "f2_knee": "knee region\nθ∈[0.85, 0.97]",
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
    "f4_leg_reb": "role rebuild (off critical path, deferrable)",
    "f4_leg_crop": "lookup + crop (critical path)",
    "f4_note": "sub-ms, independent of context length",
    "f4_xlabel": "Context length (tokens)",
    "f4_ylabel": "Latency (ms, median)",
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
        fig.savefig(FIGDIR / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"[saved] {FIGDIR / stem}.svg/.pdf/.png")


# ---------------------------------------------------------------- 图 6-1
def fig6_1():
    data = _load("exp3_consistency_judged.json")
    fractions = ["0.25", "0.5", "0.75", "boundary"]

    def rate(cond, frac, field):
        rec = [
            r
            for r in data["records"]
            if r["condition"] == cond and str(r["fraction"]) == frac
        ]
        assert rec, f"no records for {cond}/{frac}"
        return 100.0 * sum(bool(r[field]) for r in rec) / len(rec)

    gen_rule = [rate("generation", f, "referenced_unheard") for f in fractions]
    gen_judge = [rate("generation", f, "judge_referenced_unheard") for f in fractions]
    ours_rule = [rate("playback", f, "referenced_unheard") for f in fractions]
    ours_judge = [rate("playback", f, "judge_referenced_unheard") for f in fractions]

    expect = {"gen_rule": [85.4, 48.5, 20.4, 49.5], "gen_judge": [4.9, 1.0, 1.9, 2.9]}
    for key, got in (("gen_rule", gen_rule), ("gen_judge", gen_judge)):
        for e, g in zip(expect[key], got):
            assert abs(e - g) < 0.15, f"self-check fail {key}: expect {e} got {g:.2f}"
    assert max(ours_rule + ours_judge) == 0.0, "B-ours 应为构造性零"
    print(f"[fig6-1] B-gen rule {gen_rule} judge {gen_judge}  B-ours all-zero ✓")

    x = range(len(fractions))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b1 = ax.bar(
        [i - w for i in x], gen_rule, w,
        color=C_SKY, edgecolor="black", lw=0.6, label=L["f1_leg_rule"],
    )
    b2 = ax.bar(
        x, gen_judge, w,
        color=C_ORANGE, edgecolor="black", lw=0.6, label=L["f1_leg_judge"],
    )
    ax.bar(
        [i + w for i in x], ours_rule, w,
        color=C_GREEN, edgecolor="black", lw=0.6, label=L["f1_leg_ours"],
    )
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(
                f"{r.get_height():.1f}",
                (r.get_x() + r.get_width() / 2, r.get_height()),
                ha="center", va="bottom", fontsize=8.5,
            )
    for i in x:
        ax.annotate("0.0", (i + w, 0), ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(L["f1_xticks"])
    ax.set_xlabel(L["f1_xlabel"])
    ax.set_ylabel(L["f1_ylabel"])
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    _save(fig, "fig6_1")


# ---------------------------------------------------------------- 图 6-2
def fig6_2():
    curve = _load("exp2_tradeoff.json")["curve"]
    curve = sorted(curve, key=lambda p: p["threshold"])
    waste = [100 * p["spec_waste_rate"] for p in curve]
    ttft = [p["ttft_eff_ms"] for p in curve]
    surv = [100 * p["survived_rate"] for p in curve]
    thr = [p["threshold"] for p in curve]
    print(f"[fig6-2] {len(curve)} pts, waste {waste[0]:.1f}%→{waste[-1]:.1f}%")

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
    rows = _load("exp_a1_kvreuse.json")["results"]
    ctx = [r["ctx_len"] for r in rows]
    crop = [r["crop_only_ms"] for r in rows]
    rebuild = [r["role_rebuild_ms"] for r in rows]
    reprefill = [r["reprefill_ms"] for r in rows]
    speedup = [r["speedup"] for r in rows]
    print(f"[fig6-4] ctx {ctx}  speedup {speedup}")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(
        ctx, reprefill, "o-", color=C_VERMI, lw=1.8, ms=6, label=L["f4_leg_pre"],
    )
    ax.plot(
        ctx, rebuild, "s--", color=C_ORANGE, lw=1.5, ms=6, mfc="white",
        label=L["f4_leg_reb"],
    )
    ax.plot(
        ctx, crop, "^-", color=C_BLUE, lw=2.2, ms=7, mfc="white",
        label=L["f4_leg_crop"],
    )
    for x, y, sp in zip(ctx, reprefill, speedup):
        ax.annotate(
            f"{sp:g}×", (x, y), textcoords="offset points", xytext=(0, 7),
            ha="center", fontsize=8.5,
        )
    ax.annotate(
        L["f4_note"],
        (ctx[2], crop[2]), textcoords="offset points", xytext=(0, 10),
        ha="center", fontsize=9, color=G_DARK,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(top=max(reprefill) * 2.5)
    ax.set_xticks(ctx)
    ax.set_xticklabels([str(c) for c in ctx])
    ax.set_xlabel(L["f4_xlabel"])
    ax.set_ylabel(L["f4_ylabel"])
    ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, "fig6_4")


if __name__ == "__main__":
    fig6_1()
    fig6_2()
    fig6_3()
    fig6_4()
    print(("EN" if EN else "ZH"), "全部完成 →", FIGDIR)
