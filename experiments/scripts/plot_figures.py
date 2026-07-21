"""生成论文第六章图 6-1 ~ 6-4。

数据一律从 experiments/results/*.json 读取，禁止硬编码实验数字
（图内出现的数值均为运行时计算/读取，保证图文一致可复现）。

用法（项目根目录）：
    uv run python -m experiments.scripts.plot_figures

输出：paper2/figures/fig6_{1..4}.pdf + .png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"
FIGDIR = ROOT / "paper2" / "figures"

# ---- 全局风格：学位论文黑白友好 ----
_CJK_CANDIDATES = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC"]
_available = {f.name for f in font_manager.fontManager.ttflist}
_cjk = [f for f in _CJK_CANDIDATES if f in _available]
HAS_CJK = bool(_cjk)
if HAS_CJK:
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
        "pdf.fonttype": 42,  # TrueType 嵌入，学校打印/送审友好
    }
)

# 黑白友好灰阶（配合 hatch / 线型 / marker 区分，不依赖颜色）
G_DARK = "#333333"
G_MID = "#888888"
G_LIGHT = "#cccccc"


def _load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def _save(fig, stem):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(FIGDIR / f"{stem}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"[saved] {FIGDIR / stem}.pdf/.png")


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

    # 自检（handoff 实测值）
    expect = {"gen_rule": [85.4, 48.5, 20.4, 49.5], "gen_judge": [4.9, 1.0, 1.9, 2.9]}
    for key, got in (("gen_rule", gen_rule), ("gen_judge", gen_judge)):
        for e, g in zip(expect[key], got):
            assert abs(e - g) < 0.15, f"self-check fail {key}: expect {e} got {g:.2f}"
    assert max(ours_rule + ours_judge) == 0.0, "B-ours 应为构造性零"
    print(f"[fig6-1] B-gen 规则 {gen_rule} 裁判 {gen_judge}  B-ours 全零 ✓")

    x = range(len(fractions))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b1 = ax.bar(
        [i - w for i in x], gen_rule, w,
        color=G_LIGHT, edgecolor="black", hatch="//", label="B-gen（规则口径）",
    )
    b2 = ax.bar(
        x, gen_judge, w,
        color=G_MID, edgecolor="black", label="B-gen（裁判口径）",
    )
    b3 = ax.bar(
        [i + w for i in x], ours_rule, w,
        color="white", edgecolor="black", hatch="xx", label="B-ours（两口径）",
    )
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(
                f"{r.get_height():.1f}",
                (r.get_x() + r.get_width() / 2, r.get_height()),
                ha="center", va="bottom", fontsize=8.5,
            )
    for i in x:  # B-ours 贴地零柱：显式标 0.0%
        ax.annotate(
            "0.0", (i + w, 0), ha="center", va="bottom", fontsize=8.5,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(["25%", "50%", "75%", "干净边界"])
    ax.set_xlabel("打断注入位置（播放比例）")
    ax.set_ylabel("未听内容引用率（%）")
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
    print(f"[fig6-2] {len(curve)} 点, waste {waste[0]:.1f}%→{waste[-1]:.1f}%")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    # 拐点区 θ∈[0.85, 0.97]：以对应浪费率区间做浅灰底
    knee = [w for w, t in zip(waste, thr) if 0.85 <= t <= 0.97]
    ax.axvspan(min(knee), max(knee), color="0.92", zorder=0)
    ax.annotate(
        "拐点区\nθ∈[0.85, 0.97]",
        ((min(knee) + max(knee)) / 2, max(ttft) * 0.82),
        ha="center", fontsize=9, color=G_DARK,
    )

    real = [i for i, t in enumerate(thr) if t <= 1.0]
    sent = [i for i, t in enumerate(thr) if t > 1.0]  # θ=1.1 永不推测哨兵
    ax.plot(
        [waste[i] for i in real] + [waste[i] for i in sent],
        [ttft[i] for i in real] + [ttft[i] for i in sent],
        "-", color="black", lw=1.6, zorder=2,
    )
    ax.plot(
        [waste[i] for i in real], [ttft[i] for i in real],
        "o", color="black", ms=6, zorder=3, label="推测工作点",
    )
    ax.plot(
        [waste[i] for i in sent], [ttft[i] for i in sent],
        "s", mfc="white", mec="black", ms=8, mew=1.5, zorder=3,
        label="永不推测（保守极限）",
    )
    # 每点标注：低浪费区点密集，密集三点用引线甩到右侧空白区
    offset_spec = {  # thr: (dx, dy, ha) offset-points 方式
        0.0052: (0, 10, "center"),
        0.776: (5, 8, "left"),
        0.85: (-9, 6, "right"),
        0.92: (6, 4, "left"),
        0.9688: (8, 4, "left"),
        1.1: (10, -10, "left"),
    }
    leader_spec = {  # thr: (x, y) data 坐标，带引线
        0.1979: (22, 4.5),
        0.3906: (19, 8),
        0.5833: (16, 11.5),
    }
    for w_, t_, th_, s_ in zip(waste, ttft, thr, surv):
        label = "永不推测" if th_ > 1.0 else f"θ={th_:g}"
        text = f"{label}\n存活{s_:.0f}%"
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
    ax.set_xlabel("推测浪费率 ρ（%）")
    ax.set_ylabel("TTFT$_{eff}$（ms）")
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
    print(f"[fig6-3] TTFT {a_ttft}/{b_ttft}  m2e {a_m2e}/{b_m2e}  tts首块 {tts_first}")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.0, 4.6), gridspec_kw={"height_ratios": [1, 1.3]}
    )
    ylabels = ["System A", "B-ours"]

    # 上：TTFT（实测）
    ax1.barh(
        ylabels[::-1], [b_ttft, a_ttft],
        color=[G_MID, G_LIGHT], edgecolor="black", height=0.55,
    )
    for y, v in zip(ylabels[::-1], [b_ttft, a_ttft]):
        ax1.annotate(
            f"{v:g} ms", (v, y), textcoords="offset points", xytext=(5, 0),
            va="center", fontsize=9.5,
        )
    ax1.set_xlim(0, a_ttft * 1.25)
    ax1.set_xlabel("TTFT（ms，实测）")

    # 下：mouth-to-ear（建模值），B-ours 拆出 TTS 首块延迟
    ax2.barh(
        "System A", a_m2e, color=G_LIGHT, edgecolor="black", height=0.55,
    )
    ax2.barh(
        "B-ours", tts_first, color="white", edgecolor="black", hatch="//",
        height=0.55, label="其中：TTS 首块合成延迟",
    )
    ax2.barh(
        "B-ours", b_m2e - tts_first, left=tts_first, color=G_MID,
        edgecolor="black", height=0.55,
    )
    ax2.annotate(
        f"{a_m2e:.0f} ms", (a_m2e, "System A"),
        textcoords="offset points", xytext=(5, 0), va="center", fontsize=9.5,
    )
    ax2.annotate(
        f"{b_m2e:.0f} ms（TTS 首块 {tts_first:.0f} ms）", (b_m2e, "B-ours"),
        textcoords="offset points", xytext=(5, 0), va="center", fontsize=9.5,
    )
    ax2.set_xlim(0, a_m2e * 1.42)
    ax2.set_xlabel("mouth-to-ear（ms，建模值：首片段就绪 + TTS 首块，3090 实测画像）")
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
        ctx, reprefill, "o-", color="black", lw=1.6, ms=6,
        label="重新 prefill（放弃 KV 复用）",
    )
    ax.plot(
        ctx, rebuild, "s--", color=G_MID, lw=1.4, ms=6, mfc="white",
        label="角色重建（非关键路径，可延迟执行）",
    )
    ax.plot(
        ctx, crop, "^-", color="black", lw=2.2, ms=7, mfc="white",
        label="反查 + 截断（关键路径）",
    )
    for x, y, sp in zip(ctx, reprefill, speedup):
        ax.annotate(
            f"{sp:g}×", (x, y), textcoords="offset points", xytext=(0, 7),
            ha="center", fontsize=8.5,
        )
    ax.annotate(
        "亚毫秒、与上下文长度无关",
        (ctx[2], crop[2]), textcoords="offset points", xytext=(0, 10),
        ha="center", fontsize=9, color=G_DARK,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ctx)
    ax.set_xticklabels([str(c) for c in ctx])
    ax.set_xlabel("上下文长度（token）")
    ax.set_ylabel("延迟（ms，median）")
    ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, "fig6_4")


if __name__ == "__main__":
    if not HAS_CJK:
        raise SystemExit(
            "未找到中文字体（SimHei/微软雅黑/Noto CJK），请安装后重跑，"
            "或按 handoff 约定把标签改英文。"
        )
    fig6_1()
    fig6_2()
    fig6_3()
    fig6_4()
    print("全部完成 →", FIGDIR)
