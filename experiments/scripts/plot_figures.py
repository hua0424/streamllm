"""生成论文第六章图 6-1 ~ 6-4。

数据一律从 experiments/results/ 或 experiments/sci34_supplement/results/ 的
审计结果读取，禁止硬编码实验数字（图内数值均为运行时计算/读取）。
图 6-2/6-3 自 D-017 起改读确认性 campaign e1e2c 的 analysis_v1.json
（C-E1/C-E2 双口径：实际墙钟主指标 + oracle TTFT_eff 时延乐观下界）。

用法（项目根目录）：
    uv run python -m experiments.scripts.plot_figures          # 中文版（学位论文），全部四图
    uv run python -m experiments.scripts.plot_figures --en     # 英文版（期刊投稿，文件名加 _en）
    uv run python -m experiments.scripts.plot_figures fig6_2 fig6_3   # 只重画指定图（其余文件不动）

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
E1E2C_ANALYSIS = (
    SUPPLEMENT_RESULTS
    / "e1e2_confirmatory"
    / "e1e2c_b8c758b_20260901T173306Z"
    / "analysis_v1.json"
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
    "f2_title_a": "（a）oracle 口径：TTFT$_{eff}$ 时延乐观下界",
    "f2_title_b": "（b）实际墙钟主指标",
    "f2_xlabel": "推测浪费率 ρ（%，pooled）",
    "f2_ylabel_a": "TTFT$_{eff}$ 均值（ms）",
    "f2_ylabel_b": "最后段到达→首 token\n就绪均值（ms）",
    "f2_leg_pts": "推测工作点（θ 扫描）",
    "f2_leg_sent": "永不推测（保守极限）",
    "f2_sent": "永不推测",
    "f2_hl": "θ=0.92\n确认工作点",
    "f2_flat": "全条件平坦 {lo:.1f}–{hi:.1f} ms",
    "f2_note": "（a）为 oracle 时延乐观下界（推测收益上界），（b）为实际墙钟主指标（最后段到达→首 token 就绪）；\n"
               "同批九条件（8 阈值 + 永不推测），每点 n={n}（{sess} session × {dial} 对话）。",
    "f3_title_a": "（a）实际墙钟：最后段到达→首 token 就绪",
    "f3_title_b": "（b）oracle TTFT$_{eff}$：时延乐观下界",
    "f3_ylabel": "延迟（ms）",
    "f3_label_a": "系统 A\n一次性全量预填充",
    "f3_label_b": "B-ours\nθ=0.92 推测",
    "f3_bar_note": "均值 {mean:.2f}\n中位 {med:.2f}",
    "f3_diff": "配对差 A-B：{diff:+.2f} ms\n95% CI [{lo:.2f}, {hi:.2f}]",
    "f3_note": "同批配对（n={n}）：（a）为实际墙钟主指标（B 更慢）；（b）为同步 oracle 端点下的乐观下界（推测收益上界，B 更低）。\n"
               "两种口径度量不同对象，不可相加、不可混称。",
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
    "f2_title_a": "(a) Oracle view: TTFT$_{eff}$ optimistic latency lower bound",
    "f2_title_b": "(b) Wall-clock primary metric",
    "f2_xlabel": "Speculation waste rate ρ (%, pooled)",
    "f2_ylabel_a": "Mean TTFT$_{eff}$ (ms)",
    "f2_ylabel_b": "Arrival → first token ready\nmean (ms)",
    "f2_leg_pts": "speculative working points (θ sweep)",
    "f2_leg_sent": "never speculate (conservative limit)",
    "f2_sent": "never",
    "f2_hl": "θ=0.92\nconfirmatory point",
    "f2_flat": "flat across conditions, {lo:.1f}–{hi:.1f} ms",
    "f2_note": "(a) Oracle latency optimistic lower bound (= upper bound of speculation benefit); "
               "(b) actual wall-clock primary metric (last-segment arrival → first token ready).\n"
               "Nine conditions (8 thresholds + never), n={n} each ({sess} sessions × {dial} dialogues).",
    "f3_title_a": "(a) Wall clock: arrival → first token ready",
    "f3_title_b": "(b) Oracle TTFT$_{eff}$: optimistic latency lower bound",
    "f3_ylabel": "Latency (ms)",
    "f3_label_a": "System A\none-shot full prefill",
    "f3_label_b": "B-ours\nθ=0.92 speculative",
    "f3_bar_note": "mean {mean:.2f}\nmedian {med:.2f}",
    "f3_diff": "Paired A-B: {diff:+.2f} ms\n95% CI [{lo:.2f}, {hi:.2f}]",
    "f3_note": "Same paired batch (n={n}): (a) wall-clock primary metric (B slower); "
               "(b) optimistic lower bound under the synchronous oracle endpoint (upper bound of speculation benefit, B lower).\n"
               "The two views measure different objects and must not be added or conflated.",
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
    """C-E2 九工作点双口径：(a) oracle TTFT_eff 下界对 pooled 浪费率；(b) 实际墙钟平坦。"""
    with open(E1E2C_ANALYSIS, encoding="utf-8") as f:
        data = json.load(f)
    cs = data["condition_summaries"]
    conds = data["design"]["conditions"]
    thresholds = [c for c in conds if c.startswith("b_threshold_")]
    never = "b_never_speculate"
    hl = data["design"]["confirmatory_condition"]
    assert len(thresholds) == 8 and conds[-1] == never and hl in thresholds
    n = cs[never]["n"]
    assert all(cs[c]["n"] == n for c in thresholds + [never])

    def waste(c):
        return 100.0 * cs[c]["pooled_token_waste_ratio"]

    def oracle(c):
        return cs[c]["ttft_eff_ms_oracle_latency_lower_bound"]["mean"]

    def wall(c):
        return cs[c]["arrival_to_first_token_ready_ms_primary"]

    # 运行时复核验收结论：浪费率随 θ 非增、oracle 均值非降、墙钟全条件平坦
    assert all(waste(a) >= waste(b) - 1e-9 for a, b in zip(thresholds, thresholds[1:]))
    assert all(oracle(a) <= oracle(b) + 1e-9 for a, b in zip(thresholds, thresholds[1:]))
    assert waste(never) == 0.0 and oracle(never) >= oracle(thresholds[-1])
    wall_means = [wall(c)["mean"] for c in thresholds] + [wall(never)["mean"]]
    lo, hi = min(wall_means), max(wall_means)
    assert hi - lo < 1.0
    print(
        f"[fig6-2] 9 pts n={n}: waste {waste(thresholds[0]):.1f}%→0%, "
        f"oracle {oracle(thresholds[0]):.2f}→{oracle(thresholds[-1]):.2f}→never "
        f"{oracle(never):.2f} ms, wall flat {lo:.1f}-{hi:.1f} ms"
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.9, 3.5), sharex=True, gridspec_kw={"wspace": 0.32}
    )
    xs = [waste(c) for c in thresholds]

    # ---- (a) oracle TTFT_eff 下界（推测收益上界的反面：越低越好） ----
    ax_a.plot(xs, [oracle(c) for c in thresholds], "-", color=C_BLUE, lw=1.6, zorder=2)
    ax_a.plot(
        xs, [oracle(c) for c in thresholds], "o", color=C_BLUE, ms=5.5,
        zorder=3, label=L["f2_leg_pts"],
    )
    ax_a.plot(
        [waste(never)], [oracle(never)], "s", mfc="white", mec=C_VERMI,
        ms=8, mew=1.8, zorder=3, label=L["f2_leg_sent"],
    )
    # 低阈值端四点在图底部拥挤，用引线把 θ 标签扇形展开
    leader = {0.1979: (21.0, 6.0), 0.3906: (18.2, 10.8), 0.5833: (16.2, 15.0), 0.776: (14.6, 19.2)}
    offset = {0.0052: (-5, 8, "right"), 0.85: (9, 4, "left"), 0.9688: (8, -3, "left")}
    for c in thresholds:
        t = float(c.rsplit("_", 1)[1])
        if c == hl:
            continue
        if t in leader:
            ax_a.annotate(
                f"θ={t:g}", (waste(c), oracle(c)), xytext=leader[t], textcoords="data",
                ha="left", fontsize=7.3, color=G_DARK,
                arrowprops=dict(arrowstyle="-", color=G_MID, lw=0.7),
            )
        else:
            dx, dy, ha = offset[t]
            ax_a.annotate(
                f"θ={t:g}", (waste(c), oracle(c)), textcoords="offset points",
                xytext=(dx, dy), ha=ha, fontsize=7.3, color=G_DARK,
            )
    ax_a.scatter(
        [waste(hl)], [oracle(hl)], s=200, facecolors="none", edgecolors=C_VERMI,
        linewidths=1.5, zorder=4,
    )
    ax_a.annotate(
        L["f2_hl"], (waste(hl), oracle(hl)), xytext=(7.2, 26.0), textcoords="data",
        ha="left", va="center", fontsize=7.8, color=G_DARK,
        arrowprops=dict(arrowstyle="-", color=G_MID, lw=0.7),
    )
    ax_a.set_xlim(-1.8, 33.5)
    ax_a.set_ylim(-1.5, 34.5)
    ax_a.set_xlabel(L["f2_xlabel"])
    ax_a.set_ylabel(L["f2_ylabel_a"])
    ax_a.set_title(L["f2_title_a"], fontsize=10)
    ax_a.legend(loc="upper right", fontsize=7.8)

    # ---- (b) 实际墙钟主指标：同九点，全条件平坦 ----
    means = [wall(c)["mean"] for c in thresholds]
    yerr = [
        [wall(c)["mean"] - wall(c)["q1"] for c in thresholds],
        [wall(c)["q3"] - wall(c)["mean"] for c in thresholds],
    ]
    ax_b.errorbar(
        xs, means, yerr=yerr, fmt="o-", color=C_BLUE, lw=1.5, ms=5,
        capsize=3, zorder=3,
    )
    ax_b.errorbar(
        [waste(never)], [wall(never)["mean"]],
        yerr=[[wall(never)["mean"] - wall(never)["q1"]], [wall(never)["q3"] - wall(never)["mean"]]],
        fmt="s", mfc="white", mec=C_VERMI, ms=8, mew=1.8, capsize=3, zorder=3,
    )
    ax_b.scatter(
        [waste(hl)], [wall(hl)["mean"]], s=200, facecolors="none",
        edgecolors=C_VERMI, linewidths=1.5, zorder=4,
    )
    ax_b.annotate(
        L["f2_hl"], (waste(hl), wall(hl)["mean"]), textcoords="offset points",
        xytext=(8, 6), ha="left", va="bottom", fontsize=7.8, color=G_DARK,
        arrowprops=dict(arrowstyle="-", color=G_MID, lw=0.7),
    )
    ax_b.text(
        0.52, 0.28, L["f2_flat"].format(lo=lo, hi=hi), transform=ax_b.transAxes,
        ha="center", va="center", fontsize=8.2, color=G_DARK,
        bbox=dict(facecolor="white", edgecolor=G_MID, alpha=0.9, pad=2.5),
    )
    ax_b.set_ylim(0, 80)
    ax_b.set_xlabel(L["f2_xlabel"])
    ax_b.set_ylabel(L["f2_ylabel_b"], fontsize=9.5)
    ax_b.set_title(L["f2_title_b"], fontsize=10)

    fig.subplots_adjust(bottom=0.26, top=0.90, left=0.075, right=0.985)
    fig.text(
        0.075, 0.015,
        L["f2_note"].format(
            n=n, sess=data["design"]["sessions"], dial=data["design"]["dialogues_per_session"]
        ),
        fontsize=8.2, va="bottom", ha="left", color=G_DARK,
    )
    _save(fig, "fig6_2")


# ---------------------------------------------------------------- 图 6-3
def fig6_3():
    """C-E1 配对双口径：(a) 实际墙钟 arrival→first-token-ready；(b) oracle TTFT_eff 下界。"""
    with open(E1E2C_ANALYSIS, encoding="utf-8") as f:
        data = json.load(f)
    cs = data["condition_summaries"]
    cond_a, cond_b = data["design"]["e1_pair"]
    assert cond_a == "system_a_full_prefill"
    assert cond_b == data["design"]["confirmatory_condition"]
    n = data["e1"]["primary_paired"]["n"]
    assert n == cs[cond_a]["n"] == cs[cond_b]["n"]

    a_wall = cs[cond_a]["arrival_to_first_token_ready_ms_primary"]
    b_wall = cs[cond_b]["arrival_to_first_token_ready_ms_primary"]
    a_orc = cs[cond_a]["ttft_eff_ms_oracle_latency_lower_bound"]
    b_orc = cs[cond_b]["ttft_eff_ms_oracle_latency_lower_bound"]
    # System A 无推测：oracle 口径与墙钟口径是同一观测量
    assert abs(a_wall["mean"] - a_orc["mean"]) < 1e-6

    diff_wall = data["e1"]["primary_paired"]["absolute_difference_ms_a_minus_b"]["mean"]
    ci_wall = data["bootstrap"]["ci"][
        "e1_primary_mean_arrival_to_ready_difference_ms_system_a_minus_b092"
    ]
    diff_orc = data["e1"]["oracle_ttft_eff_latency_lower_bound_paired"][
        "absolute_difference_ms_a_minus_b"
    ]["mean"]
    ci_orc = data["bootstrap"]["ci"][
        "e1_oracle_mean_ttft_eff_difference_ms_system_a_minus_b092"
    ]
    assert diff_wall < 0 < diff_orc
    assert ci_wall["upper"] < 0 < ci_orc["lower"]
    print(
        f"[fig6-3] n={n} wall A {a_wall['mean']:.2f} vs B {b_wall['mean']:.2f} "
        f"diff {diff_wall:+.2f} CI [{ci_wall['lower']:.2f},{ci_wall['upper']:.2f}]; "
        f"oracle A {a_orc['mean']:.2f} vs B {b_orc['mean']:.2f} "
        f"diff {diff_orc:+.2f} CI [{ci_orc['lower']:.2f},{ci_orc['upper']:.2f}]"
    )

    def panel(ax, a, b, title, diff, ci):
        x = [0, 1]
        means = [a["mean"], b["mean"]]
        ax.bar(
            x, means, width=0.52, color=[C_ORANGE, C_BLUE],
            edgecolor="black", lw=0.6, zorder=2,
        )
        # IQR 竖线（q1→q3，带端帽）+ 中位刻度：均值可能落在 IQR 外（如 bimodal oracle），不能作误差棒
        for xi, stat in zip(x, (a, b)):
            ax.plot([xi, xi], [stat["q1"], stat["q3"]], color=G_DARK, lw=1.2, zorder=3)
            for q in ("q1", "q3"):
                ax.plot([xi - 0.07, xi + 0.07], [stat[q], stat[q]], color=G_DARK, lw=1.2, zorder=3)
            ax.plot(
                [xi - 0.10, xi + 0.10], [stat["median"], stat["median"]],
                color="white", lw=1.6, zorder=4,
            )
            ax.annotate(
                L["f3_bar_note"].format(mean=stat["mean"], med=stat["median"]),
                (xi, stat["q3"]), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8.2,
            )
        ax.text(
            0.5, 0.97, L["f3_diff"].format(diff=diff, lo=ci["lower"], hi=ci["upper"]),
            transform=ax.transAxes, ha="center", va="top", fontsize=8.2,
            bbox=dict(facecolor="white", edgecolor=G_MID, alpha=0.9, pad=2.5),
        )
        ax.set_xticks(x)
        ax.set_xticklabels([L["f3_label_a"], L["f3_label_b"]])
        ax.set_ylim(0, max(a["q3"], b["q3"]) * 1.55)
        ax.set_ylabel(L["f3_ylabel"])
        ax.set_title(title, fontsize=10)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.9, 3.6), gridspec_kw={"wspace": 0.30}
    )
    panel(ax1, a_wall, b_wall, L["f3_title_a"], diff_wall, ci_wall)
    panel(ax2, a_orc, b_orc, L["f3_title_b"], diff_orc, ci_orc)
    fig.subplots_adjust(bottom=0.24, top=0.90, left=0.075, right=0.985)
    fig.text(
        0.075, 0.015, L["f3_note"].format(n=n),
        fontsize=8.2, va="bottom", ha="left", color=G_DARK,
    )
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
    figures = {"fig6_1": fig6_1, "fig6_2": fig6_2, "fig6_3": fig6_3, "fig6_4": fig6_4}
    requested = [a for a in sys.argv[1:] if not a.startswith("-")]
    todo = {name: fn for name, fn in figures.items() if not requested or name in requested}
    unknown = set(requested) - set(figures)
    assert not unknown, f"unknown figure names: {sorted(unknown)}"
    for fn in todo.values():
        fn()
    print(("EN" if EN else "ZH"), "完成：", ", ".join(todo), "→", FIGDIR)
