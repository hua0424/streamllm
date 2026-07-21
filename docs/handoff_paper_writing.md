# 论文收尾交接文档（绘图 + 统稿）

> 面向下一台主机 / 下一个会话。目标：**画出 4 张图 → 统稿 → 交稿**。
> 实验已全部完成（数据在 `experiments/results/*.json`，可直接读）；八章初稿已全部写完。
> 实验执行相关见 `docs/handoff.md`（实验机 runbook）；设计决策见 `docs/decisions.md`（D-001~D-012）；
> 项目全景与里程碑见 `docs/paper2_context.md`。

**生成时间**：2026-07-17　**分支**：`paper2`（HEAD `40d4585`，已推送）

---

## 一、当前状态

| 项 | 状态 |
|---|---|
| 实验数据 | ✅ 全部完成（7B + 真实 MultiWOZ + TEN 7B + Mistral 裁判 + 37 条人工仲裁） |
| 论文八章初稿 | ✅ 全部完成（`paper2/chapter1..8_*.md`） |
| **图 6-1 ~ 6-4** | ✅ 完成（2026-07-21，`plot_figures.py` → `paper2/figures/`，已织入 ch6） |
| 统稿（符号/交叉引用/编号） | ✅ 完成（2026-07-21，见下方勘误记录；⚠️ 本文档 §3.3 的"ρ(KV复用率)、W(浪费率)"写反了——ch3 定义 3.9 中 **ρ=推测浪费率**，以 ch3 为准） |
| 摘要 + 参考文献合并 | ✅ 完成（`paper2/abstract.md`、`paper2/references.md`；4 条 † arXiv 编号已复核无误） |
| 方法示意图 5 张 | ✅ 完成（2026-07-21）：图 3-1 三指针、图 4-1 架构（替换 ASCII）、图 4-2 状态机、图 4-3 时间轴、图 4-4 KV 截断。drawio 源在 `paper2/figures/src/*.drawio`，导出 SVG+PNG 在 `paper2/figures/`。重导命令：`& "D:\Program Files\draw.io\draw.io.exe" -x -f svg -e -b 10 -o <out.svg> <src.drawio>`（⚠️ XML 属性值内不能有裸双引号，否则 drawio 静默挂死；改完先 `[xml](Get-Content -Raw ...)` 验证） |
| 全文合并草稿 | ✅ `paper2/thesis_draft.md`（自动拼接：摘要+ch1..8+文献；改章节后重跑拼接命令，勿直接编辑） |
| 剩余 | 按学校模板转 DOCX/LaTeX（`/ars-format-convert`）+ 送审前 `/ars-reviewer` 模拟评审 |

**注意**：ch2 与 ch1/7/8 由一个会话写，ch3~ch6 由并行会话写——**符号与表号需统稿时对齐**（详见 §四）。

---

## 二、绘图任务（主任务）

### 通用要求

- **脚本位置**：`experiments/scripts/plot_figures.py`（新建，一个脚本出全部 4 图；ch6 第 4 行已注明由 `plot_*.py` 生成）
- **输出**：`paper2/figures/fig6_1.pdf` ~ `fig6_4.pdf`（矢量；同时出 `.png` 便于预览）
- **依赖**：`matplotlib`（项目已装）；中文标签需设 `plt.rcParams['font.sans-serif']=['SimHei'|'Noto Sans CJK SC']`、`axes.unicode_minus=False`；**若目标机无中文字体，全部标签改用英文**（学位论文可用中文图注 + 英文轴标）
- **风格**：学位论文黑白友好——线型/填充图案区分（不能只靠颜色），字号 ≥9pt，`bbox_inches='tight'`
- **数据一律从 JSON 读，禁止硬编码数字**（保证图文一致、可复现）

### 图 6-1　E3 分注入位置的未听引用率（条形图）

- **数据**：`experiments/results/exp3_consistency_judged.json` → `records[]`
- **聚合**：按 `(condition, fraction)` 分组，统计 `referenced_unheard`（规则口径）与 `judge_referenced_unheard`（裁判口径）的均值
- **实测值（应与脚本输出一致，用于自检）**：

| fraction | B-gen 规则 | B-gen 裁判 | B-ours 规则/裁判 |
|---|---|---|---|
| 0.25 | 85.4% | 4.9% | 0.0% / 0.0% |
| 0.5 | 48.5% | 1.0% | 0.0% / 0.0% |
| 0.75 | 20.4% | 1.9% | 0.0% / 0.0% |
| boundary | 49.5% | 2.9% | 0.0% / 0.0% |

- **画法**：x 轴 = 注入位置（0.25/0.5/0.75/boundary）；分组柱：B-gen-规则、B-gen-裁判、B-ours（两口径均为 0，画贴地零柱并标 "0.0%" 使"构造性为零"可见）；y 轴 = 未听内容引用率（%）
- **要点**：突出 B-gen 随注入位置**单调递减**（打断越早未听内容越多→越易被复现）；B-ours 恒零
- **图注**：需说明"B-ours 的零为机制构造性保证，非统计结果"（D-006/D-012 护栏）

### 图 6-2　E2 推测浪费率–TTFT 权衡前沿（核心图）

- **数据**：`exp2_tradeoff.json` → `curve[]`，字段 `threshold` / `spec_waste_rate` / `ttft_eff_ms` / `survived_rate` / `avg_ready_tokens`（9 点）
- **画法**：**主图**散点+连线，x = `spec_waste_rate`（%），y = `ttft_eff_ms`（ms），每点标注其 `threshold`；标出拐点区（θ≈0.85~0.97，浪费率 10.7%→0.8% 而 TTFT 5.8→29.3 ms）；右上角/次坐标可叠 `survived_rate` 作辅助线（虚线、右轴）
- **实测九点（自检）**：(29.2%, 0.5ms, θ=0.0052) → (16.2%, 0.1) → (14.4%, 0.6) → (13.6%, 1.1) → (11.5%, 3.9) → (10.7%, 5.8, θ=0.85) → (4.5%, 12.1, θ=0.92) → (0.8%, 29.3, θ=0.9688) → (0%, 48.5, θ=1.1 永不推测哨兵)
- **要点**：这是**论文核心图**——展示"以冗余计算换响应速度"的连续可调前沿；哨兵点（永不推测）作为保守极限基线单独标记（不同 marker）

### 图 6-3　E1 延迟对比（分组柱 + 建模量标注）

- **数据**：`exp1_latency.json` → `summary`：`a_ttft_ms=27.4`、`b_ttft_eff_ms=0.6`、`a_mouth_to_ear_ms_modeled=9080.2`、`b_mouth_to_ear_ms_modeled=2481.8`
- **画法**：两组柱（System A / B-ours），每组两根（TTFT、mouth-to-ear）；**因量级差异大（0.6ms vs 9080ms），y 轴用对数刻度**或拆成上下两子图（推荐子图：上 TTFT，下 mouth-to-ear）
- **必须标注**：mouth-to-ear 为**建模值**（首片段就绪 + TTS 首块延迟，画像为 3090 实测）；在 B-ours 的 m2e 柱上标出"其中 2434 ms 为 TTS 首块合成延迟"（分段填充或注释箭头）——ch6 强调该项强依赖推理加速条件
- **图注**：注明 A100+TensorRT 官方 45 ms 外推下 B-ours 可至 ~100ms 量级，而 A 的"等全部生成+整段合成"是结构性成本

### 图 6-4　A1 KV 复用 vs 重新 prefill（双对数线图）

- **数据**：`exp_a1_kvreuse.json` → `results[]`，字段 `ctx_len` / `crop_only_ms` / `role_rebuild_ms` / `reprefill_ms` / `speedup`（6 点：275/527/1031/2060/4097/8192）
- **画法**：x = `ctx_len`（对数轴），y = 延迟 ms（对数轴）；三条线：① `crop_only_ms`（≈0.13~0.34ms，近水平——**关键路径**）② `role_rebuild_ms`（≈12~47ms，标注"非关键路径，可延迟执行"）③ `reprefill_ms`（71.7→1863.4ms，近线性）；次轴或标注 `speedup`（2.8×→39.7×）
- **要点**：视觉上呈现"亚毫秒恒定 vs 线性增长"的量级鸿沟，支撑"打断即停"卖点

### 绘图完成后

1. 把 ch6 里的 `[图 6-x：...]` 占位替换为正式引用（`如图 6-1 所示`），并确认图号与出现顺序一致
2. 若图中数字与正文表格有出入 → **以 JSON 为准**改正文（禁止改图迁就文字）

---

## 三、其余收尾任务

### 3.1 摘要 + 关键词
中英双语。骨架：问题（级联三进度不一致致打断后历史失真）→ 方法（播放感知 KV 截断 + 推测调度 + 历史策略）→ 结果（引用率构造性归零 / 亚毫秒响应 / 8k 处 39.7× / 前沿可调）→ 意义（首个开源可复现级联实现）。**可用 `/ars-abstract`**。

### 3.2 参考文献合并
ch2 末尾是章内局部文献表，需并入全文统一编号。**送审前必须复核 † 标记的 arXiv 预印本编号**（RelayS2S 2603.23346、LTS-VoiceAgent 2601.19952、IntentKV 2606.09916、Speculative Interaction Agents 2605.13360——`26xx` 月份前缀需确认是否已正式发表）。源清单见 `docs/research_novelty_check.md` §附。

### 3.3 统稿检查清单
- [ ] **符号一致性**：ch3 定义的 $g/s/p$、$H(p)$、$\varepsilon(p)$、$\Phi$(反查)、$\rho$(KV复用率)、$W$(浪费率)、$\theta_s/\theta_c$ 在 ch4~ch6 中用法一致（ch2 与 ch3~6 出自不同会话，重点查这里）
- [ ] **表号连续**：ch6 现有"表 6-x"（人工仲裁表）需改为正式编号并检查 6-1/6-2/6-3 顺序
- [ ] **交叉引用**：ch1 §1.3 的贡献编号 C1/C2/C3 与 ch4 各节、ch6 各实验的对应关系
- [ ] **D-006 护栏全文复查**：任何"首次提出/原创"式表述必须限定在"开源可复现实现 + 机制级"范围；商用系统必须以 prior art 引用（ch1/ch2/ch7 已守，统稿时再扫一遍）
- [ ] **数字一致**：全文所有实验数字与 JSON 对齐（39.7×、0.649、51%、2.7%、4.5%、12.1ms 等）

---

## 四、必读上下文（避免重复踩坑）

1. **E3 的双口径必须成对出现**：loose（表面重叠上界，规则）与 strict（严格 GT），且 **B-ours 的 loose=0 是构造性保证不是实验发现**——这是 D-012 修复的核心，写作与图注都不能表述成"实验测得 B-ours 为 0"
2. **主数字用裁判口径**：人工仲裁 κ_人-裁判=0.649 vs κ_人-规则=−0.073，规则口径只作上界
3. **A2 重写未显收益如实报告**（naive 3.76 > rewrite 3.62），ch7 已有分析，勿粉饰
4. **B-syn 不得称"已验证"**：Mock 同步合成下与 generation 等价（D-012）

---

## 五、建议调用的 skills

- **`/ars-abstract`** —— 双语摘要 + 关键词
- **`/ars-citation-check`** —— 参考文献错误报告（正好覆盖 †arXiv 编号复核）
- **`/ars-reviewer`** —— 统稿后模拟同行评审，查 D-006 护栏是否有漏网的过度主张
- **`/ars-format-convert`** —— Markdown → DOCX/LaTeX（按学校模板要求）
- **`code-review`** —— 若还要改实验代码（本项目两轮审查抓出过 3 个会污染数字的 bug，保持习惯）

---

## 六、路径速查

- 论文正文：`paper2/chapter1..8_*.md`、大纲 `paper2/outline.md`、实验设计 `paper2/experiment_design.md`
- 数据：`experiments/results/{exp1_latency,exp2_tradeoff,exp3_consistency_judged,exp_a1_kvreuse,exp_a2_history_judged,e3_human_arbitration,cosyvoice_profile,trigger_calibration}.json`
- 待建：`experiments/scripts/plot_figures.py`、`paper2/figures/`
- 核心实现（写 ch5 需引用）：`src/dialogue/{timeline,orchestrator,trigger,rewriter,unheard_detector}.py`、`src/tts/{sentence_chunker,streaming_tts,cosyvoice_tts}.py`、`src/player/player.py`、`src/llm/stream_llm_inference.py`
