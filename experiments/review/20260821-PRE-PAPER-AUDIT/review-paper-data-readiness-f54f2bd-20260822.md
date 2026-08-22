# 论文修改前实验数据就绪度复核（f54f2bd，2026-08-22）

- **审查对象**：`experiments/CISR_REVISION_PLAN.md`、`experiments/EXPERIMENT_DESIGN.md`、`experiments/results/revision/PAPER_HANDOFF.md`、`PAPER_WRITING_REFERENCE.md`、`REVISION_CHANGELOG.md`、`review/20260821-PRE-PAPER-AUDIT/deviation-waiver-r7-tts-control-20260822.md`，以及 R1–R7 正式产物。
- **当前提交**：`f54f2bd3594b6725646bc122f184c7eff7c01cb8`，工作树 clean。
- **结论**：**核心实验数据已经齐全并可支持按路线图改稿；但“论文写作包已完全无歧义、可直接交给作者动笔”尚未达到。需要先完成一次机械性的 Table VIII 装配，并清理若干仍会误导写作者的旧口径；另外 W7 人工抽检仍是未完成项。**

## 1. 已经齐全并通过审查的数据

### R1：统计稳健性

已具备：

- `table3_latency_percentiles.csv`；
- `table4_ablation_percentiles.csv`；
- `table5_context_percentiles.csv`；
- `repeat_cv_summary.csv`、`repeat_cv_persample.csv`；
- Table III/IV/V 的过滤清单和分位数结果；
- Fig.6 的 PDF、PNG 和 bins CSV。

W3 的 `ddof=1` 口径也已明确：System B mean CV 5.19%、median 4.05%、P90 10.73%、max 18.96%；System A 对应 5.23%、4.65%、9.92%、14.01%。因此“CV<5%”只能作为历史作废口径，不能再写入论文。

### R2：真实语音

已具备：

- LibriSpeech/AISHELL-1 clean 语音及构建 manifest；
- 12 个增强变体及静态 QA；
- `ttft_real.csv`、`wer_real.csv`、逐样本结果和 `qa_transcribe.corrected.csv`；
- `reference_full` 强制使用、英文大小写折叠、中文去接缝空格等评分口径；
- babble 失败边界及空输出统计；
- AISHELL 中文数字/阿拉伯数字失配脚注所需数字。

该部分数据可用于 Table VI、真实语音方法段、limitations 和审稿回复。不能把拼接朗读语音称为自然对话，也不能把 babble 结果概括为“各种噪声下均优”。

### R3：LocalAgreement 基线

已具备：

- 498 条配对清单及排除规则；
- System A/B/LA 三方同机结果；
- TTFT、WER/CER、逐样本结果；
- LA 修复后的绝对时间轴、句界裁剪和 `la_max_buffer_s=15.0` 方法说明；
- 旧无效现场与重跑过程归档。

可直接支持 Table VII 和意见 4 回复，但论文必须写“修复后的 LA 实现语义”，不能只写 LocalAgreement-2 名称。

### R4/R5：机制与语义

已具备：

- `commit_divergence.json`；
- `tokenizer_seams.csv`；
- 完整回复、逐片段提交日志；
- `semantic_consistency.csv`、summary、逐样本 judge JSON；
- `REPRO_METADATA.md`；
- W5 的 paired bootstrap/Wilcoxon/Holm/effect-size 结果。

可支持：回滚下发为 0、内部漂移分布、BPE 接缝分歧、嵌入余弦和独立意图满足度。不能写“内部文本从不变化”“不匹配率极低”或“统计等价/不可区分”等已经撤销的强表述。

### R7：统一 TTFA

`r7_main` 已满足论文数据所需的正式证据链：

- 140/140 success；
- QA 0，结果级复核 47/47；
- 统一 `perf_counter_ns` 时间轴；
- A/B 配对、计划、平台、Silero、配置及样本 hash 绑定；
- `ttfa_summary_r7_main.csv` 具备总 TTFA 和全部组件分项；
- `ttfa_subset_cv_r7_main.csv` 具备重复子集稳定性；
- `r7_tts_control` 32 条数据经偏差豁免采信，且披露和不可变双哈希归档已完成。

组件闭合可由 CSV 直接复核。例如 ALL 行：

```text
B: 0.1 + 133.0 + 1383.1 + 387.0 + 0.4 + 3578.3 ≈ 5481.9 ms
A: 0.1 + 0.0 + 4182.9 + 4681.1 + 0.6 + 13561.1 ≈ 22425.7 ms
```

因此 Table VIII 的底层数据已经齐全。R7 是唯一合法 TTFA 数据源；旧 `r6_ttfa/ttfa_budget.csv` 不得再参与装配。

## 2. 仍未达到“可直接写作”的事项

### 2.1 W7 人工 spot check 仍未完成

`r2_real_speech/MANUAL_SPOT_CHECK.md` 仍是空模板，明确写着“需需求方本人完成”。候选 5 条样本、字段和试听标准已经准备好，但尚未记录：

- 试听者与日期；
- 可懂度；
- 截断、错序、爆音/削波、异常静音、音量异常；
- 拼接缝感知；
- 每条样本通过/不通过结论。

这不是新的模型实验，也不影响 R2 数值本身；但按原路线图，W7 属于进入论文前的人工 QA 证据，严格说应完成后再宣称“所有前置条件全部满足”。它只能被称为 manual spot check，不能升级成 human evaluation。

### 2.2 `PAPER_HANDOFF.md` 仍有未清理的旧口径

当前文档仍有会被直接复制进论文的旧数字：

- §E1 仍写“逐样本 CV mean 4.2% / median 3.3%，结论 CV<5%”；这与已裁定的 `ddof=1` 结果矛盾；
- R2 顶部仍写 AISHELL CER 6.72%，而终版应为 10.73%/10.77% 所对应的修正口径，并附数字写法失配说明；
- 历史 E5/E6 装配段仍直接显示 B 14.79s/A 22.67s。虽然 R7 小节已明确作废，但这类“论文可用”旧段落仍容易被误用。

建议将这些旧段落改为醒目的“历史作废、不得引用”，或删除可直接复制的旧结论，只保留审计追溯链接。

### 2.3 `PAPER_WRITING_REFERENCE.md` 仍有一处旧数字引用冲突

总册 §十已正确记录 ddof=1 新口径并将“CV<5%”标为作废，但 §七回复信证据表仍写：

```text
CV mean 4.2%/median 3.3%
```

这会使写作者在同一文档内看到两个不同答案。该行应改为 ddof=1 的完整分布口径，或直接引用 `repeat_cv_summary.csv`，不得继续保留 4.2%/3.3% 作为摘要数字。

此外总册开头对 Table VIII 的旧来源仍先写 E5/E6/补测，后面才切换为 R7。虽然有作废声明，但建议把开头平台/来源表直接改成 R7，避免作者先读到错误来源。

### 2.4 Table VIII 尚未实际装配成最终论文表

当前已经有全部分项，但 `PAPER_WRITING_REFERENCE.md` 明确仍处于：

```text
W8 阶段 2：新 Table VIII 装配
```

因此目前是“底层数据齐全”，不是“最终表格已锁定”。装配时至少要固定：

- 单位为 ms 或 s，全文统一；
- repeat0、n=50 的统计范围；
- B/A 的 mean、std、p50、p90、p95；
- 组件分项表的行列布局；
- `received` 与 `playable` 选择哪一项作为主指标（建议主表使用 `first_playable_pcm`，received 作为 QA/补充）；
- `tts_control` 7076 ms 的用途和偏差豁免脚注；
- 不再列入旧装配表中的估计项和跨运行分项。

这一步不需要 GPU，也不需要新增实验，但在完成前不应称 Table VIII 已最终定稿。

### 2.5 路线图的 R7 映射仍使用旧编号

`CISR_REVISION_PLAN.md` §8 的表格仍把新 Table VIII 映射为 R6；这属于路线图历史结构未同步，不是数据缺失。建议在写作前将其更新为“R7 统一 TTFA（替代原 R6 装配）”，并注明 R6 的 E5/E6 单项结果仅作背景/边界证据。

### 2.6 原始 `EXPERIMENT_DESIGN.md` 的旧“待完成”标题仍存在

§5.3 仍写“执行与分析（待完成）”，并列出完整实验、收集数据、绘图未勾选。这一节属于原始实验设计的历史章节，但当前会让读者误以为项目仍缺少基础实验。建议改为“历史基线执行记录（已由归档结果覆盖）”或添加明确的历史状态说明。

## 3. 数据锁定状态判定

| 层级 | 状态 | 说明 |
|---|---|---|
| R1–R5 核心数值 | 通过 | 结果文件、统计脚本、方法口径和 QA 齐全 |
| R2 人工 spot check | 未完成 | 模板已在，人工字段为空 |
| R6 单项背景测量 | 通过 | E5/E6/补测可追溯，但不再装配 Table VIII |
| R7 正式 TTFA | 通过 | r7_main 47/47，底层组件齐全 |
| r7_tts_control | 通过（偏差豁免） | 可采信，但引用必须带披露 |
| 最终 Table VIII | 未装配 | W8 阶段 2 尚未完成 |
| 文档口径一致性 | 未完全通过 | PAPER_HANDOFF 和证据表有旧数字/旧来源 |
| 论文改稿可开始 | 条件通过 | 可开始结构和方法文字；正式填表前先完成上述清理 |

## 4. 最终结论

**实验数据本身已准备齐全，足以按 CISR 修订路线图开始论文修改；但不能据此宣称“所有论文数据前置工作已完全结束”。**

建议采用以下边界：

- **现在可以做**：论文结构修改、方法描述、真实语音/R3/R4/R5 章节草稿、R7 TTFA 表的排版草稿；
- **先完成再锁定**：W7 五条人工 spot check、W8 Table VIII 装配、清理 PAPER_HANDOFF/PAPER_WRITING_REFERENCE 的冲突旧数字、同步 CISR_REVISION_PLAN 和 EXPERIMENT_DESIGN 的状态；
- **之后才可宣布**：最终数据锁定、正式 Table III–VIII 定稿、基于锁定数据修改 `main.tex` 和最终回复信。

没有发现需要新增 GPU 实验、重跑 r7_main 或重跑 tts_control 的科学性缺口。当前剩余工作是人工 QA、机械装配和文档去歧义，不是实验数据采集。