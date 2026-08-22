# 正式论文修改前置准备复核（ed9e648，2026-08-22）

- **数据锁定基线**：`7c93b77`
- **文档清理提交**：`ed9e648`
- **目标论文**：`paper/tougao_new/3次调整/latex/CISR-submission/submission/main.tex`
- **最终结论**：**没有剩余的数据或实验前置阻塞，可以立即开始修改论文。** 当前仍需完成的 Fig.6 替换、参考文献补充、旧论断更新和回复信撰写，均属于正式改稿本身的首批工作，不需要在开工前另行等待。

## 1. 数据与清理状态

### 已通过

- `7c93b77...ed9e648` 未修改 R1–R5 结果/统计 CSV、`r7_main/` 或 `tts_control/`；
- 清理仅涉及写作总册、交接文档、changelog、历史回复函更正注和最终锁定审查报告；
- 工作树 clean，`git diff --check` 通过；
- 旧 TTFA 方案 (a)、装配待完成、数据锁定待确认和 W7 待试听等状态已历史化；
- `PAPER_HANDOFF.md` 的六项待决已明确全部裁决；
- 当前唯一正式数据源、冻结规则和书面变更规则均明确。

### 一处非阻塞审计链接笔误

`review-final-data-lock-7c93b77-20260822.md` 的规格来源写成：

```text
review-reply-paper-data-readiness-f54f2bd-20260822.md
```

实际文件名是：

```text
reply-review-paper-data-readiness-f54f2bd-20260822.md
```

这是审计链接文字错误，不影响实验、锁定或论文开工。可按纯路径修复规则登记后直接更正。

## 2. 论文工程现状

### 基线编译可用

将完整投稿目录复制到临时目录后，执行完整 LaTeX/BibTeX 构建成功：

```text
pdflatex → bibtex → pdflatex ×2
```

输出：

- 13 页；
- IEEE conference 双栏、Letter 页面；
- 无 fatal、无未定义引用、无未定义 citation；
- 只有非致命 underfull box 警告。

仓库中的旧 `main.log` 记录“无法写 main.pdf”，原因是当时 PDF 被外部程序占用，不是 TeX 源文件错误。改稿时应关闭 PDF 占用程序，或在临时/构建目录编译。

投稿须知只规定正文双栏不少于 4 页（不含参考文献），未在仓库材料中发现最大页数限制。当前 13 页满足最低要求，但新增三张表、扩展 Table III–V 和新增实验段落后会明显增长，需要在改稿过程中持续检查版面。

## 3. 改稿首批必须完成的内容

以下不是“开始前等待项”，而是正式修改的第一批工作。

### 3.1 替换 Fig.6

投稿目录中的 `Fig6.pdf` 与锁定的新图不同：

```text
当前投稿目录 Fig6 SHA-256: 6b21203e...
锁定新版 Fig6 SHA-256:     ef95050b...
```

新版图位于：

```text
experiments/results/revision/fig/Fig6.pdf
```

必须用新版 12 等频分箱、P5–P95 阴影带图替换投稿目录旧图，并重新检查图中文字在 100% 缩放下可读、无中文字符、图与标题同页。

### 3.2 补充新实验所需参考文献

当前 `refs.bib` 仅有原 22 条，缺少新小节必需来源：

- LibriSpeech；
- AISHELL-1；
- MUSAN；
- whisper-streaming/LocalAgreement 策略来源；
- BGE-M3；
- DeepSeek judge 模型或其正式服务/技术报告来源。

写稿前无需先单独建立一份新 BibTeX 文件，但在新增对应方法和结果段落时必须同步加入 `refs.bib` 并正文引用。若 DeepSeek 当前没有适合的学术条目，应引用可核验的官方模型/服务文档，并在方法中报告固定模型名、日期、temperature=0 和顺序随机化，不得仅写“an LLM judge”。

投稿须知要求：全部文献必须在正文引用，参考文献不少于 10 条，须含近三年文献，来源作者覆盖至少三个国家。当前数量已超过 10 条；新增条目后仍需在最终构建中检查没有未引用或未定义项。

### 3.3 系统替换旧论文数据和强声明

`main.tex` 目前仍是修订前版本，必须按写作总册逐项修改。高风险旧内容包括：

- Table III baseline Extra Long `6753.43` 更新为锁定值，5.67s 更新为 5.66s；
- Table III–V 增加锁定的 std/P95/P99 等统计信息；
- Table IV 全表替换，并把 KV 独立贡献从旧 9.3% 改为 Extra Long 3.3%，删除递增显著的旧叙述；
- `50 utterances per group` 改为 498 条配对样本及 108/150/240 分组与排除规则；
- 删除或限定“TTFT 恒定”“任意长度”“15 秒是普遍阈值”等强声明；
- 不再声称 repetition penalty 1.1 实际生效；历史运行的有效配置是未应用该参数；
- append-only 只对下游提交成立，内部重识别漂移需报告 224 次、涉及段 52.7%；
- 删除“不降低语义理解”“小误差不影响推理”等未经限定强结论，改用 R5 探索性证据；
- 真实语音不称自然对话，应称“concatenated human-read speech”；
- Future Work 中“未来加入语义一致性指标”已过时，需更新；
- Limitations 中“主要只有 TTS 合成音频、真实语音待验证”已过时，需改为拼接真人朗读、babble 边界、单机独占平台、TTS 部署边界等。

### 3.4 新增锁定表格和实验小节

必须加入：

- Table VI：真实语音与增强条件；
- Table VII：System A/System B/LA-2 同机对比；
- Table VIII：R7 统一 TTFA 总量和组件分解；
- 重复测量与统计推断；
- append-only/内部漂移/tokenizer seam；
- 语义一致性三轨结果；
- TTFA 定义和 `first_playable_pcm` 边界。

Table VIII 只允许引用 R7 装配稿。第二分项必须写：

```text
t_feed_to_close_wait = input_close - feed_end
```

不得把 133ms 解释为 flush 计算开销。TTS control 的 7076ms 不作为 Table VIII 行项；若在回复信或正文讨论中引用，必须附流程偏差豁免披露。

## 4. 建议在写稿前固定的工作方式

这些不是数据缺口，但可减少返工：

1. 以 `ed9e648` 当前 HEAD 为论文编辑起点，以 `7c93b77` 为数字锁定基线；
2. 先复制新版 Fig.6、补 BibTeX，再分节修改正文；
3. 每完成一个实验小节即完整编译，不要等所有表格加入后一次处理浮动体；
4. 每次编译检查：undefined citation/reference、overfull box、图表标题同页、表格跨栏宽度；
5. 对 Table III–VIII 建立“论文单元格→锁定 CSV/MD”核对清单；
6. 修改全文后搜索禁止恢复词：`6753.43`、`9.3\%`、`50 utterances per group`、`CV<5`、`6.72`、`14.79`、`22.67`、`statistically indistinguishable`、`for speech of any length`；
7. 不修改已冻结结果文件；若写稿时确实需要新数字，先提出书面变更，不从现有数据临时推导后直接入稿。

## 5. 回复信准备

当前投稿目录没有与本轮五项审稿意见匹配的最终回复信。旧的替换文本/说明材料基于“无需新增实验”的早期状态，不应继续作为最终回复。

建议在正文修改完成后生成最终 response letter，并逐条包含：

- reviewer comment 原文；
- response；
- manuscript change 的 section/table/figure/page；
- 锁定数据文件和关键数字；
- 对未完全接受的强要求说明限制和澄清；
- TTS control 的偏差豁免披露。

回复信不必阻塞 `main.tex` 开工，但应与正文同步维护，避免最后无法定位修改位置。

## 6. 投稿前而非开工前的任务

以下可在论文内容定稿后完成：

- 全文英文润色和术语一致性检查；
- 相似度检查：全文不高于 24%，单项不高于 3%；
- IEEE PDF eXpress 验证；
- 图中文字清晰度、无中文字符、图题同页；
- 表格可编辑、表题同页；
- 所有公式和引用编号顺序检查；
- 最后一页双栏平衡；
- 关闭 PDF 占用后生成最终 `main.pdf`；
- 重建 `submission.zip`，确保只包含最终源文件、图片、参考文献和需要的模板文件；
- 版权转让和最终校稿确认。

## 7. 最终判断

**现在可以直接进入正式论文修改，不需要再做任何实验或数据准备。**

改稿前唯一值得立即安排的两项资源准备是：

1. 将锁定新版 Fig.6 放入投稿目录；
2. 为 LibriSpeech、AISHELL-1、MUSAN、LocalAgreement、BGE-M3 和 DeepSeek 补齐可核验引用。

除此以外，旧数字替换、新表格、新小节和回复信均属于改稿实施本身。当前最主要的项目风险不再是数据不足，而是 `main.tex` 仍保留大量修订前的强结论和旧数字，以及新增内容可能带来的版面增长。