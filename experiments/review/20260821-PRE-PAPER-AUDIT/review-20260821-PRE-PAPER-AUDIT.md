# CISR 正式改稿前实验与结论审计报告

- **日期**：2026-08-21
- **用途**：提交开发侧评估并修正实验代码、统计口径和结果说明；在本报告 P0 项关闭前，不建议正式修改论文。
- **审查对象**：
  - `experiments/CISR_REVISION_PLAN.md`
  - `experiments/results/revision/PAPER_WRITING_REFERENCE.md`
  - `experiments/results/revision/` 下实验结果、运行说明及相关实验代码
  - `paper/tougao_new/3次调整/latex/main.tex`
- **审查方式**：只读核查计划、逐样本结果、汇总 CSV/JSON、RUNINFO、统计和装配脚本，并对关键数字与计时定义交叉验证。

---

## 1. 执行摘要

新实验已覆盖大部分审稿要求，论文的核心方向性结论仍成立：流水线式流式 ASR 与增量预填能够把大量计算前移到用户发声期间，显著降低长语音的 speech-end-to-first-token latency（TTFT）。该收益已在两个双 RTX 3090 平台、合成长语音和干净真人录音上复现。

但是，目前不能把 `PAPER_WRITING_REFERENCE.md` 中的“全部数据格已定稿，可直接动笔”视为最终结论。审计发现以下阻塞项：

1. Table VIII 的 TTFA 是跨运行装配的组件预算，不是同一请求、同一时间轴上的闭合端到端实测；`T_endpoint + E4 TTFT` 还可能定义重叠。
2. E5 endpoint 指标有 8/50 个负值，`53.1 ms` 不能直接解释为物理意义上的端点等待。
3. System A 的 TTFC 使用 `0.09 s/字符`估计，但正式50条数据不能支持该线性模型，A 的 `22.67 s` TTFA 不宜作为可靠对比值。
4. 重复性材料中的“CV<5%”口径过强；使用三轮样本标准差 `ddof=1` 后，System B 的逐样本 CV mean 为5.19%，max 为18.96%。
5. 当前 WER/CER 为逐样本错误率宏平均，不是标准 corpus WER/CER，论文必须明确命名或重新聚合。
6. 语义评估没有统计检验支持“统计不可区分”；只能说样本均值接近。

此外，新实验要求收窄或推翻若干原稿次要结论：

- KV Cache 在当前配置中的独立边际收益较小，原稿“收益随长度愈加显著”的结论不成立。
- “稳定提交”只能指下游 append-only、rollback=0，不能指内部 ASR 文本不再变化。
- tokenizer seam mismatch 出现在25/50样本中，并不罕见。
- TTFT 约1.1秒不能代表完整可听响应；现有组件测量显示 TTS 首包约12.9秒，是主要瓶颈。
- 多人 babble 是明确失败边界，不能声称各种真实噪声条件下均保持优势。

**总体裁决：核心贡献未被推翻；真实语音、LocalAgreement、机制和语义实验大体足以回应审稿人；TTFA 当前仅部分满足。开发侧应先关闭本报告 P0 项，然后再锁定论文写作数据。**

---

## 2. 审稿意见满足度

| 审稿要求 | 当前状态 | 主要证据 | 裁决 |
|---|---|---|---|
| 公开真人语音验证 | LibriSpeech/AISHELL-1各75条干净样本，另有噪声、babble和变速增强 | `r2_real_speech/` | **基本满足**；必须称“拼接的真人朗读语音”，不是自然对话 |
| TTFA/首个可听输出 | B组件预算14.79s；A预算22.67s | `r6_ttfa/` | **部分满足、P0阻塞**；当前不是严格闭合端到端测量 |
| 方差、尾延迟、重复性 | mean/std/P50/P90/P95/P99及50样本×3轮 | `r1_stats/` | **描述统计满足**；CV口径需修正，显著性检验尚缺 |
| 更强流式基线 | 同机498样本A/B/LA三方比较 | `r3_baseline_la/` | **满足**；须描述为修复后的同引擎LA-2变体 |
| 回滚与提交漂移 | rollback=0；内部correction 224次 | `r4_commit/` | **已测量，但原强主张被否定** |
| tokenizer边界 | 25/50 mismatch；50/50解码文本一致 | `r4_commit/tokenizer_seams*` | **已测量，但不能声称token/KV状态等价** |
| 下游语义影响 | cosine 0.8832；solo A/B 3.10/3.04 | `r5_semantic/` | **部分满足**；有限样本探索性证据，不能写“统计不可区分” |

---

## 3. CPU平台差异及论文可比性

### 3.1 已确认的影响

同一50个Very Long样本中：

| System B指标 | 原平台 | 第二平台 | 变化 |
|---|---:|---:|---:|
| TTFT | 981.2 ms | 1434.0 ms | +46% |
| ASR尾处理 | 894.3 ms | 1339.8 ms | +50% |
| LLM预填 | 86.8 ms | 94.2 ms | +9% |

证据：`experiments/results/revision/r1_stats/attribution/README.md`。

这表明第二平台的绝对时延主要受CPU对Whisper尾部解码和流式调度的影响，而非GPU配置变化或已知代码回归。

### 3.2 可以保留的跨平台结论

同一498样本的总体结果：

| 平台 | System A | System B | 改善率 |
|---|---:|---:|---:|
| 原平台 | 4503.1 ms | 1155.5 ms | 74.3% |
| 第二平台 | 5310.8 ms | 1573.9 ms | 70.4% |

因此可写：

> 两个平台均复现长语音TTFT的大幅相对改善，总体改善率约70%–74%；绝对时延受CPU性能影响。

### 3.3 论文红线

- 原平台：Table III、Table IV、Table V、Fig.6。
- 第二平台：真实语音、LA-2、TTFA、E1重复性绝对值。
- 不得跨平台混排毫秒值。
- 不得按比例缩放第二平台结果去“还原”原平台。
- 每个相对结论只能使用同机A/B或A/B/LA数据。
- “约1.1秒”只能绑定原平台；第二平台约为1.5–1.8秒。

---

## 4. 原论文结论影响

### 4.1 核心TTFT收益：成立

原平台Long/Very Long/Extra Long的System B均值约为1126.63/1099.16/1087.70 ms；第二平台同样本总体改善70.4%；干净真人语音Extra Long改善为：

- LibriSpeech：4805 → 1559 ms，67.5%；
- AISHELL-1：5140 → 1763 ms，65.7%。

原稿“最长分组减少5.67秒”应按成对过滤后的Table III改为约 **5.66秒**。

### 4.2 “TTFT与长度无关/固定上界”：需要收窄

原平台System B的P99为1979/2174/2605 ms；Long到Extra Long增长1.32倍，明显低于基线4.96倍，但并非严格不增长。第二平台均值也从1464.3增至1637.5 ms。

建议改成：

> 在已测试的长语音范围内，System B的TTFT增长显著弱于非流式基线，平均值呈平台区间，但尾延迟仍随输入长度缓慢增加。

### 4.3 “15秒是通用拐点”：需要收窄

原合成集在约15秒出现经验转折；第二平台干净真人语音Long组中，System B反而略慢：

- LibriSpeech：A 1654.8，B 1772.7 ms；
- AISHELL-1：A 1627.4，B 1658.7 ms。

应说明切换阈值依赖数据分布、CPU和分段参数，15秒不是通用阈值。

### 4.4 “流式ASR是主要收益来源”：强力支持

498样本重算：

| 分组 | Baseline | ASR-only | Full | ASR收益 |
|---|---:|---:|---:|---:|
| Long | 1690.81 | 1064.67 | 1087.69 | 626.13 ms |
| Very Long | 3307.96 | 1155.09 | 1152.36 | 2152.87 ms |
| Extra Long | 6515.67 | 1228.82 | 1188.00 | 5286.86 ms |

可以保留，但需限定为本文模型和硬件配置。

### 4.5 “KV收益随长度显著增加”：原结论不成立

| 分组 | ASR-only − Full |
|---|---:|
| Long | −23.01 ms |
| Very Long | +2.73 ms |
| Extra Long | +40.82 ms |

Extra Long独立收益约占ASR-only TTFT的3.3%，不是原稿中的9.3%。应改为：当前配置下KV增量预填独立边际收益较小，其主要设计价值是避免重复预填全部历史文本并将工作前移，而非贡献主要TTFT降幅。

### 4.6 “稳定提交”：必须拆成两层

50样本中：

- 375次commit事件；
- 425个已提交段；
- 224次内部correction；
- 224/425，即52.7%的已提交段后续发生重识别漂移；
- 49/50样本涉及漂移；
- rollback事件为0。

可写：

> 下游接口保持append-only，已发送文本不会撤回；内部后续ASR重识别仍可能修改同一音频段的假设，这些修订不会反向传播到已构建的下游缓存。

不得写“committed text is immutable”或“提交后内部文本不再变化”。

### 4.7 Tokenizer seam：测量完成，但预期被否定

- 25/50样本存在seam mismatch；
- mismatch样本平均4.36个差异块；
- 50/50解码回文本后完全一致。

只能证明文本层面的可逆一致，不能证明token序列、KV表征或生成概率完全等价。

### 4.8 语义影响：仅能支持“未观察到明显平均退化”

- BGE-M3 cosine mean：0.8832；
- 成对等价judge mean：2.96/5，≥4占40%；
- 独立意图满足：A 3.10，B 3.04。

没有配对检验、bootstrap CI、judge重复或多裁判一致性，不能写“statistically indistinguishable”。建议写样本均值接近，并披露50个Very Long合成样本、128-token上限、独立随机生成和DeepSeek裁判设置。

### 4.9 TTFT不能代表完整可听响应

当前组件预算中System B：

- endpoint：53.1 ms；
- E4 TTFT：1422.9 ms；
- 首token至首句：389.0 ms；
- TTS首包：12922.3 ms；
- 机械合计：14787.3 ms。

即便暂不考虑计时口径问题，TTS也占绝大部分延迟。因此原稿中把约1.1秒置于“全链路端到端”或人类对话间隙语境的表达必须删除。

---

## 5. 开发侧P0阻塞项

### P0-1：重新定义并最好重跑TTFA

#### 发现

`run_exp_latency.py` 中流式TTFT为：

```text
first_token_time - audio_end_time
```

而 `assemble_ttfa_budget.py` 又在E4 TTFT之前加上`T_endpoint`。E4 TTFT不是严格的“端点提交之后”分项，二者可能覆盖同一段等待。

同一次E5运行还给出：

- speech_end → first_token mean：3065.1 ms；
- final_enqueue → first_token mean：1012.5 ms；
- 当前表采用 endpoint 53.1 + E4 TTFT 1422.9 = 1476.0 ms。

三者不是同一时间轴的闭合分解。

#### 额外异常

E5的`final_speech_segment_commit_time - speech_end_time`有8/50个负值，约−439至−324 ms。原因是VAD闭段时记录commit，而`speech_end_time`在最后真实音频chunk完成500 ms sleep后记录，两者不是同一语义时间点。

#### 要求

推荐在同一请求、同一时间轴中记录：

1. physical speech end（最后一个非静音音频样本对应时间）；
2. VAD endpoint decision；
3. first LLM token；
4. first sentence boundary；
5. first non-empty PCM byte/chunk；
6. 直接计算 `TTFA = first_pcm_time - physical_speech_end_time`。

System A和B必须使用同一批样本直接测量，不再用字符数估计A的TTFC。

如果无法补跑，Table VIII必须降级为：

> Component-wise latency budget assembled from separately measured runs.

并且不能把B标为“全实测端到端TTFA”，不能把A 22.67秒用于可靠A/B端到端差值结论。

### P0-2：移除或验证System A的0.09 s/字符估计

当前装配脚本使用：

```text
A TTFC = 0.09 s/字符 × A回复字符数
```

但正式`tts_first_chunk.csv`的50条样本上，简单线性回归约为：

- slope ≈ −13.8 ms/字符；
- R²≈0.017；
- 字符范围99–260。

这批正式数据不能支持0.09 s/字符的正线性外推。开发侧需：

- 优先直接测量System A回复的TTFC；或
- 提供可复算的独立长度扫描数据、模型拟合和适用范围；否则删除A的TTFC/TTFA估计比较。

### P0-3：统一CV统计口径并重生成重复性摘要

当前4.2%/3.3%接近使用总体标准差`ddof=0`的结果。按三轮样本标准差`ddof=1`重算：

| 系统 | mean CV | median CV | max CV |
|---|---:|---:|---:|
| System B | 5.19% | 4.05% | 18.96% |
| System A | 5.23% | 4.65% | 14.01% |

开发侧需统一统计脚本（建议`ddof=1`），至少输出：

- mean CV；
- median CV；
- P90 CV；
- max CV；
- CV>5%的样本数和比例；
- 使用的标准差定义。

不得再写“三轮CV均小于5%”。

### P0-4：明确WER/CER是宏平均或补算corpus指标

`score_wer_offline.py`先计算逐样本WER/CER，再平均，因此现有数字属于mean utterance WER/CER，而非：

```text
sum(S+D+I) / sum(N)
```

开发侧应选择：

1. 表头改为`Mean utterance WER/CER`，方法中明确macro-average；或
2. 同时补算标准corpus WER/CER，并保存总S/D/I/N或等价可复算字段。

AISHELL终版应使用A CER 10.77%、B CER 11.80%；不得再引用`PAPER_HANDOFF.md`里的旧6.72%。

### P0-5：修正文档总册状态

在上述事项关闭前，修改：

- `experiments/results/revision/PAPER_WRITING_REFERENCE.md`
- 必要时同步`PAPER_HANDOFF.md`和`REVISION_CHANGELOG.md`

至少移除或标记：

- “全部数据格已定稿，可直接动笔”；
- “B全实测端到端TTFA”；
- “端点等待约0.05秒”的物理解读；
- “3轮CV<5%”；
- “统计不可区分”；
- 未说明宏平均的WER/CER表头。

---

## 6. P1强烈建议项

### P1-1：补充成对统计推断

无需GPU，可从现有逐样本数据计算：

- A/B、B/LA的paired bootstrap 95% CI；
- Wilcoxon signed-rank test或配对t检验（按分布选择）；
- 配对差值及改善率CI；
- 效应量。

避免只凭均值或分位数使用“statistically significant”。

### P1-2：真人语音QA说明

- 补计划中“抽5条人工试听”的样本ID、试听者/日期、结论记录；
- 明确数据是同章节/同说话人朗读句拼接，并插入人工静音；
- 解释`qa_transcribe.corrected.csv`中`reference`与`reference_full`的恢复和校验过程；
- 单独报告babble空输出率：LibriSpeech 12/30、AISHELL-1 5/30，不能用“error=0”暗示识别成功率100%。

### P1-3：LocalAgreement方法与排除规则

论文须说明最终实现包含：

- absolute audio timeline；
- sentence-boundary-aware trimming；
- `la_max_buffer_s=15.0`；
- punctuation-robust agreement。

498样本来自505个候选，排除3个运行错误和4个任一流式模式TTFT>10秒的样本。应披露阈值、成对排除和未过滤失败/挂起率。自动汇总必须白名单最终结果，避免纳入`invalid_dev3_frame_bug/`。

### P1-4：语义实验复现元数据

建议补记：

- BGE-M3模型revision/commit和tokenizer revision；
- judge完整模型ID/服务版本、prompt hash、时间戳；
- 输入与A/B回复hash；
- 评分调用顺序；
- `temperature=0`和128-token上限；
- 语义样本仅为50个Very Long合成样本。

当前结果适合作为有限探索性证据，不应泛化到所有长度、真人语音和噪声条件。

### P1-5：环境记录补齐

`env_versions.txt`有Python包和GPU信息，但缺少CPU/虚拟化原始记录。若GPU主机仍可访问，建议保存：

- `lscpu`；
- OS/kernel；
- 内存、NUMA和线程设置；
- CPU governor/frequency；
- TTS服务commit、模型revision和运行配置。

---

## 7. 其他方法风险（不一定要求重跑，但写作必须限定）

1. R2不同增强条件不是对完全相同的固定30条样本做变换；任意条件通常只重叠14–18条。因此单条件内部A/B比较成立，但跨条件差异混入了样本构成差异。
2. 除E1外，R2、LA、E4、endpoint、TTS和decode主要实验均只运行一次，E1重复性不能自动代表其他实验。
3. `full_response`仍受128-token上限限制，不能称无限制完整回复。
4. Fig.6的P5–P95阴影是观测分布带，不是均值置信区间或“P95误差带”。
5. GPU竞争实验未做，应在limitations中明确实验在独占GPU条件下完成。
6. 单机线程队列下可说明`T_Net≈0`，但不能据此忽略真实分布式部署网络延迟。

---

## 8. 开发侧验收标准

开发侧修改后，应提交新的回复文档并至少包含：

### TTFA

- 同时间轴字段定义与时序图；
- endpoint无负值，或对负值给出合理且可验证的定义；
- A/B同一批样本的直接TTFA，或明确降级为组件预算；
- 不再使用无数据支持的A TTFC线性估计；
- 汇总值能从逐样本原始时间戳复算；
- `T_endpoint + T_post + T_decode + T_TTFC`与直接TTFA的闭合误差有明确阈值。

### 统计

- CV使用统一标准差定义；
- 输出mean/median/P90/max及>5%比例；
- 主要配对比较提供CI和检验方法；
- 表格中的std定义一致。

### 质量

- 明确macro mean utterance与corpus WER/CER；
- 如报告corpus指标，保留可复算的编辑计数；
- AISHELL不再出现6.72%的旧值；
- babble空输出率单列。

### 文档

- `PAPER_WRITING_REFERENCE.md`与最终CSV/JSON逐项一致；
- 删除“统计不可区分”等无检验支撑表述；
- 删除“内部提交文本不变化”；
- KV边际收益采用498样本重算结果；
- 平台绑定和max_tokens口径完整。

---

## 9. 开发侧可直接使用的最终判断边界

若P0关闭，论文可以围绕以下边界修改：

> 本文提出的流水线流式架构能够将ASR和LLM预填充的大部分计算前移到用户发声期间，从而显著降低长语音的speech-end-to-first-token latency。该方向性收益在两个双RTX 3090平台、合成长语音和干净真人朗读语音上得到复现，总体改善约70%–74%，但绝对TTFT受CPU性能影响。消融结果表明主要收益来自流式ASR，当前模型和输入范围内增量预填充的独立边际收益较小。系统对下游输出保持append-only，但内部ASR重识别和tokenizer seam差异并不罕见。有限样本语义评测未显示明显的平均意图满足度下降。多人babble是当前方法的明确失败边界，而完整首个可听响应延迟主要受句段级TTS限制。

在P0关闭前，本报告的最终裁决保持为：**暂缓正式修改论文，先由开发侧修正TTFA计时/装配、CV、WER/CER及写作总册口径。**
