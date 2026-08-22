# CISR 投稿前完整复审报告

- **稿件**：`main.tex`
- **题目**：*Latency Optimization of Cascaded Voice Dialogue Systems with a Pipeline-Parallel Streaming Architecture*
- **复审日期**：2026-08-22
- **模式**：`academic-paper-reviewer / full` 等效执行
- **说明**：当前会话中的 `/ars-reviewer` 未注册，直接调用返回 `Skill not found`；本报告依照本机 `academic-paper-reviewer` v1.11.1 的 full-mode 流程，由 Field Analysis、四个角色分离审稿席、固定 Devil's Advocate 和独立 Editorial Synthesis 完成。
- **操作边界**：只读审稿；未修改 `main.tex`、`refs.bib`、实验数据或实现代码。
- **校准状态**：`NOT_CALIBRATED`。
- **评审独立性说明**：五个审稿席分别提交意见且互不可见，但均继承当前会话模型；这是角色与调用上下文分离，不是统计意义上的独立误差过程。

---

## 1. 编辑决定

# **Major Revision（大修后重审）**

当前版本不建议直接投稿，但也不构成不可修复的 Reject。论文具有可信的系统实现、较宽的实验覆盖和少见的负面结果披露；然而，投稿前仍有三个科学阻断问题：

1. **系统实现、实验政策与论文中的因果归因不一致**；
2. **对话聚类、失败样本排除和样本构造尚未得到正确统计处理**；
3. **增量预填充、TTFA 和质量保持的贡献表述强于现有证据**。

核心延迟效应并未被敏感性检查完全推翻：过滤后的 498 对样本平均降幅为 74.34%，纳入全部具有数值的 503 对样本后仍为 61.59%。问题不在于“完全没有效果”，而在于当前标题性数字、尾部可靠性和机制归因没有被同等严谨地限定。

对于 broad-scope CISR，本文可定位为**流式智能系统及系统政策的实证研究**。不要求以完整 HCI 用户研究作为录用前提，但必须删除或收窄未经用户实验支持的 usability、体验改善和部署普适性结论。

---

## 2. 本轮独立核验结果

### 2.1 LaTeX、引用和 PDF：本地机械检查通过

在隔离临时目录完整执行：

```text
pdflatex → bibtex → pdflatex ×2
```

结果：

- 13 页，Letter，IEEE conference 双栏；
- 28 个唯一 citation key，`refs.bib` 28 条，**28/28 闭合**；
- 无 missing key、orphan entry、duplicate key；
- 所有图表均有正文交叉引用；
- 无 undefined citation/reference；
- 无 overfull box；
- 字体全部 embedded/subset；
- Type 3 字体数量为 0；
- PDF Title、Author、Subject、Keywords metadata 均存在；
- 仅有若干 `Underfull \hbox/\vbox` 警告，不构成提交阻断。

因此，先前已完成的引用和 Figure 6 修复有效。当前科学审稿问题不是 citation closure 或 PDF 字体问题。

### 2.2 新发现：ASR 识别时长触发器的实现语义与论文描述不一致

论文将触发条件描述为当前临时队列的累计时长达到 `recognition_threshold`，见 `main.tex:228-249`。源码实际行为为：

- `total_duration` 初始化为 0：`src/asr/faster_whisper_streamer.py:136`；
- 新等待片段并入时递增：`src/asr/faster_whisper_streamer.py:144-150`；
- 移除队首片段时只执行 `pop(0)`，**未递减 `total_duration`**：`src/asr/faster_whisper_streamer.py:152-157`；
- `should_process()` 仍以该值判断 2 秒门限：`src/asr/faster_whisper_streamer.py:175-191`；
- 已处理片段会在识别后被移除：`src/asr/faster_whisper_streamer.py:587-607`。

Git 历史表明该行为自提交 `3ee6157`（2025-12-04）起一直存在；R7 记录的运行提交 `c9437c3a4a69c58f7ea714c72af2df6db6ec7a97` 也具有相同逻辑。因而它不是后续代码漂移，而是实验实际测量的政策：

> **累计已见音频时长门第一次跨过 2 秒后锁存；随后识别触发主要由队列片段数和 final 条件决定。**

这不意味着历史测量是伪造或不可用；它们是实际实现的有效观测。但论文不能继续将其描述为“每轮当前队列累计达到 2 秒”，也不能声称它与按新增音频时长触发的 LA-style baseline 完全 matched。修复递减逻辑后的算法行为不能由现有数据直接推断。

### 2.3 失败样本敏感性

`experiments/results/exp2_ablation/exp2_gains_exclusions.csv` 记录 7/505 个排除样本：

- 3 个 HTTP 504 runtime errors；
- 4 个 streaming TTFT > 10 s；其中数值包括约 11.1 s、61.3 s、65.9 s 和 168.1 s。

独立重算：

| 分析集 | n | System A mean | System B mean | 平均降幅 | System B 最大值 |
|---|---:|---:|---:|---:|---:|
| 当前过滤集 | 498 | 4503.14 ms | 1155.51 ms | **74.34%** | 3228.51 ms |
| 所有 baseline/full 数值完整样本 | 503 | 4515.65 ms | 1734.27 ms | **61.59%** | **168089.06 ms** |

结论：正向均值效应仍存在，但幅度和尾部风险明显依赖排除政策。74.3% 不能继续作为唯一 headline，而应与 failure-inclusive 结果、完成率和失败类别并列。

### 2.4 对话聚类敏感性

主 1,132 样本来自约 100 个源对话；498 样本子集来自 99 个源对话。累计 turn 共享大量前缀，不是 498 个独立对话。

对过滤后的 498 对样本，按 dialogue cluster 重采样：

| 方法 | 74.34% 降幅的 bootstrap 95% CI |
|---|---|
| 当前样本级 bootstrap 等价敏感性检查 | [72.94%, 75.61%] |
| 对话簇 bootstrap | **[72.65%, 75.76%]** |

大效应在这一敏感性检查下仍存在，但论文仍需以 dialogue 为主要重采样/推断单位，报告唯一对话数、cluster size 和 cluster-aware CI/p 值。

### 2.5 TTFA 输出政策与语言核验

R7 主分析的 100 条 primary records 显示：

| 指标 | Non-streaming | Streaming |
|---|---:|---:|
| TTS 输入文本平均长度 | 204.10 字符 | **17.74 字符** |
| 简单问候（“你好！”/“您好！”） | 0/50 | **25/50** |
| 达到 128-token cap | 41/50 | 43/50 |

进一步：

- 对 25 条英文输入的 streaming 记录，**19/25 的 TTS 文本不含拉丁字母**；
- 因此 Table VIII 的 “Language” 是**输入语言**，不能被读作 TTS 输出语言；
- `platform_conditions.txt:61-64` 显示 TTS 与 ASR 同驻 GPU0，LLM 在 GPU1；论文当前“实验期间无其他 GPU 作业”不能被误读为 ASR GPU 上没有 TTS 共驻负载。

TTFA 的 22.27 s → 3.11 s 是可复现的**系统政策组合结果**，但不是输出文本、停止条件和 TTS 输入边界匹配后的架构因果效应。

---

## 3. Reviewer Configuration

| 席位 | 专家身份 | 主要职责 |
|---|---|---|
| Journal-Fit | 智能系统与语音计算方向的 conference area chair | CISR fit、贡献定位、标题/摘要/结论一致性、篇幅与呈现 |
| Reviewer 1 — Methodology | 实验系统与统计方法审稿人 | 设计、统计单位、排除、执行顺序、可复现性和因果归因 |
| Reviewer 2 — Domain | Streaming ASR、spoken dialogue 与 LLM inference 专家 | ASR 触发/提交、KV cache、tokenizer seam、baseline 与文献新颖性 |
| Reviewer 3 — Perspective | Spoken HCI/HRI、无障碍和生产部署评估者 | 生态效度、用户含义、资源、可靠性、语言与部署边界 |
| Devil's Advocate | 固定反方席 | 构造最强拒稿论证，检验 headline 是否由证据支持 |

领域判定：speech/LLM systems、streaming ASR、LLM serving 与智能语音交互交叉领域；方法属于 artifact-building + controlled empirical systems evaluation。CISR 的 intelligent systems/NLP 范围具有合理匹配，robotics 仅是应用背景而非已验证对象。

---

## 4. 五席评审摘要

## 4.1 Journal-Fit Reviewer

**建议：Major Revision / borderline weak reject；置信度 0.89。**

### 主要优点

- 实验覆盖较宽：两平台、重复测量、human-read speech、扰动、LA-style baseline、TTFA 和失败分析；
- 对负面结果披露充分；
- 系统在长语音上移动 post-speech critical-path work 的工程观察具有价值；
- 本地 IEEE 格式、引用和 PDF 质量良好。

### 决策相关问题

1. **贡献叙事强于消融。** Streaming ASR 提供 626–5287 ms，而 incremental prefill 仅为 −23、2.73、40.82 ms（`main.tex:425-444`）。论文更适合定位为 scheduling/interface design，而非新 KV-cache 算法。
2. **TTFA 混合输出政策。** 摘要和结论未在数字出现处立即说明 first-sentence 与 full-response 的差别（`main.tex:45-46,528-562,568-570`）。
3. **13 页会议稿过长且教程化。** Whisper、generic attention、naive cubic decoding 和标准 KV-cache 原理占用较多篇幅（`main.tex:119-166`）。
4. **CISR fit 应通过 intelligent voice systems 表述，而不是暗示已验证机器人部署。**
5. **标题过宽。** 建议强调“post-utterance latency”与“streaming ASR–LLM scheduling”。
6. 图表和 Algorithm 1 偏密，部分 caption 不够自解释。

## 4.2 Reviewer 1 — Methodology

**建议：Major Revision；置信度 0.94。**

### 主要优点

- TTFT/TTFA 定义和统一时钟分解较清楚；
- 对 append-only、internal drift 和非 KV equivalence 的边界表述较严谨；
- 使用 paired bootstrap、Wilcoxon、Holm 和重复测量；
- 负面结果及外部效度限制披露较好。

### 决策相关问题

1. **统计单位错误。** 累计 turn 聚集在约 100 个源对话，现有推断却按 turn 独立处理（`main.tex:321-381,186-195`）。
2. **样本构造文字与代码不完全一致。** 实现先选择最长 source dialogues，再生成多个累计 turn；稿件写成选择最长 fragments，且没有给出完整版本、split、对话数和每对话样本数。
3. **失败条件化。** 498/505 的 headline 排除了最不利于 streaming 的长尾和 runtime failures（`main.tex:425-428`）。
4. **legacy arm 固定顺序。** 主要 TTFT 为 B→A，三臂 ablation 为 Baseline→ASR-only→Full-streaming，未 counterbalance。
5. **TTFA 政策不匹配。** 大部分差异来自 first-token→text-ready 与 TTS 阶段，而非纯 ASR/prefill。
6. legacy TTFT 是 waveform feed completion 后的 residual latency，不是统一的 physical speech-end metric。
7. KV marginal contrast 缺配对 CI 和 cluster-aware uncertainty。
8. 速度没有与 prespecified WER/CER 或任务质量 noninferiority 联合评估。
9. 原平台模型、软件、prompt、dtype/backend 和 artifact 绑定不足。

## 4.3 Reviewer 2 — Domain

**建议：Major Revision；置信度 0.93。**

### 主要优点

- 对 append-only downstream 与 internal stability/KV equivalence 的区别处理正确；
- 消融坦诚报告 prefill 的边际效果很小；
- latency endpoint 清楚；
- babble、internal drift、tokenizer seams 和 cap-hit 均未隐藏。

### 决策相关问题

1. **ASR `total_duration` 锁存问题。** 经独立复核确认，方法描述和 baseline-matching 结论不准确；这是本轮最重要的新发现。
2. **TTFA policy confound。** 不能把全部 86% 归因于 ASR/prefill 架构。
3. **相关工作不够贴近 2026 状态。** 应补充并比较 Simul-Whisper、SimulStreaming、Prompt Cache、SGLang/RadixAttention 等直接先例。
4. **LA-style 只有一个定制 operating point。** 结果只能说明本文配置快于该配置，不能概括 LocalAgreement 方法族。
5. **不可逆提交的稳定性不足。** 224/425 committed segments 后续发生变化，需报告 emission-time error 与语义类别。
6. **tokenizer seam 未解决。** 25/50 token sequence 不同；当前 50 样本 embedding/judge 实验不能证明影响无害。
7. Complexity 段应把 `O(M(N+M))` 表述为在 `M << N` 下约为 `O(MN)`，并删除“保存末步 logits 是 incremental prefill 独有优势”的暗示。
8. “adaptive sliding window” 过强；当前实际是 fixed-context/VAD-driven overlapped window。

## 4.4 Reviewer 3 — Cross-Disciplinary Perspective

**建议：Major Revision / weak reject；置信度 0.92。**

### 主要优点

- closed timing decomposition；
- 负面结果和边界披露较完整；
- 增加 human-read speech、噪声和语速；
- state-correctness 报告对生产可靠性有价值。

### 决策相关问题

1. TTFA 是 server-side first-playable-output proxy，不是已验证的 usability outcome。
2. first-sentence vs full-response 政策不匹配。
3. 52.7% committed segments 后续发生内部变化，但没有按 entity、number、negation、intent 或 task action 分类。
4. 累计对话和拼接 read speech 是 stress workloads，不是自然交互分布。
5. 样本极度偏向 Extra Long，而系统在 Short/Medium 存在回归。
6. 仅英语/普通话不足以支持 broad multilingual/accessibility 外推。
7. 双 RTX 3090、batch 1、exclusive laboratory setting 缺少资源、吞吐和多用户数据。
8. 应报告 failure-aware reliability、fallback policy 与 server/client latency 边界。
9. 隐私、tenant isolation、cache lifecycle 和 routing fairness 尚未讨论。

**编辑裁决**：不采纳“必须完整 HCI 用户研究才能投稿”的要求。若论文定位为 systems paper 并收窄 usability claim，则用户研究可列为未来工作；但必须把 TTFA 称为 server-side metric，并删除已证明用户体验改善的暗示。

## 4.5 Devil's Advocate

### 最强反方论证

论文自己的消融表明总体收益主要来自 streaming ASR，而不是 incremental prefill；长、嵌套、选择性压力工作负载又为“在说话期间隐藏工作”提供了最大重叠窗口。TTFA 比较让 Pipeline arm 合成平均 17.74 字符、且一半只是简单问候，而 Serial arm 合成平均 204.10 字符，因此 22.27→3.11 s 大部分是不同 response/TTS policy 的结果。与此同时，49/50 样本存在 post-commit drift，clean English WER 上升，babble 可出现空输出和约 21 s queue drain，四个灾难性流式尾部又被排除。最强可支持结论是：**当前 online Whisper scheduling 对长、较有利的输入可显著减少 post-feed residual latency，但不能证明一种普适、质量匹配、可靠的 voice-dialogue latency optimization。**

### 三项 CRITICAL 裁决

| DA CRITICAL | 裁决 | 后果 |
|---|---|---|
| Incremental prefill 没有有意义的独立增益 | **成立** | 是 framing/novelty blocker；可通过收窄贡献或新增匹配消融修复 |
| TTFA 政策不匹配 | **成立** | 阻断 architecture-level causal attribution；仍可作为 policy bundle 结果 |
| 排除灾难性 streaming failures | **成立** | 必须 failure-inclusive 报告；但 503 样本重算仍为正向 61.59%，没有彻底抹除效果 |

---

## 5. 三个科学阻断项

## B1. 实现、政策与因果归因不一致

### 必须修复

- 准确写出实际的锁存触发语义；
- 删除 current-queue 2-s trigger 和完全 matched trigger 的错误表述；
- 对修正递减逻辑后的实现做**定向同样本复跑**，至少报告 latency、failure rate 与 quality；
- 若不复跑，则把论文限定为对 historical implementation 的测量，不能推断 intended trigger 的性能；
- TTFA 在摘要、正文、表题/表注和结论中统一称为 **policy-level, server-side comparison**；
- 不把 22.27→3.11 s 归因于 incremental prefill。

### 是否要求全量重跑

**不机械要求把所有历史实验全部重跑。** 原数据测量的是实际实现，仍可报告。投稿前最低需要：

1. 修正触发器单元测试；
2. 在固定、代表性的同样本 paired subset 上比较 original-latched 与 corrected-current-queue policy；
3. 如果主要结论变化明显，再扩大重跑范围；
4. 任何未重跑表格必须明确标注为 historical-latched implementation。

## B2. 聚类、失败排除与样本账本

### 必须修复

- dialogue-level cluster bootstrap/检验；
- 每个表同时报告 accumulated-turn `n` 和 unique-dialogue `n`；
- 报告 cluster-size 分布摘要；
- 说明 505 candidate、503 numeric pairs、498 filtered pairs 与 2 个无完整数值样本；
- 以 failure-inclusive 结果为主，或与过滤结果同等醒目并列；
- 报告 completion/error/timeout/empty-output rate、worst case 和 tail；
- 将固定执行顺序列为 bias，最好做 reversed/counterbalanced subset；
- 使样本构造描述与代码完全一致，并将其称为 purposive long-input stress benchmark。

## B3. 贡献归因和质量有效性

### 必须修复

- 把 incremental prefill 从“主要已证明加速来源”调整为“架构组件”；
- 若保留其独立贡献主张，必须给出 paired/cluster-aware CI，并解释 Long 组负收益；
- LA-style 结论改为“faster than our configured LA-2-style implementation”；
- 报告 latency–quality trade-off，至少提供 paired WER/CER difference、empty-output/failure rate 和 committed-text quality；
- 对 50 样本 semantic analysis 保持 exploratory，不以 0.883 证明 equivalence；
- 如果无法增加任务级评估，删除“质量/任务效果保持”的主张，而不是把缺失结果解释为等价。

---

## 6. 必需修订路线图（按稿件顺序，不代表工作优先级）

### R1 — 标题、摘要和关键词

- 收窄题目，例如：
  - *Reducing Post-Utterance Latency in Cascaded Voice Dialogue Systems through Streaming ASR–LLM Scheduling*。
- 在摘要的 TTFA 数字同一句中说明 first-sentence vs full-response policy。
- 删除 abstract 中不易解释的 0.883 cosine，或同时给出“exploratory, not equivalence”限定。
- 将 HCI 作为应用语境而非已验证效果；可考虑删除 `human--computer interaction` keyword，或在正文明确无用户研究。

### R2 — Introduction 与 contributions

- 在第一页直接陈述研究问题、实际 gap 和主要机制；
- 把贡献改为：
  1. modular cascade 的 streaming scheduling/interface；
  2. append-only state contract 与 failure characterization；
  3. 多工作负载、failure-aware、policy-level evaluation；
- 明确 large gain 主要来自 streaming ASR，incremental prefill 在 Qwen2-7B 下边际小。

### R3 — Related Work

- 补充 Simul-Whisper、SimulStreaming、Prompt Cache、SGLang/RadixAttention；
- 区分 StreamingLLM attention-sink 与本文 growing full cache；
- 最好增加 feature-comparison paragraph/table：windowing、agreement、timestamp commit、incremental prefill、rollback、TTFA。

### R4 — Methods 与实现语义

- 修正 ASR trigger 描述；
- 把 “adaptive” 改成 fixed-context 或 VAD-driven overlapped；
- 将 legacy TTFT 改称 post-feed residual TTFT；
- 将 TTFA 改称 server-side first-playable PCM；说明 1324 bytes = 22050 Hz × 16-bit × 30 ms；
- 说明 TTS 与 ASR 共驻 GPU0；
- 说明 fixed arm order；
- 修正复杂度与 last-logit 叙述；
- 采用半开时间区间或明确 boundary tie-break，避免相邻 segment 同时包含边界 word。

### R5 — Data construction

- 给出 corpus version/split、所选 dialogue 数、ranking rule、每对话累计 turn 数、manifest/hash；
- 正确说明以前序 user/system turn 构造单一 TTS 输入；
- 将 synthetic 和 concatenated speech 称为 controlled stress workloads；
- 报告 duration distribution，并强调整体结果取决于 duration mix。

### R6 — Statistics 与 failure-aware reporting

- 按 dialogue cluster 重算主 CI/p 值；
- 报告 n=498 与 n=503 两组结果；
- 给 KV marginal effect 配对 CI；
- 为 median/quantile difference 给不确定性，n=25 的 P95 只作 descriptive；
- 枚举 Holm family 和 ratio-of-means estimand；
- 加入 completion/error/timeout/empty-output 表。

### R7 — TTFA

最低可接受路径有两种：

1. **更强路径**：做 2×2 factorial：Serial/Pipeline × first-sentence/full-response，并尽可能 matched TTS text；
2. **最小路径**：保留现有实验，但将其严格标为 policy bundle，披露输出长度、问候比例、输入/输出语言错位、cap-hit 和 GPU 共驻，不作 architecture-only attribution。

### R8 — Quality、state correctness 与 baseline

- 报告 emission-time committed WER/CER 或 word/token revision rate；
- 按 entity、number、date/time、negation、intent 分类 drift；
- 用相同 decoding policy 或 deterministic decoding 比较 seam/no-seam response；
- 如资源允许，增加 LA operating points 和 latency–quality frontier；否则收窄结论。

### R9 — Discussion、Limitations、Conclusion

- 明确 short/medium regression、babble、catastrophic tails、half-duplex、no client playback、no contention test；
- 加入 privacy/cache lifecycle、routing fairness、accessibility 和 multi-tenant 边界的简短说明；
- 结论同时报告 filtered 74.34% 与 failure-inclusive 61.59%；
- 说明 cluster sensitivity 未推翻大效应，但 corrected trigger 尚未测量；
- 不声称用户体验、任务成功或生产可靠性已验证。

### R10 — 篇幅与呈现

- 压缩 generic Whisper/attention/KV tutorial 1.5–2 页；
- 保留贡献特定的 topology、window、state 和 result figures；
- 提高 Fig. 4/5、Algorithm 1 和 Tables IV–VIII 在 100% 缩放下的可读性；
- 使用 `Serial`/`Pipeline` 替代需反复查找的 A/B；
- 说明 Fig. 1 是否 author-created/adapted；若只是 generic Whisper diagram，建议删除。

---

## 7. 建议但不单独阻断的增强项

- counterbalanced/reversed-order subset；
- corrected trigger 与 original-latched trigger 的 staged sensitivity run；
- resource metrics：VRAM、GPU-seconds、RTF、ASR invocation count、processed audio seconds、throughput；
- matched first-informative-phrase latency，而不是 first bytes/greeting；
- task slot/intent correctness；
- spontaneous speech 小样本；
- client-side playback timestamp；
- 多 judge 或 human rating；
- secure cancellation/queue draining/cache deletion policy。

---

## 8. 投稿流程门槛

以下不是科学拒稿理由，但投稿前仍必须完成：

- **Similarity check**：全文 ≤24%，单一来源 ≤3%；
- **IEEE PDF eXpress**：对最终、修订后的 PDF 执行；
- 若 PDF eXpress 改变分页，重新核对 response letter 页码；
- 最终 LaTeX ZIP 不包含内部审计报告、构建中间文件或无关实验材料。

---

## 9. 最终判定表

| 维度 | 判定 |
|---|---|
| CISR intelligent-systems scope | PASS，需弱化 robotics/HCI 已验证暗示 |
| 本地 LaTeX/PDF 合规 | PASS |
| 引用机械完整性 | PASS（28/28） |
| 引用/相关工作充分性 | PARTIAL，缺 2024–2025 直接先例 |
| 核心长输入延迟方向 | SUPPORTED，但幅度对 failure handling 敏感 |
| 对话聚类后的过滤集效应 | SUPPORTED（敏感性 CI 仍为大正效应） |
| Incremental prefill 独立大收益 | NOT SUPPORTED |
| TTFA architecture-only attribution | NOT SUPPORTED |
| Historical implementation 的测量有效性 | SUPPORTED |
| 论文所写 current-queue trigger | IMPLEMENTATION MISMATCH |
| Quality/task noninferiority | NOT ESTABLISHED |
| Failure/tail robustness | NOT ESTABLISHED；已有反例 |
| 当前科学投稿就绪度 | **NOT READY — MAJOR REVISION** |
| PDF eXpress | PENDING |
| Similarity check | PENDING |

## 10. 重审通过标准

满足下列条件后，稿件才适合再次进行投稿就绪判断：

1. ASR trigger 的实际语义被准确记录，并完成至少一个 corrected-trigger 的配对定向复跑或明确降格主张；
2. TTFA 不再被错误归因于 architecture/prefill，或新增 policy-matched 对照；
3. 对话聚类推断、failure-inclusive 结果和完整样本账本进入主文；
4. incremental prefill、LA-style superiority、quality 和 usability 主张与证据严格对齐；
5. 标题、摘要、贡献和结论统一收窄到实际实现与测试边界；
6. 修订后的最终 PDF 再通过本地预检、similarity check 和 IEEE PDF eXpress。
