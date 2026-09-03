# 第六章 实验与结果分析

## 6.1 研究问题与实验设置

本章围绕以下研究问题展开。

- **RQ1：** 在固定被打断回复轨迹及固定自动检测器下，software-cursor playback 与 generation 历史边界的后续信息复现率有何差异？
- **RQ2：** 推测阈值如何影响作废计算与首候选 token 选择/内部计算就绪？实际墙钟口径与同步 oracle 接受口径分别给出什么结论？
- **RQ3：** 在受控同步文本段条件下，增量推测实现路径相对一次性预填充实现路径的候选计算就绪延迟有何差异？
- **RQ4：** 冻结模型与后端下，software-consumed-sample cursor 到 KV 裁剪及角色恢复的核心状态操作是否满足直接完整性合同，其模型侧成本和 prepared-state 软件控制路径时延如何？
- **RQ5：** 本次探索性运行中，朴素保留、打断标记和轻量重写三种实现的连贯性分数与重写耗时如何？

### 6.1.1 硬件、模型与数据

实验使用两张 NVIDIA RTX 3090（24 GB）。主模型为 Qwen2-7B-Instruct[11]；话轮检测器为 TEN Turn Detection[12]（7.6B）；历史重写模型为 Qwen3-0.6B[16]；自动裁判为 Mistral-7B-Instruct-v0.3[13]；TTS 时长画像由 CosyVoice2-0.5B[14]采集。裁判与主模型来自不同模型家族，但这不能消除单裁判和单提示词偏差。

数据由 MultiWOZ 2.1[15]派生。RQ2 与 RQ3 使用确认性 C-E1/C-E2 run `e1e2c_b8c758b_20260901T173306Z`：100 条唯一话语在 5 个独立初始化的进程 session 中重复，每个 session 执行 10 个条件，形成每条件 500 个 session×utterance 观测和 5000 条总记录。100 条话语是内容采样单位，5 个 session 是技术重复。RQ1 使用独立的固定轨迹 E3：100 条对话、400 个 `(dialogue, injection_label)` 配对、800 条条件记录及 1600 条自动裁判记录。RQ4 使用 C2 v3、联合 A1 和 prepared-state P1 三组独立工件。P1 覆盖 3 个上下文长度、3 个软件游标注入位置和每单元 20 次正式重复，共 180 条记录。

TEN 的标定成对正确率 1.00 来自 8 条手工完整句和 8 条手工不完整句形成的 64 个跨类对，仅用于确定阈值扫描范围，不代表独立口语测试集上的端点检测性能。

### 6.1.2 时序、播放边界与统计口径

C-E1/C-E2 使用确定性预切分文本段，不包含真实音频、ASR 墙钟、在线 TEN 前向、TTS、播放器或声卡。raw 字段 `first_token_ready` 的回调位于 token 选择之后、cache-update forward 与 generator yield 之前，因此本文统一称其为**首候选 token 选择/内部计算就绪**，不称可交付 token、consumer observation、TTS admission 或声学输出。`endpoint_accept` 是同步 harness 在候选生成后的 oracle 接受事件，并非自然端点检测输出。`TTFT_eff` 仅为同步 oracle 条件下的时延乐观下界，即推测收益上界。

C-E1/C-E2 的点估计来自完整、未加权的 5 session×100 utterance 网格。不确定性采用 crossed/product bootstrap：独立有放回抽取全局 session 与全局 utterance，再取二者笛卡尔积并重算估计量；重复 10,000 次，seed 为 20260901，报告 percentile 95% CI。该设计不把 500 个观测误作 500 个独立内容样本。

播放侧以 $p$ 表示 **software-consumed-sample cursor**；$\widehat{H}(p)$ 表示 TTS-fragment-level software retention boundary。二者均不等同于 device-presented samples 或 acoustically heard content。工件中的 `heard_text`、`n_heard` 和 `strict_unheard` 仅是兼容字段：分别对应片段保留或字符比例—空白吸附代理，不是人类听觉真值。

**表 6-1　主要证据的测量层级**

| 证据 | 测量层级 | 主要限制 |
|---|---|---|
| C2 v3 | 冻结 Qwen2-7B/BF16/SDPA/Transformers 下的直接 KV 裁剪完整性与匹配恢复 | 不检验 clean re-prefill 数值等价，不覆盖其他模型、后端、在线音频或生产系统 |
| C-E1/C-E2 | 同步文本段 harness 的候选选择/计算就绪及 oracle 下界 | 100 唯一话语×5 技术 session；无真实 ASR、在线 TTS、设备播放或生产可交付性 |
| 固定轨迹 E3 | 固定规则与单一 LLM 裁判条件下的信息复现率 | CI 仅含对话抽样不确定性；无检测器误差、提示词变化或人类感知误差 |
| 联合 A1 | 同步 GPU 的 crop+角色恢复微基准 | 固定执行顺序、固定移除 32-token suffix；不是完整打断路径 |
| P1 v2 | prepared-state、headless 软件游标控制路径 | 无 device/acoustic stop、在线 TTS、真实模块并发或 HCI 测量 |
| A2 | 受混杂探索性评分与重写耗时 | 三种实现的生成轨迹不一致，不支持处理效应解释 |

## 6.2 RQ1：固定检测器条件下的信息复现（E3）

固定轨迹 E3 使同一对话内的 playback 与 generation 条件共享被打断 assistant token 轨迹、断句时间轴和注入标签，后续 probe 使用 greedy 解码。主 estimand 为 label-weighted：每个符合资格的注入标签权重相同；差值统一定义为 generation 减 playback。点估计和区间均使用该 estimand，区间由 10,000 次 paired dialogue-cluster bootstrap 得到。

**表 6-2　固定轨迹 E3 的 label-weighted 信息复现率**

| 目标 / 检测器 | Playback | Generation | 差值 | 95% dialogue-cluster CI |
|---|---:|---:|---:|---:|
| 片段目标 / 词面规则 | 67.00% (199/297) | 63.64% (189/297) | −3.37 pp | [−10.49, 3.40] pp |
| 片段目标 / 自动裁判 | 42.76% (127/297) | 40.74% (121/297) | −2.02 pp | [−10.70, 6.13] pp |
| 字符比例—空白边界代理 / 词面规则 | 75.26% (286/380) | 73.68% (280/380) | −1.58 pp | [−6.08, 2.67] pp |
| 字符比例—空白边界代理 / 自动裁判 | 43.95% (167/380) | 41.32% (157/380) | −2.63 pp | [−8.57, 2.90] pp |

片段目标包含 297 个有效标签，来自 96 条对话；按目标、两条件历史、软件保留边界和轨迹精确去重后为 169 个语义组，即 128 个标签属于重复组。代理目标按自身非空性确定资格，共 380 个标签、100 条对话和 379 个语义组，仅移除 1 个重复标签。四个主区间均跨零，因此这些结果不能确定方向性优势，也不能推出差异不存在。

为检查标签权重和重复边界的影响，表 6-3 同时给出每条有效对话等权的 effect，以及每个唯一语义组等权的条件率与 effect。所有差值仍为 generation 减 playback。

**表 6-3　E3 weighting 与精确去重敏感性**

| 目标 / 检测器 | Dialogue-weighted effect [95% CI] | Unique-group Playback / Generation | Unique-group effect [95% CI] |
|---|---:|---:|---:|
| 片段目标 / 词面规则 | −3.21 pp [−9.55, 2.78] | 71.60% / 68.64% | −2.96 pp [−9.04, 2.63] |
| 片段目标 / 自动裁判 | −1.30 pp [−8.94, 6.08] | 43.20% / 43.20% | 0.00 pp [−7.98, 7.47] |
| 字符比例—空白边界代理 / 词面规则 | −1.50 pp [−5.75, 2.50] | 75.20% / 73.61% | −1.58 pp [−6.10, 2.69] |
| 字符比例—空白边界代理 / 自动裁判 | −2.58 pp [−8.25, 2.67] | 43.80% / 41.16% | −2.64 pp [−8.57, 2.90] |

所有 E3 区间仅表示：在冻结词面规则、`specific-reference-v3` Mistral 自动裁判、目标构造、固定轨迹、提示词和 40-token cap 条件下，由对话抽样产生的不确定性。区间不包含检测器误差、提示词或模型变化，以及人类感知误差。

### 6.2.1 构造检查与自动代理一致性

400/400 个 playback 条件记录在片段边界之后的局部完整未保留文本为空，对应局部规则阳性数为 0。该结果是 software-cursor retention 规则与指标共同定义的 implementation invariant check，不是语义效果或声学边界准确率。

词面规则与自动裁判在 label level 的合并一致数为：片段目标 370/594，代理目标 442/760；在 unique-group level 分别为 207/338 和 440/758。这些数值只描述两个自动代理之间的一致性，不能视为人工验证、检测器校准或 HCI 证据。

![图 6-1](figures/fig6_1.png)

**图 6-1　固定轨迹 E3 的效应区间示意。** 正式 label-weighted 点估计、匹配的 dialogue-cluster percentile 95% CI 及 weighting/dedup 敏感性以表 6-2 和表 6-3 为准。

## 6.3 RQ2：推测浪费与候选计算就绪（C-E2）

确认性扫描包含八个数值阈值和一个 never-speculate 对照，共九个工作点。每点由 100 条唯一话语×5 个独立进程 session 形成 500 个观测。0.92 是在新 holdout 结果揭示前冻结的 confirmatory candidate，不是部署最优阈值。浪费率定义为 pooled 的 $\sum W_i/(\sum W_i+\sum G_i)$。

**表 6-4　确认性九点扫描（每点 100 条唯一话语×5 个 session）**

| 阈值 $\theta$ | 0.0052 | 0.1979 | 0.3906 | 0.5833 | 0.7760 | 0.8500 | 0.9200 | 0.9688 | never |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pooled 浪费率 $\rho$ | 31.0% | 19.3% | 15.8% | 13.2% | 11.3% | 10.7% | **2.85%** | 0% | 0% |
| 候选存活率 | 100% | 99% | 98% | 97% | 96% | 84% | **67%** | 28% | 0% |
| oracle TTFT_eff 均值 (ms) | 0.0 | 0.3 | 0.7 | 0.9 | 1.3 | 5.0 | **10.3** | 22.4 | 31.1 |
| arrival→candidate selection/readiness 均值 (ms) | 62.4 | 62.4 | 62.3 | 62.1 | 62.2 | 62.0 | 62.4 | 62.3 | 62.4 |

C-E2 的 candidate-readiness 差值 `never − B@0.92` 为 −0.03 ms，crossed 95% CI [−0.64, 0.61] ms。oracle TTFT_eff 下界的差值为 +20.80 ms，[17.85, 23.65] ms。B@0.92 的 pooled waste 为 2.85%，[1.12%, 4.73%]；survival 为 67%，[58%, 76%]。因此该候选工作点刻画了同步 oracle 接受语义下的计算浪费与候选可用性，而不是生产可交付延迟改善。

在同一同步 harness 中，B@0.92 的 arrival→first-deliverable marker 均值为 257.58 ms，arrival→consumer marker 均值为 265.57 ms。二者受“先同步生成候选、再 oracle 接受”的执行顺序支配，只作为程序诊断，不作为生产系统 headline。约 291 ms 也仅是 candidate-first-selection 到 post-candidate oracle acceptance 的内部间隔中位数，不能解释为自然端点提前量或用户继续发言时长。

![图 6-2](figures/fig6_2.png)

**图 6-2　确认性九点扫描。** 左图对应同步 oracle 时延乐观下界，右图对应内部 candidate selection/readiness；右图竖线为各条件的 Q1–Q3 描述性离散范围，不是 crossed 95% CI。图中的“平坦”仅指九个 B-path 工作点的均值为 62.0–62.4 ms，不包含 System A。正式 crossed 差值区间以本节文字为准，两种口径不可互换。

## 6.4 RQ3：实现路径比较（C-E1）

C-E1 在每个 session×utterance 单元内配对比较 System A 的 full-string tokenization/full-prefill 路径与 B@0.92 的 segment-wise tokenization/incremental 路径。该比较不满足相同 tokenized context 条件，因此估计的是整体 implementation-path difference，而非单一增量预填充操作的效应。

**表 6-5　C-E1 实现路径配对结果**

| 指标 | System A | B@0.92 | A−B | Crossed 95% CI |
|---|---:|---:|---:|---:|
| arrival→candidate selection/readiness 均值 | 27.70 ms | 62.38 ms | −34.69 ms | [−35.44, −33.95] ms |
| oracle TTFT_eff 下界均值 | 27.70 ms | 10.26 ms | +17.44 ms | [14.41, 20.32] ms |

输出身份诊断进一步界定了这一比较：A 与 B@0.92 的完整 `output_token_ids` 仅 280/500 相同，首 token 为 465/500 相同，长度/EOS/max-token 状态为 495/500 相同；44/100 条唯一话语至少在一个 session 出现完整输出不一致，且每个 session 均为 44/100。相比之下，B@0.92 与 B-never 的完整 token、首 token、长度/EOS/max-token 及文本均为 500/500 相同，支持 C-E2 作为同一 B-path 内部比较。

C-E1 的差异混合了 full-string 与 segment-wise tokenization、forward topology/shape、角色边界、kernel 和 Python scheduling。主延迟分析不按 280 条完整输出一致记录筛选，因为该筛选位于实现路径之后。结果不得排他归因于某一次额外 forward，也不能从 oracle 下界反推真实音频或生产系统的墙钟改善。

![图 6-3](figures/fig6_3.png)

**图 6-3　C-E1 的双口径实现路径比较。** 柱高为条件均值，竖线表示各条件的 Q1–Q3 描述性范围，白色横刻度为中位数；图内方框才是配对均值差的 crossed 95% CI：candidate-readiness [−35.44, −33.95] ms、oracle 下界 [14.41, 20.32] ms。B 条件均为 B@0.92；该图比较非 token-equivalent implementation paths。

### 6.4.1 旧探索性 E1/E2 的口径审计

旧 E1/E2 的 0.581 ms 和 48.3→12.1 ms 将同步推测完成后的时间原点误作用户端点，属于 oracle 口径 artifact。其结果只保留为探索性审计和 0.92 候选来源，不作为正式墙钟证据。旧 mouth-to-ear 数值亦由模型计时与 6 句 TTS 画像组合建模，不是在线音频闭环测量。

## 6.5 RQ4：C2 核心正确性与支持性时延证据

### 6.5.1 C2 v3 direct crop-integrity addendum

正式接受的 C2 v3 run 为 `c2crop_82103004_20260903T080512Z`。协议固定 Qwen2-7B snapshot、BF16、SDPA、Transformers backend、24 个 case 和 27 个 crop event。308 个 fixture assistant token 均逐 token 走 production append；另含 3 个 no-op、60 个 recovery step 和 27 个 wrong-length negative control。

**表 6-6　C2 v3 核心正确性结果**

| 检查 | 结果 |
|---|---:|
| Case / crop event | 24/24；27/27 |
| 逐 token production append | 308 tokens |
| K/V 层数 | 28 |
| Recovery step | 60/60 |
| Wrong-length negative control | 27/27 检出 |
| No-op crop | 3/3 |

在该冻结网格中，每个 event 的 crop 前 retained K/V prefix、production `crop_to_token` 后 K/V 与独立 slicing oracle 在 28 层上逐张量 shape、dtype、device、hash 和 runtime `torch.equal` 均 bitwise exact；keep、attention mask、token ledger、sequence length 与 KV length exact 一致。使用相同 token-ID chunks 继续恢复后，K/V、logits、attention mask、token ledger、retained prefix 以及 role/end/content state 也 exact 一致。该结果只支持受测 snapshot/backend 下的 direct crop integrity 与 matched-recovery determinism。

v1 与 v2 clean-reprefill 协议均按冻结门槛 rejected。v2 的 24/24 termination probe 与 45/45 token/state/EOT/scenario 检查通过，但单控制 2× numerical gate 仅 42/45；由于 control 与 production forward topology 不匹配，三项失败既不能识别 crop 缺陷，也不能建立 clean-reprefill numerical equivalence。v3 是直接 crop-integrity addendum，不改变 v1/v2 verdict，也不支持 32-token continuation、跨模型/后端/硬件、在线音频或生产端到端正确性主张。

### 6.5.2 固定协议的联合微基准（A1）

A1 覆盖 256、512、1024、2048、4096 和 8192 token；每个长度执行 5 次预热和 50 次正式重复。各重复按固定而非随机化的 operation order 运行，并固定移除 32-token suffix。计时区间前后执行设备同步；联合路径在同一区间内执行 KV crop 与角色恢复。

**表 6-7　联合 crop+角色恢复与重新预填充的同步 GPU 微基准**

| 上下文长度 | 256 | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---:|---:|---:|---:|---:|---:|
| 联合路径中位数 (ms) | 31.616 | 31.852 | 31.054 | 31.519 | 36.903 | 48.315 |
| 联合路径 IQR (ms) | 2.356 | 2.162 | 3.099 | 1.197 | 0.635 | 0.928 |
| 重新预填充 / 联合路径中位数 | 2.254× | 4.124× | 7.707× | 15.020× | 25.453× | 40.620× |

![图 6-4](figures/fig6_4.png)

**图 6-4　固定 32-token suffix 协议下的联合微基准。** 误差线为 50 次正式重复的 IQR。

结果仅适用于固定执行顺序和固定移除长度，不代表自然打断位置、其他 crop length 或完整 barge-in。时间轴查询、软件停播、服务通信、线程调度和并发负载均不在 A1 窗口内。

### 6.5.3 Prepared-state 软件控制路径（P1）

P1 v2 在 512、2048、8192 token 与 0.25、0.50、0.75 三个 software-cursor 位置上各运行 20 次，共 9 个单元、180 条记录。120 条为片段内位置，60 条为片段边界。播放前屏障完成 setup 和 GPU 同步，并将其排除在 stop 路径之外。180/180 条 request 与 acknowledgement 精确命中目标 software-consumed-sample cursor，`leaked_samples=0`。

**表 6-8　Prepared-state P1 软件控制路径时延（每单元 n=20）**

| 计时区间 | 单元中位数范围 (ms) | 最大单元 empirical P95 (ms) |
|---|---:|---:|
| 软件停播确认 | 0.055–0.062 | 约 0.077 |
| 播放器确认后的 CUDA/GPU 同步 | 0.167–0.176 | 约 0.352 |
| 时间轴反查 | 0.47–0.50 | 约 0.94 |
| stop→crop 完成 | 2.44–2.53 | 约 3.492 |
| stop→角色恢复完成 | 78.6–80.8 | 约 86.1 |

每单元仅有 20 个值，empirical P95 主要由 1–2 个上尾观测决定，只作描述性次序统计，不代表生产 SLO。stop→crop 与 stop→角色恢复是从同一 stop 请求起点计算的嵌套累计区间，不能与组件中位数相加。P1 不测 device-presented 或 acoustically heard boundary，也不包含声卡/扬声器停止、在线 TTS 取消、真实 ASR/LLM/TTS/播放器并发、用户体验或生产端到端 barge-in。

## 6.6 RQ5：三种历史处理实现的描述性结果（A2）

本次 A2 每种实现包含 100 条记录。单一 Mistral 自动裁判的连贯性均值为：朴素保留 3.76、轻量重写 3.62、打断标记 3.29。重写调用耗时均值为 639 ms，中位数为 670 ms，线性插值 P90 约 935 ms，最大值为 1165 ms。

三种实现分别重新生成首轮与下一轮回复，只有 33/100 个对话的兼容字段 `heard_text` 在三条件完全相同，朴素与重写成对相同的仅 49/100。评分差异同时混入首轮内容、断句边界与下一轮采样差异，因而只能描述本次运行，不能解释为策略处理效应。约 0.64 s 的重写调用具备与下一轮用户发言并行的工程可能，但本实验未记录真实发言时长，也未测异步重叠，故不声称其耗时已被隐藏。

## 6.7 本章结论

1. **RQ1：** label-weighted 主分析及 dialogue-weighted、unique-group 敏感性分析的区间均跨零；结果仅是 fixed-detector-conditioned information-reproduction rate，不能确定方向性优势或差异不存在。
2. **RQ2：** 0.92 是预冻结 candidate。B@0.92 的 pooled waste 为 2.85% [1.12%, 4.73%]，survival 为 67% [58%, 76%]；oracle 下界相对 never 的差值为 +20.80 ms [17.85, 23.65]，candidate-readiness 差值为 −0.03 ms [−0.64, 0.61]。同步 deliverable/consumer marker 仅作诊断。
3. **RQ3：** C-E1 是 implementation-path comparison。A−B 的 candidate-readiness 差值为 −34.69 ms [−35.44, −33.95]，oracle 下界差值为 +17.44 ms [14.41, 20.32]；结果不能归因于单一 forward 或外推至生产交付。
4. **RQ4：** C2 v3 在冻结网格内建立了 28 层 K/V 的 direct crop bitwise integrity 与 matched recovery exactness。A1 和 P1 分别提供固定 32-token suffix 模型侧微基准及 headless software-cursor 控制路径的描述性时延，不提供设备、声学或 HCI 结论。
5. **RQ5：** 本次受混杂运行中三种实现的均值为 3.76、3.62 和 3.29，重写均值耗时 639 ms；这些是描述性结果，不是因果比较。
