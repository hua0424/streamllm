# 开发整改方案复审与补充要求（2026-08-21）

- **复审对象**：`dev-assessment-and-plan-20260821.md`
- **对应初审**：`review-20260821-PRE-PAPER-AUDIT.md`
- **目的**：判断开发整改方案执行后能否满足论文及审稿回复需要，并明确实施前必须补足的定义和验收条件。

---

## 1. 总体裁决

**有条件批准整改方案进入实施，但不建议按当前文本直接启动 W1 正式 GPU 实验。**

开发侧已经正确接受了主要审计事实，以下方向是正确的：

1. 重新执行 A/B 配对 TTFA，废弃旧 Table VIII 的跨运行机械装配；
2. 不再用 `0.09 s/字符`估算 System A 的 TTFC；
3. CV 改用 `ddof=1`并报告完整分布；
4. 区分 mean-utterance 与 corpus WER/CER；
5. 增加配对 bootstrap、Wilcoxon 和效应量；
6. 补充环境、语义评测和数据 QA 元数据；
7. 在整改完成前将写作总册降级为“整改中”。

但是，当前方案仍存在一个核心阻塞和若干覆盖缺口：

- **W1 的 endpoint、时间轴和 first-audio 定义还没有闭合**；按现方案跑出的数据仍可能无法称为“直接测量的首个可听输出延迟”。
- P1-3 LocalAgreement 方法与排除规则完全未纳入工作项。
- 真人语音 QA、语义元数据和总册同步只覆盖了部分要求。
- 统计检验只列出了方法名称，尚未冻结可复算协议。

因此，本报告把要求分成：

- **实施前必须澄清（阻塞 W1 正式运行）**；
- **实施中必须完成（阻塞论文数据放行）**；
- **建议增强（资源允许时做，不作为本轮硬阻塞）**。

---

## 2. 对开发工作项的逐项裁决

| 工作项 | 当前判断 | 是否可直接实施 |
|---|---|---|
| W1 统一时间轴 TTFA | 方向正确，但事件定义不足，endpoint 被换成 feed wait，A 的分项不闭合 | **否，先按第3节补充协议** |
| W2 环境记录 | 基本充分 | **可以**，按第6.5节补少量指纹 |
| W3 CV 重算 | 基本充分 | **可以**，补公式和聚合规则 |
| W4 macro/corpus WER/CER | 概念正确 | **可以**，但须分别保存 WER/CER 编辑计数 |
| W5 配对统计推断 | 方法方向正确，协议未冻结 | **补齐参数后可以** |
| W6 语义元数据 | 大部分覆盖 | **可以**，补 tokenizer revision、调用时间和样本范围 |
| W7 真人语音人工试听 | 只覆盖 P1-2 的一部分 | **可以实施，但需扩大记录内容** |
| W8 写作总册同步 | 两阶段策略正确，清单不完整 | **可以实施，须扩大同步范围** |
| P1-3 LocalAgreement | 原方案遗漏 | **必须新增工作项** |

---

## 3. W1 实施前必须补充的 TTFA 协议

这是本轮唯一必须在正式 GPU 运行前解决的设计问题。

### 3.1 不得再使用“同一请求 A/B”表述

System A 和 System B 是互斥管线，无法在一个请求实例中同时运行。正确描述是：

> 同一批样本、相同输入音频和配置下，分别执行的配对 A/B 请求。

为避免固定执行顺序带来的热缓存、温升和系统漂移偏差，应预先生成确定性的平衡顺序：

- 25条执行 A→B；
- 25条执行 B→A；
- 按语言和时长分层平衡；
- 顺序表、seed 和 hash 落盘。

最低验收：每个 `sample_id × repeat_idx` 恰好有一条 A 和一条 B，输入 WAV SHA-256 一致。

### 3.2 必须区分三个不同概念

开发方案把 `feed_end - physical_speech_end`作为 endpoint 分项，这不能等同于 VAD endpoint latency。必须明确区分：

1. `source_trailing_silence`：源 WAV 中物理语音结束后的尾静音；
2. `trailing_feed_wait`：物理语音结束至输入播放/推送结束；
3. `endpoint_decision_latency`：物理语音结束至 VAD/系统实际作出结束决策。

如果系统最后依赖文件结束或显式 flush，而没有在线 VAD endpoint decision，应如实命名为 `explicit_flush_time`，不能包装成 VAD endpoint。

原 E5 的负值不能通过把指标改名为天然非负的 feed wait 来宣称“endpoint 已修复”。论文中是否报告 endpoint latency，取决于是否真的记录了 endpoint decision。

### 3.3 冻结 physical speech end 的可复算定义

开发方案中的“能量门限 + Silero VAD 复核”还不足以复现。正式运行前须固定：

- 使用的音频采样率；
- 分析窗口和 hop；
- 能量指标及阈值；
- Silero 版本、threshold、padding/min-speech/min-silence参数；
- 两种方法冲突时的裁决规则；
- 输出 `estimated_physical_speech_end_sample`；
- 源 WAV SHA-256；
- sample index 到单调时钟的映射公式。

建议原始区间使用 `time.perf_counter_ns()`，UTC墙钟只用于RUNINFO，不参与延迟计算。

如果源 WAV 自带尾静音，则 `feed_end - physical_end`完全可能超过500 ms。因此取消“必然≤500 ms”的验收条件，改为分别报告源尾静音、chunk量化等待和调度误差。

### 3.4 每条 A/B 至少保存以下原始事件

```text
clock_type
playout_start_ns
physical_speech_end_sample
physical_speech_end_ns
last_input_sample_ns
feed_end_ns
endpoint_decision_or_flush_ns
last_asr_commit_ns
first_token_ns
first_sentence_boundary_ns
generation_end_ns
tts_request_start_ns
tts_response_headers_ns
first_pcm_byte_ns
first_playable_pcm_ns
tts_done_ns
```

其中：

- System A 必须记录 `generation_end_ns`，因为它等待 capped full response 后才调用 TTS；
- A/B 都必须记录真实的 `tts_request_start_ns`；
- 不得假设检测到句末和TTS请求发出是同一时刻；
- “完整回复”必须注明是 `max_tokens=128`约束下的 capped response，并保存 `stop_reason=eos|max_tokens|error`。

### 3.5 直接指标和组件分解

主要指标直接定义为：

```text
TTFA_received = first_pcm_byte_ns - physical_speech_end_ns
TTFA_playable = first_playable_pcm_ns - physical_speech_end_ns
```

论文主表优先使用 `TTFA_playable`；若无法可靠判定可播放音频，则降级使用 `TTFA_received`并称：

> speech-end-to-first-received-PCM latency

不得称“first audible”。

A/B组件可以使用不同的实际边界，但每项必须由连续时间戳差分构成。例如：

- 输入尾部/端点或flush；
- 端点/flush至first token；
- first token至TTS文本就绪；
- TTS文本就绪至请求发出；
- 请求发出至first playable PCM。

组件之和必须与直接TTFA由同一组边界严格闭合。若全部使用连续的同一组`*_ns`事件，残差应仅为数值舍入误差；原方案的固定`|Δ|≤50 ms`过宽。若真实实现存在无法归入的异步间隔，应单独列为调度项，而不是留在50 ms残差中。

### 3.6 first PCM 不等于 first audible

现有“首个非空 HTTP body chunk”最多只能代表 first received bytes。正式实现需检查：

- 响应确实为预期裸 PCM/已知音频格式，而非WAV header或错误JSON；
- 字节数满足sample width对齐；
- 达到最小可播放缓冲，例如20–40 ms；
- 可选：首个可播放块包含非静音能量。

建议同时记录：

- `first_pcm_byte_ns`；
- `first_playable_pcm_ns`；
- 首块字节数、对应音频时长、RMS和peak。

若未接入真实声卡，论文必须说明未计入声卡和播放器缓冲延迟。

### 3.7 A/B TTS策略差异必须解释

主系统协议可以保持：

- B：首句就绪后立即调用TTS；
- A：完整 capped response 结束后调用TTS。

该比较回答的是**两套系统策略的总体TTFA差异**，不能把差异全部归因于ASR/KV。需保存：

```text
tts_text
tts_text_sha256
tts_n_chars
tts_n_bytes_utf8
tts_text_source=first_sentence|capped_full_response
sentence_end_found
sentence_fallback
response_token_count
generation_stop_reason
```

最低要求再加一个轻量的匹配文本TTS控制：对同一固定文本或同一首句进行TTS首包测量，以区分TTS输入长度/内容与早启动策略的影响。该控制可只做分层小样本，不要求重跑完整ASR/LLM管线。

### 3.8 正式运行的最低重复要求

50条A/B各一次足以补充“存在TTFA直接测量”，但不足以声称TTFA本身具有三轮稳定性。

最低建议：

- 主实验：50条配对A/B一次；
- 重复性子集：至少10条，语言/时长分层，A/B各3次；
- 报告子集的median/mean CV和顺序效应。

如果不做重复性子集，可以继续完成论文，但必须把TTFA写成单轮配对观察，不得把E1的TTFT重复性外推到TTS/TTFA。

### 3.9 W1必须具备的工程保护

- `schema_version`和`run_id`；
- 唯一键为`sample_id, mode, repeat_idx`；
- checkpoint保存配置hash；
- 恢复时校验commit、dirty状态、样本清单、音频hash、模型、TTS、endpoint算法、顺序表和生成参数；
- 正式运行要求显式输入路径，不通过glob猜“最新文件”；
- timeout/HTTP非200/空响应必须作为失败落盘，不可静默重试后当单次成功；
- 固定并保存Python/NumPy/Torch/CUDA seed和实际生成参数。

---

## 4. W3、W4、W5 的必须补充项

### 4.1 W3：CV

固定定义：

```text
CV_i = std(x_i1, x_i2, x_i3, ddof=1) / mean(x_i1, x_i2, x_i3)
```

需明确：

- 输出以百分比表示；
- 每个保留样本必须恰有3轮、同模式、同配置、无error；
- P90使用的percentile插值方法；
- 输出逐样本明细，以及mean/median/P90/max/CV>5%数量与比例；
- 输入三个结果文件路径和SHA-256；
- 所有论文表格中的std定义统一注明，不能只改CV摘要。

### 4.2 W4：macro和corpus WER/CER

逐样本文件必须分别保存WER与CER的计数：

```text
wer_substitutions / wer_deletions / wer_insertions / wer_ref_units
cer_substitutions / cer_deletions / cer_insertions / cer_ref_units
```

汇总文件按语言、条件、系统输出：

- mean utterance WER/CER；
- corpus WER/CER；
- ΣS/ΣD/ΣI/ΣN；
- n和配对过滤规则。

必须满足：

```text
corpus WER = Σ(wer_S + wer_D + wer_I) / Σwer_N
corpus CER = Σ(cer_S + cer_D + cer_I) / Σcer_N
```

A/B比较使用同一sample ID集合，并生成paired filter manifest。英文case-fold、中文去接缝空格、标点处理必须复用同一套归一化实现。

最终文档不得再出现AISHELL旧值6.72%。现有10.77%/11.80%应明确为哪一种宏平均口径；corpus值以W4新结果为准。

### 4.3 W5：配对统计协议

正式运行前固定：

- bootstrap以sample ID为配对重采样单位；
- 建议10,000次，固定seed；
- CI方法，例如percentile 95% CI；
- 改善率统一定义为`(mean(A)-mean(B))/mean(A)`，不要与逐样本改善率均值混用；
- Wilcoxon使用双侧或单侧、`zero_method`、exact/asymptotic和连续性校正；
- 多条件推断是否采用Holm校正；
- 效应量名称，例如rank-biserial correlation或paired Cohen's dz；
- 输出n、点估计、CI、统计量、原始p、校正p、效应量及过滤规则。

文字规则：

- CI跨0或校正后p不达标：不得写“统计显著”；
- 不显著不等于等价；
- R5单裁判结果即使p不显著，也不得写“statistically equivalent/indistinguishable”。

---

## 5. 原方案遗漏或覆盖不足的项目

### 5.1 新增 W9：LocalAgreement审计说明

不要求重跑，但必须形成最终方法/过滤说明：

1. 命名为项目内实现的`LocalAgreement-2-style baseline`；
2. 描述：
   - absolute audio timeline；
   - sentence-boundary-aware trimming；
   - `la_max_buffer_s=15.0`；
   - punctuation-robust agreement；
3. 披露505→498：
   - 3个运行错误；
   - 4个任一流式模式TTFT>10秒；
   - 成对排除；
   - 未过滤失败/挂起率7/505；
4. 汇总脚本只白名单最终文件，排除`invalid_dev3_frame_bug/`；
5. 不把“质量同量级”写成“统计等价”。

### 5.2 扩大 W7：真人语音QA

人工试听记录至少包含：

- 5条样本ID；
- 中英文均覆盖；
- 至少含extra-long；
- 试听者、日期；
- 是否有截断、错序、爆音、异常静音或音量问题；
- 结论标为manual spot check，不称human evaluation。

同时补充文档说明：

- 同章节/同说话人朗读句拼接；
- 人工静音构造；
- `reference`与`reference_full`恢复和校验过程；
- babble空输出率：LibriSpeech 12/30、AISHELL-1 5/30；
- `error=0`只表示程序未崩溃，不表示有效转写率100%。

### 5.3 扩大 W6：语义复现元数据

需区分两组参数：

- 被评价A/B回复：temperature=0.1，max_tokens=128；
- judge：temperature=0，judge自身max_tokens另记。

补充：

- BGE-M3 model和tokenizer revision/hash；
- Hugging Face snapshot commit；
- judge完整模型ID、供应商/base URL、请求时间；
- prompt hash、输入/回复hash、原始响应hash；
- 失败/重试；
- 样本范围仅为50个Very Long合成样本；
- 无法回溯的服务版本明确写unknown，不得事后推断。

语义结论继续定位为探索性，不建立等价性。

### 5.4 扩大 W8：全链路同步清单

至少同步：

- `PAPER_WRITING_REFERENCE.md`；
- `PAPER_HANDOFF.md`中仍被引用的旧数字；
- `REVISION_CHANGELOG.md`；
- 最终CSV/JSON和RUNINFO；
- 后续论文正文和回复信。

建立“主张→来源文件→生成脚本→论文位置”清单，并覆盖：

- TTFA新口径；
- CV新口径；
- macro/corpus WER/CER；
- AISHELL旧值清除；
- append-only与内部漂移；
- tokenizer seam 50%；
- KV边际收益；
- LA排除规则；
- babble空输出；
- 平台绑定；
- 128-token上限；
- 语义探索性限制。

---

## 6. 建议项：不作为本轮硬阻塞

以下项目能提高可复现性，但不应无限扩大本轮修改范围：

1. TTFA全部50条重复3轮——推荐，但最低可用10条分层子集重复3轮。
2. 第二个LLM judge或judge重复评分——有价值，但可在论文中把现有语义结果明确降级为探索性来替代。
3. 真实声卡播放测量——本轮可不做，只需把指标称first received/playable PCM并声明未计入播放设备延迟。
4. 所有增强条件使用完全相同30条样本重做——本轮可不重跑；论文只做条件内A/B比较，不进行严格跨条件因果排序。
5. GPU竞争实验——仍可不做，但必须列为独占GPU限制。
6. 原平台重跑TTFA——无需做；新TTFA只绑定第二平台。

---

## 7. 执行后能否满足论文需求

### 7.1 按当前开发方案原文直接执行

**不足以完全满足。** 最大风险仍是跑出一张数值可相加、但endpoint和first-audible语义不成立的Table VIII。

### 7.2 按本报告补充后执行

可以满足论文和审稿回复的主要需求，但结论必须限定为：

- 固定离线实时回放协议；
- 同一批样本上的配对独立A/B请求；
- 第二平台、单机、独占GPU；
- 指定CosyVoice服务、speaker和speed；
- speech-end-to-first-received/playable-PCM，而非包含真实声卡播放的用户可听延迟；
- System B与A的TTFA差异包含各自TTS调用策略；
- 结果不泛化到所有TTS、自然对话、分布式网络或高并发条件。

其他审稿意见可按以下强度回应：

- 真人语音：公开真人朗读语音拼接长样本上的验证，非自然对话；
- 统计：描述统计、三轮有限重复性和配对推断，不声称全面复现性；
- LA：同引擎自实现LA-2-style baseline上的同机比较；
- 机制：下游无回滚，但内部漂移常见；
- 语义：50条合成Very Long样本上的探索性证据，不能声称等价或无损。

---

## 8. 开发侧回复本报告时需明确的事项

开发侧无需再写长篇论证，只需逐项确认以下内容并更新实施方案：

1. W1是否增加真正的`endpoint_decision_or_flush_ns`，并区分endpoint、feed wait和源尾静音；
2. physical speech end算法的完整固定参数；
3. A/B原始事件字段、分项公式和单调性断言；
4. first received PCM与first playable PCM的定义；
5. A/B配对独立请求和AB/BA平衡顺序；
6. System A直接TTS测量及`generation_end/tts_request_start`字段；
7. 是否执行最低10条×A/B×3轮的TTFA重复性子集；若不执行，确认论文降级为单轮观察；
8. CV公式、P90方法及所有std口径；
9. WER/CER分别保存S/D/I/N及汇总结构；
10. bootstrap/Wilcoxon/效应量的固定参数；
11. 新增LocalAgreement说明工作项；
12. 扩大的真人QA、语义元数据和写作总册同步清单。

开发侧完成上述方案澄清后，可以启动实现和GPU正式实验；正式结果产生后仍需进行一次结果级QA，之后才放行论文修改。
