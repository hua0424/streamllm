# 开发整改回复再审报告 r2（2026-08-21）

- **再审对象**：
  - `reply-review-dev-assessment-and-plan-20260821.md`
  - `dev-assessment-and-plan-20260821.md`（v2）
- **前序评审**：`review-dev-assessment-and-plan-20260821.md`
- **目的**：确认开发回复是否闭环，以及是否可以放行实现、正式GPU实验和论文修改。

---

## 1. 最终裁决

开发回复已经逐项接受上一轮提出的12项要求，更新后的v2计划没有整项遗漏，**可以放行进入实现阶段**。

但当前只能放行“编写代码、self-test、静态审查和冒烟准备”，**暂不直接放行正式GPU实验**。剩余问题不再是论文实验范围问题，而是W1统一TTFA脚本必须在实现中冻结的底层计时语义。若这些定义不先固定，正式结果仍可能出现物理因果错误、无法闭合或不可复算。

阶段性裁决如下：

| 阶段 | 裁决 |
|---|---|
| W1/W3–W9代码与文档实现 | **放行** |
| W1 self-test与3条冒烟 | **实现完成后放行执行** |
| 正式50条A/B GPU实验 | **暂缓；须通过第3节Gate 1** |
| 论文数据锁定与正式改稿 | **暂缓；须通过结果级QA** |

本轮不要求再扩大实验范围。无需新增公开数据集、多judge、真实声卡、GPU竞争、原平台TTFA或全50条三轮重复。

---

## 2. 上一轮12项回复闭环情况

| 项目 | 状态 | 说明 |
|---|---|---|
| 区分endpoint、feed wait、源尾静音 | **基本闭环** | 已承诺不再把feed wait冒充VAD endpoint |
| 冻结physical speech end | **部分闭环** | 参数方向已有，实际算法、Silero revision和映射公式仍需在实现前固化 |
| A/B事件链、公式和单调性 | **部分闭环** | 字段已列，但按模式公式和偏序仍需落实到schema/self-test |
| received/playable PCM区分 | **闭环** | 论文只能称first playable/received PCM，不称真实声卡first audible |
| 配对独立请求与AB/BA平衡 | **基本闭环** | 需区分记录主键和配对键 |
| System A直接测TTS | **闭环** | 已废弃0.09s/字符估计 |
| 10条×A/B×3轮子集 | **承诺闭环** | 需明确主实验轮是否计入三轮 |
| CV ddof与P90/std口径 | **闭环** | 实现后按预期数量级验收 |
| macro/corpus WER/CER | **闭环** | 实现须分别保存WER/CER的S/D/I/N |
| bootstrap/Wilcoxon/效应量 | **基本闭环** | 需冻结比较族、correction和符号方向 |
| LocalAgreement W9 | **闭环** | 不需要重跑，只需方法、排除和白名单落地 |
| 真人QA、语义元数据、总册同步 | **方案闭环** | 阻塞论文放行，不阻塞W1代码实现 |

---

## 3. 正式GPU实验前必须关闭的W1实现门槛

以下要求应直接落实到W1代码、schema、RUNINFO和self-test，不需要再写一轮长篇方案说明。代码审查通过后即可启动正式GPU实验。

### 3.1 冻结physical speech end算法

必须明确并落盘：

1. 原始WAV SHA-256和分析波形SHA-256；
2. mono/downmix和重采样实现及版本；
3. 分析采样率、25ms窗口、10ms hop；
4. 底噪估计区间、统计量及dB到线性幅值公式；
5. 窗口尾部补零、阈值比较符、sample rounding/clamp规则；
6. energy与Silero差值≤200ms时采用哪一个结果；
7. 任一算法无speech或失败时的fail-closed策略；
8. Silero包版本、模型revision/commit及可取得的模型hash；
9. `estimated_physical_speech_end_sample`及其来源标记。

不应再写“源WAV必然16kHz mono”而不校验；现有加载路径允许运行时重采样。

### 3.2 修正实时回放因果语义

现有旧脚本先把整个500ms chunk放入队列，再sleep 500ms。若新W1继续这样做，消费者会在chunk末尾样本按物理时间到达前看到整块音频，可能系统性低估TTFA或产生负区间。

W1必须采用因果一致的计划回放：

```text
physical_speech_end_ns
  = playout_start_ns
  + round(physical_speech_end_sample * 1e9 / sample_rate)
```

并满足：

- 使用`time.perf_counter_ns()`；
- chunk只在其末样本计划到达时刻或等价的因果边界释放给消费者；
- 使用绝对deadline调度，避免相对sleep累计漂移；
- 保存每个chunk的planned release、actual release和scheduler error；
- `feed_end_ns`记录最后chunk实际释放完成时刻。

如果仍选择chunk起始时整体释放，就不能把chunk内部sample线性映射为实际到达时间，也不能称实时回放TTFA。

### 3.3 System A也必须服从同一播放时间轴

旧System A直接对完整内存音频执行ASR，没有等待实时播放结束。W1不得复用该行为。

要求：

- A和B使用同一个计划回放定义；
- System A在`feed_end_ns`之前不得启动full-audio ASR；
- A增加`asr_start_ns`和`asr_complete_ns`；
- 不得把旧`audio_load_time`当成TTFA起点。

否则A可能在用户“尚未说完”时已经处理完整音频，A/B端到端比较无效。

### 3.4 拆分endpoint、flush和full-input事件

不要继续使用含混的：

```text
endpoint_decision_or_flush_ns
```

建议改为：

```text
endpoint_mode = online_vad | explicit_flush | full_input
endpoint_decision_ns
explicit_flush_start_ns
explicit_flush_done_ns
full_input_ready_ns
```

不适用字段置null。

本实验若没有可靠的在线VAD endpoint：

- B使用`endpoint_mode=explicit_flush`；
- A使用`endpoint_mode=full_input`；
- 不报告VAD endpoint latency。

至少分别输出：

```text
trailing_feed_wait_ms
  = feed_end_ns - physical_speech_end_ns

feed_to_close_wait_ms
  = pipeline_input_close_ns - feed_end_ns
```

源尾静音是输入属性，不应再作为额外组件重复加进TTFA。

### 3.5 冻结A/B组件公式

定义：

```text
text_ready_ns =
  B且找到句末: first_sentence_boundary_ns
  B且fallback: generation_end_ns
  A: generation_end_ns
```

每条记录统一闭合为：

```text
TTFA_playable =
    (feed_end_ns - physical_speech_end_ns)
  + (pipeline_input_close_ns - feed_end_ns)
  + (first_token_ns - pipeline_input_close_ns)
  + (text_ready_ns - first_token_ns)
  + (tts_request_start_ns - text_ready_ns)
  + (first_playable_pcm_ns - tts_request_start_ns)
```

要求：

- 原始ns字段闭合残差应严格为0；
- 导出ms后的残差可允许仅由舍入产生的<1ms差异；
- 不再允许用50ms残差掩盖遗漏区间；
- A/B分别定义合法偏序，不能简单要求字段清单全序。

特别是流式B中`first_token_ns`可能早于最后一次ASR commit，因此单调性断言必须按真实因果边，而不是按字段排列顺序。

### 3.6 冻结B首句检测和fallback

W1必须复用一个明确版本的句末检测实现，并记录：

- 中文`。！？`和英文`!?`规则；
- 英文`.`对小数、缩写或其他边界的处理；
- 标点是否包含在TTS文本内；
- 逐token累积后的检测时机；
- 空字符串或special token处理；
- 无句末时仅在EOS/max_tokens后fallback到capped full response；
- `first_sentence_boundary_ns`定义为待发送文本已冻结的时刻。

该算法直接决定System B何时启动TTS，不能留到实现者自行选择。

### 3.7 固定PCM读取粒度和playable定义

旧客户端`iter_content(chunk_size=16000)`约对应363ms PCM，会系统性推迟应用层观察到首块的时刻，不能用于新的30ms playable阈值。

W1应冻结：

- 应用层读取粒度不大于30ms PCM，建议256或512 bytes；
- 22050Hz、16-bit mono下，30ms阈值为：

```text
ceil(0.030 * 22050) * 2 = 1324 bytes
```

- `first_pcm_byte_ns`为首次读取有效PCM body的时刻；
- `first_playable_pcm_ns`为累计完整sample首次达到1324 bytes的时刻；
- 校验HTTP status、Content-Type/Encoding和WAV/JSON magic；
- 格式错误记整行error，不得降级为received成功；
- 保存首块字节数、PCM时长、RMS和peak。

“playable”表示已经收到最低播放缓冲，不表示声卡已经发声。

### 3.8 生成接口必须暴露token和stop reason

现有生成器只yield解码文本，无法可靠区分EOS和max_tokens，也可能把EOS解码后的空字符串当成first token。

W1实现需要获得：

```text
token_id
decoded_text
is_eos
token_index
stop_reason = eos | max_tokens | error
```

并规定：

- `first_token_ns`对应首个非EOS模型token；
- EOS不计入`response_token_count`；
- 零内容回复记error；
- RUNINFO同时记录请求参数和实际生效参数；
- 当前代码中的`repetition_penalty`是否真正生效必须核实，不能只记录调用参数。

### 3.9 明确重复轮次和键

必须区分：

- **记录主键**：`sample_id, mode, repeat_idx`；
- **A/B配对键**：`sample_id, repeat_idx`。

建议定义：

- `repeat_idx=0`为主实验，并计入10条子集的三轮；
- 子集只额外执行`repeat_idx=1,2`；
- 每个子集样本三轮采用交替顺序，例如AB/BA/AB或BA/AB/BA；
- 完整schedule预生成并保存hash；
- CV只使用同一`sample_id × mode`的恰三条有效记录。

若开发选择主实验之外再做三轮也可以，但必须明确总计四次，并指明CV使用哪三轮，不能由分析脚本自动猜测。

### 3.10 checkpoint和异常处理必须fail closed

至少实现：

- JSON schema记录类型、单位、nullable和枚举；
- checkpoint原子写入并含`schema_version/run_id/config_hash/schedule_hash`；
- checkpoint损坏或hash不匹配立即退出；
- 配置变化必须新建run；
- error行保留，不自动静默重试；重跑使用新`attempt_idx`或run_id；
- worker异常通过共享exception channel上报并触发取消；
- 每个pair有总deadline；
- TTS有connect/read及独立total timeout；
- 正式QA按预期唯一键检查所有终态行，不能只统计成功行。

---

## 4. W3–W9仍需明确但不阻塞W1代码实现的事项

### 4.1 W3：CV

当前计划可以实施。预期验收应接近上一轮独立重算：

- B mean CV约5.19%，median约4.05%，max约18.96%；
- A mean CV约5.23%，median约4.65%，max约14.01%。

如果偏差明显，应先查轮次匹配、过滤和单位，不应直接覆盖旧摘要。

### 4.2 W4：WER/CER覆盖范围

W4应先列出论文和回复信最终保留的全部质量数字，至少覆盖：

- R2干净真人语音；
- R2增强条件；
- Table VII A/B/LA；
- R4流式拼接与System A外部一致性；
- QA及总册中仍引用的WER/CER。

每个保留数字必须明确是mean utterance还是corpus口径。无需重跑ASR。

### 4.3 W5：冻结最小比较族

避免对所有表格单元进行无边界检验。建议限定：

1. R2各条件内A/B TTFT，12个增强条件为一个Holm family；
2. Table VII的B vs LA为主要比较，A vs B为验证性比较；
3. Table III总体或三个预定义长度组的A/B配对差；
4. R5可以只报bootstrap CI并删除“统计不可区分”，不强制新增等价性检验。

实现时固定SciPy版本、Wilcoxon `correction`、rank-biserial符号方向和全零差处理。

### 4.4 W6–W9

开发回复已基本覆盖，不新增实验要求。最终结果级QA需确认：

- W6无法回溯项确实写`unknown`，不补猜；
- W7有5条实际试听记录，并单列babble空输出率；
- W8对总册、交接、changelog、论文和回复信执行旧数字/旧措辞残留搜索；
- W9只读取最终白名单结果，列出7个排除样本并报告7/505未过滤失败/挂起率。

---

## 5. 正式GPU放行Gate 1

W1实现后，必须先通过以下检查：

1. physical speech end算法、Silero revision/hash及分析波形hash落盘；
2. chunk回放遵循因果到达语义；
3. System A等待feed结束后才启动full ASR；
4. endpoint/feed/flush字段拆分；
5. A/B组件公式和偏序断言固化；
6. B句末检测和fallback固化；
7. PCM读取粒度及1324-byte playable阈值固化；
8. token ID、EOS和stop reason可审计；
9. 主键、配对键、三轮定义无歧义；
10. checkpoint和错误处理fail closed；
11. TTS探活确认返回格式；
12. 3条冒烟通过schema、因果顺序、非负TTFA、ns级闭合、PCM格式和错误落盘测试。

Gate 1通过后即可放行正式GPU实验，无需再提交新的宏观实验方案。

---

## 6. 结果级QA与论文放行

正式运行结束后仍须检查：

- 50条主实验全部有A/B配对记录；
- 10条子集各模式恰有冻结协议规定的3轮；
- WAV/分析波形hash一致；
- 失败行完整保留，无静默过滤；
- 所有成功TTFA非负；
- A/B事件偏序均通过；
- 原始ns闭合残差为0，导出ms残差<1ms；
- System A无任何字符线性估计项；
- received/playable降级情况完整披露；
- 匹配文本TTS控制完成；
- 顺序效应和重复子集CV已输出；
- W3–W9产物与总册一致。

通过后方可更新`PAPER_WRITING_REFERENCE.md`为“定稿”并正式修改论文。

---

## 7. 论文最终可支持的范围

整改全部通过后，五条审稿意见可以得到充分且诚实的回应，但结论仍须限定为：

- 固定离线音频实时回放协议；
- 同一批样本上的配对独立A/B请求；
- 第二平台、单机、独占GPU；
- 指定TTS服务、speaker和speed；
- speech-end-to-first-playable-PCM，不含声卡/扬声器播放；
- System B与A的TTFA差异包含各自TTS调用策略；
- 真人语音是拼接真人朗读语音；
- LA是项目内实现的LA-2-style baseline；
- 下游无回滚但内部漂移常见；
- 语义结果仅为50条合成Very Long样本的探索性证据；
- babble是明确失败边界。

---

## 8. 本轮结论

**回复函可以接受，v2开发计划可以进入实现。**

但正式GPU运行前仍需由W1实现和self-test实际关闭第3节问题；不能仅凭回复函文字视为已经关闭。开发侧下一步应直接实现并提交W1代码、schema、self-test和3条冒烟结果供脚本级审查，不需要继续扩写总体方案。
