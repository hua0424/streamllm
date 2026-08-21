# 开发整改方案 v3 再审报告（2026-08-21）

- **审查对象**：
  - `dev-assessment-and-plan-20260821.md`（顶部 v3 修订节）
  - `reply-review-dev-assessment-and-plan-20260821-r2.md`
- **前序审查**：`review-reply-dev-assessment-and-plan-20260821-r2.md`
- **目的**：确认 v3 是否足以进入实现、GPU 冒烟和正式 GPU 实验。

---

## 1. 总体裁决

**v3 有条件通过，可以进入代码实现和 self-test；当前仍不放行正式 50 条 A/B GPU 实验。**

v3 已吸收前序审查的大部分实质要求，没有遗漏 W3–W9，也没有不必要扩大实验范围。五条审稿意见在方案执行成功、通过 Gate 1 和结果级 QA 后，可以得到充分且诚实的回应。

当前分阶段状态：

| 阶段 | 裁决 |
|---|---|
| W1/W3–W9 实现 | **放行** |
| W1 schema、RUNINFO、self-test | **放行** |
| TTS 探活与 3 条 GPU 冒烟 | **脚本级审查通过后放行** |
| 正式 50×A/B GPU 实验 | **暂不放行** |
| 论文数据锁定与正式改稿 | **暂不放行** |

剩余问题应直接在 W1 代码、schema 和 self-test 中关闭，不需要继续扩写宏观实验方案。开发侧下一次应提交实现和测试证据，而不是再提交 v4 总体方案。

---

## 2. v3 已正确解决的事项

以下设计已达到可实施方向：

1. A/B 改为同一样本上的配对独立请求，并采用 AB/BA 平衡计划；
2. chunk 按末样本计划到达时刻释放，并使用绝对 deadline；
3. System A 必须等待完整输入播放结束后才启动全量 ASR；
4. endpoint、explicit flush 和 full input 不再作为同一概念；
5. System A 直接调用 TTS，废弃 `0.09 s/字符`估计；
6. 同一组 ns 时间戳形成闭合恒等式；
7. 同时记录 first received PCM 和 first playable PCM；
8. 30 ms、22050 Hz、16-bit mono 的 playable 阈值 1324 bytes 计算正确；
9. 计划新增 token/EOS/stop reason 元信息接口，同时保留旧 `generate()`兼容性；
10. `repeat_idx=0`计入三轮，重复子集只补 `repeat_idx=1,2`；
11. checkpoint 采用 schema/run/config/schedule hash 和 fail-closed 方向；
12. W3 CV、W4 macro/corpus WER/CER、W5配对统计、W6–W9均继续保留。

---

## 3. Gate 0：实现前需统一的文档口径

以下四项不要求新增实验，但应在编码前明确，以免 v3 顶部与旧 v2 段落产生双重定义。

### 3.1 v3 修订节优先于旧 v2 正文

方案文件中仍保留若干旧表述，与顶部 v3 冲突：

- 顶部要求不假设源 WAV 一定为 16 kHz mono，旧段仍写“源 WAV（16kHz mono）”；
- 顶部弃用 `endpoint_decision_or_flush_ns`，旧字段清单仍保留该字段；
- 顶部规定格式错误整行 error，旧段允许 playable 失败后降级 received；
- 顶部要求按真实因果边断言，旧段容易被理解为全部字段严格全序；
- 顶部确认 repetition penalty 未实际应用，旧 W6 仍将1.1写成实际参数。

开发实现必须明确：**顶部 v3 修订节覆盖所有冲突的旧 v2 表述。** 最好在方案顶部增加一句优先级声明，并在实现 schema 中只保留 v3 字段。

### 3.2 repetition penalty 使用 requested/effective 双字段

已确认历史代码请求值为1.1，但 `_decode_logits()`并未应用。最终元数据必须写：

```text
requested_repetition_penalty = 1.1
effective_repetition_penalty = not_applied
```

有效采样配置是 temperature=0.1、top_p=0.9、无实际重复惩罚。本轮保持历史行为、不修复后重跑是合理的，但论文不得写成“repetition penalty=1.1实际生效”。

### 3.3 W5 唯一冻结统计比较族

不得在看到结果后再选择“Table III总体或三个分组”。建议预先固定为：

- Table III：总体 A/B 为主比较；Long/Very Long/Extra Long 为一个三比较 Holm family；
- Table VII：B vs LA 为主比较，A vs B 为验证性比较，二者分开标注；
- R2：每个条件内部 A/B，12个增强条件为一个 Holm family；
- R5：只报告独立意图满足评分的配对均值差 B−A 的 bootstrap 95% CI，不做等价性检验。

如开发侧选择其他唯一方案也可以，但必须在运行统计脚本前固定，不能保留“或”。

### 3.4 W8 加入方法文档同步

全链路同步清单除总册、handoff、changelog、CSV/RUNINFO、论文和回复信外，还应包含：

- `experiments/EXPERIMENT_DESIGN.md`；
- `experiments/CISR_REVISION_PLAN.md`中的完成状态或“后续口径以v3审计方案为准”的说明。

---

## 4. 正式 GPU 前必须在 W1 实现中关闭的阻塞项

### 4.1 冻结 physical speech end 的实际算法

v3 仍把若干关键值推迟到“写成脚本常量”，因此 Gate 1 需要实际检查：

- `physical_speech_end_sample`定义为排他右边界，范围 `[0, N]`；
- 原始 WAV hash 与重采样分析波形 hash；
- downmix和重采样库、模式、dtype；
- 底噪估计区间及短音频 fallback；
- frame RMS 和聚合统计量；
- `db_to_amplitude = 10 ** (db / 20)`；
- 25 ms窗口、10 ms hop；
- 尾窗补零、比较符、round/clamp；
- energy/Silero 差值≤200 ms时的明确选择；
- no-speech、NaN/Inf、空文件和单算法失败的fail-closed行为；
- Silero包版本、固定模型revision/commit及可取得的模型hash。

当前 `torch.hub.load('snakers4/silero-vad', ...)`没有固定revision，W1的PSE分析不能只依赖浮动远端仓库或不明缓存。

### 4.2 保证chunk回放的物理因果一致性

旧脚本是“先put整个500 ms chunk，再sleep 500 ms”，会让消费者提前看到未来样本。W1不得复用该顺序。

必须满足：

```text
planned_release_ns = playout_start_ns + 累计样本数 / sample_rate
actual_release_ns >= planned_release_ns
```

并且：

- 到deadline后才发布chunk；
- 最后一块按实际样本数计算deadline；
- 保存planned/actual/scheduler error；
- 提前发布直接记error；
- 论文说明这是500 ms chunked real-time replay，而不是逐sample连续流。

### 4.3 保持原 System B 生成语义

当前正式System B是在最终ASR/input close后才开始生成。v3中“first token可早于最后ASR commit”的表述可能诱使实现者改变系统，使LLM在ASR尚未完成时提前生成。

W1必须保持原系统：

```text
explicit_flush_done
<= pipeline_input_close
<= asr_processing_done
<= first_token
```

- 新增`asr_processing_done_ns`，不要用内部最后一次commit代替；
- 首句冻结后启动独立TTS worker；
- LLM继续生成，不因调用TTS而提前停止；
- 不得为了满足时间戳公式改变System B的生成起点。

### 4.4 用无条件 INPUT_CLOSED sentinel 消除死锁

现有流式close路径存在确定性死锁风险：如果`flush()`返回None，最终`is_final`可能永远不进入ASR队列，transcriber会一直等待。

W1必须：

- 将音频数据和生命周期控制分离；
- 无论flush是否产生音频，都发送`INPUT_CLOSED` sentinel；
- collector收到sentinel后设置close event；
- worker异常通过共享exception queue上报；
- 任一失败触发cancel event；
- `join()`具有总deadline；
- GPU worker异常后fail-stop当前run，不带污染状态继续正式样本。

self-test必须覆盖`flush() -> None`。

### 4.5 明确定义feed/flush/close事件

建议冻结：

```text
feed_end_ns
  = producer在最后chunk deadline后完成发布的时刻

explicit_flush_start_ns
  = segmentation consumer消费完最后chunk、即将调用flush的时刻

explicit_flush_done_ns
  = flush返回且结果和INPUT_CLOSED均发布到ASR输入队列的时刻

pipeline_input_close_ns
  = ASR collector收到INPUT_CLOSED的时刻

full_input_ready_ns (A)
  = 最后chunk完成发布后允许完整数组交给full ASR的时刻
```

并断言：

```text
A.asr_start_ns >= A.full_input_ready_ns >= A.feed_end_ns
```

v3代码/schema中不再保留`endpoint_decision_or_flush_ns`。

### 4.6 修正首句检测的跨token问题

现有英文句点规则在逐token流上会把`3`、`.`、`5`中的小数点提前判成句末。W1应：

- 基于累计token IDs重新解码累计文本，避免简单拼接per-token decode；
- 对`.`使用一字符lookahead；
- EOS/max_tokens时再裁决末尾pending句点；
- 边界时刻取完成判定、冻结TTS文本的时刻；
- 标点保留在TTS文本中；
- 无句末只在EOS/max_tokens后fallback到capped full response；
- 明确缩写不做完整处理时的限制。

self-test至少覆盖：

- `3` + `.` + `5`；
- `Mr` + `.` + ` Smith`；
- token内句点；
- EOS前末尾句点；
- 空decoded token；
- 中文标点；
- 同一token中标点后还有文本。

### 4.7 固定PCM读取、超时和格式验证

512-byte应用读取粒度和1324-byte playable阈值可采用。还需落实：

- `requests`使用connect/read timeout；
- 外层使用`perf_counter_ns` total deadline；
- 超时/取消时主动关闭response；
- TTS探活在冒烟前单独执行并保存status、Content-Type、Content-Encoding、magic、服务PCM配置；
- 原始字节连续累积，不因奇数字节读取丢弃半个sample；
- `first_pcm_byte_ns`为应用层首次读取有效body；
- `first_playable_pcm_ns`为累计完整PCM首次达到1324 bytes；
- RMS/peak基于playable buffer；
- HTTP/格式/对齐/零内容错误整行error，不降级成功；
- 主Table VIII不得混合received与playable口径。

如果服务没有稳定Content-Type，应在探活后固定允许的缺失/取值策略，不得正式运行时临时放宽。

### 4.8 生成接口、EOS和随机数

新增接口应暴露：

```text
token_id
decoded_text
is_eos
token_index
stop_reason = eos | max_tokens | error
```

建议另记：

```text
first_model_token_ns
first_content_token_ns
```

历史TTFT兼容可用first model token；首句/TTS推进必须使用累计可解码内容。零内容回复记error。

为避免AB/BA执行顺序改变全局`torch.multinomial`随机序列：

- 按配对键用canonical JSON + SHA-256派生generation seed；
- 同一`sample_id, repeat_idx`的A/B使用同一基础seed；
- 每次生成前重置对应设备RNG，或使用独立`torch.Generator`；
- 保存generation seed；
- 不使用Python内置`hash()`派生seed；
- 若TTS不支持seed，标为不可控变异。

### 4.9 重复schedule

固定：

- 记录主键：`sample_id, mode, repeat_idx`；
- 配对键：`sample_id, repeat_idx`；
- repeat 0计入三轮；
- 子集只补repeat 1/2；
- 10条子集中5条采用AB/BA/AB，5条采用BA/AB/BA，并按语言×时长平衡；
- 完整schedule预生成并hash；
- CV只使用每个sample/mode的恰三条有效记录。

### 4.10 checkpoint和worker必须真正fail closed

必须实现并self-test：

- `schema_version/run_id/config_hash/schedule_hash`；
- 主键状态`success|error|cancelled|timeout`；
- 原子写入、flush和fsync；
- checkpoint损坏或hash不匹配立即退出；
- error key不静默重跑；
- pair一侧失败时另一侧具有明确终态；
- worker异常共享上报并取消；
- pair总deadline；
- TTS慢流不会无限续命；
- schedule所有预期键最终恰有一条终态记录。

---

## 5. W3–W9完整性判断

| 工作项 | 结论 | 补充要求 |
|---|---|---|
| W3 CV | 完整 | 新结果应接近独立锚点；偏差先查轮次/过滤/单位 |
| W4 WER/CER | 完整 | 覆盖R2、增强、Table VII、R4及最终保留的QA/总册数字 |
| W5统计 | 基本完整 | Gate 0唯一冻结比较族、Wilcoxon correction及效应量符号 |
| W6语义元数据 | 基本完整 | requested/effective repetition penalty分开，历史未知项写unknown |
| W7真人QA | 完整 | 5条试听、babble空输出、reference恢复说明均落盘 |
| W8全链路同步 | 基本完整 | 增加`EXPERIMENT_DESIGN.md`及修订计划状态说明 |
| W9 LocalAgreement | 完整 | 7/505失败率、排除样本和最终白名单实际落地 |

无需新增数据集、第二judge、真实声卡、GPU竞争、原平台TTFA或全50条三轮重复。

---

## 6. Gate 1与后续放行

### 6.1 冒烟前脚本级Gate

W1实现后，先核验：

- PSE算法和双hash；
- chunk因果释放；
- A等待feed结束；
- B保持原生成语义；
- 无条件close sentinel；
- endpoint/flush/full-input字段和公式；
- 首句lookahead；
- PCM读取、格式和TTS探活；
- token/EOS/seed；
- schedule；
- checkpoint/worker fail-closed。

### 6.2 三条冒烟

三条GPU冒烟至少覆盖：

- 中英文成功路径；
- A/B两种模式；
- 句末正常路径；
- 至少一个可控故障注入路径用于验证error落盘；
- schema有效；
- 事件边非负；
- TTFA非负；
- 原始ns闭合残差为0；
- PCM达到1324-byte playable阈值；
- A没有提前启动ASR。

另以本机self-test覆盖flush=None、split decimal、EOS-only、checkpoint损坏、hash mismatch、worker exception和TTS慢流。

### 6.3 正式GPU

Gate 1与冒烟全部通过后，可放行：

- 50条主实验A/B单轮配对；
- 10条分层子集补repeat 1/2；
- 匹配文本TTS控制；
- W2环境记录。

无需再提交宏观实验方案。

### 6.4 论文放行

正式结果还需结果级QA：

- 50条A/B配对完整；
- 10条子集每模式恰三轮；
- 失败行保留；
- hash一致；
- 所有成功分项非负；
- ns闭合严格为0；
- playable/received不混口径；
- 顺序效应和CV输出；
- W3–W9全部落盘；
- W8完成总册、方法文档、论文和回复信同步。

通过后才可把`PAPER_WRITING_REFERENCE.md`重新标为定稿并开始正式改稿。

---

## 7. 最终论文结论边界

整改全部通过后，可以支持：

- 两个平台上的同机相对TTFT改善约70%–74%；
- 主要收益来自流式ASR；
- 当前配置下KV增量预填独立边际收益较小；
- 第二平台固定chunked实时回放协议下的speech-end-to-first-playable-PCM；
- 下游append-only、无回滚，但内部重识别漂移常见；
- 项目内LocalAgreement-2-style baseline；
- 50条合成Very Long样本上的探索性语义证据；
- babble为明确失败边界。

不得写：

- TTFA为真实声卡first audible；
- 测得在线VAD endpoint latency；
- A/B是同一请求；
- TTFA差异全部来自ASR/KV；
- 三轮CV均小于5%；
- 统计等价或语义无损；
- 内部提交文本不可变；
- token/KV状态等价；
- 真实自然对话验证；
- 各种噪声均有优势；
- repetition penalty=1.1实际生效；
- 15秒为通用切换阈值；
- 约1.1秒为完整可听响应。

---

## 8. 本轮最终结论

**v3及r2回复函可以接受，并放行进入实现和self-test。**

正式GPU仍需等待W1代码/schema消除上述实现级歧义，完成TTS探活并通过脚本审查和3条冒烟。下一轮审查对象应是实际代码、schema、self-test和冒烟证据，不再是总体方案文本。
