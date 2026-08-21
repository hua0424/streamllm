# 开发侧评估与修改方案 v3.1（对应 review-20260821-PRE-PAPER-AUDIT.md 及三轮复审）

- 日期：2026-08-21（v2：按复审补充要求修订；v3：按 r2 再审冻结实现层计时语义；v3.1：按 v3 再审 `review-dev-assessment-and-plan-v3-20260821.md` 关闭 Gate 0 口径冲突并冻结 Gate 1 实现细则）
- **优先级声明（Gate 0 §3.1）：本文件顶部"v3/v3.1 修订节"覆盖下方所有冲突的旧 v2 表述；实现 schema 只保留 v3.1 字段。**
- 结论先行：初审全部可核验声明属实（核验证据见 v1 第一节，仍有效）；三轮复审要求**全部接纳**。v3 再审核实发现：Silero 加载未固定 revision（`streamaudio_segmenter.py:124`）、flush=None 时 is_final 段不入队导致 transcriber 永久等待（`run_exp_latency.py:675-678` 配合 `:712-720`，**此即历史上 4 条流式挂起样本的机制**）、System B 生成必在最后 ASR 转写完成后启动（`:755-771`）、现有句末检测依赖完整文本双侧上下文、流式需 lookahead——四条全部属实。

---

## v3.1 增补（对应 v3 再审 Gate 0/Gate 1，2026-08-21）

### Gate 0 口径统一（已落实为本文档修订）

1. v2 旧段冲突表述以本修订节为准（§2.3 的"源 WAV（16kHz mono）"假设、§2.4 的 `endpoint_decision_or_flush_ns` 字段、§2.5 的"playable 失败降级 received"、§五 W5 的"总体或三个分组"选择、§六 W6 的 repetition_penalty=1.1 写法）；
2. 采样参数双字段：`requested_repetition_penalty=1.1` / `effective_repetition_penalty=not_applied`，论文不得写"1.1 实际生效"；
3. **W5 唯一冻结比较族**（采用再审建议，运行前固定）：Table III 总体 A/B 为主比较、Long/Very Long/Extra Long 三比较为一个 Holm family；Table VII B vs LA 主比较、A vs B 验证性比较分开标注；R2 十二增强条件为一个 Holm family；R5 只报 B−A 配对均值差 bootstrap 95% CI、不做等价性检验；
4. W8 同步清单增补 `experiments/EXPERIMENT_DESIGN.md` 与 `experiments/CISR_REVISION_PLAN.md`（完成状态或"后续口径以 v3.1 审计方案为准"说明）。

### Gate 1 实现细则（在 v3 修订节第 1 条基础上冻结）

1. **PSE 算法定稿**：`physical_speech_end_sample` 为排他右边界、范围 [0, N]；原始 WAV 与重采样分析波形双 SHA-256；downmix/重采样库、模式、dtype 落盘；底噪估计取全文件 RMS 最低的 10% 帧均值（音频 <3s 时 fallback 为全局 RMS×0.1）；`db_to_amplitude = 10**(db/20)`；25ms 窗/10ms hop、尾窗补零、严格大于比较、round 后 clamp 到 [0, N]；energy/Silero 差 ≤200ms 取 energy、>200ms 取 Silero 并标记；no-speech/NaN/Inf/空文件/单算法失败均 fail-closed；**Silero 固定 revision**：`torch.hub.load` 增加固定 commit（取 GPU 主机现有缓存对应 commit 并落盘 model hash），禁止浮动远端依赖。
2. **因果回放**：`planned_release_ns = playout_start_ns + 累计样本数×1e9/sample_rate`；`actual_release_ns ≥ planned_release_ns`，提前发布直接记 error；最后一块按实际样本数计 deadline；逐 chunk 保存 planned/actual/scheduler error；论文说明为 500ms chunked real-time replay 而非逐 sample 连续流。
3. **System B 生成语义保持原样**（更正 v3 第 5 条中照抄 r2 的"first_token 可早于最后 ASR commit"表述——经核实生产链路 `llm_worker` 仅在 `is_end` 后进入 `generate()`，该情形不存在）：链条冻结为 `explicit_flush_done ≤ pipeline_input_close ≤ asr_processing_done ≤ first_token`；新增 `asr_processing_done_ns`（不以内部最后 commit 代替）；首句冻结后启动独立 TTS worker、LLM 继续生成不提前停止；不得为满足时间戳公式改变生成起点。
4. **无条件 INPUT_CLOSED sentinel**：音频数据与生命周期控制分离；无论 flush 是否产出音频都发送 `INPUT_CLOSED`；collector 收到后设置 close event；worker 异常经共享 exception queue 上报并触发 cancel event；`join()` 带总 deadline；GPU worker 异常后 fail-stop 当前 run，不带污染状态继续。self-test 必须覆盖 `flush()→None`（该路径即历史挂起样本机制，修复后此类样本应转为正常完成或显式 error，而非挂起）。
5. **事件定义定稿**：`feed_end_ns`=producer 在最后 chunk deadline 后完成发布；`explicit_flush_start_ns`=segmentation 消费完最后 chunk、即将调用 flush；`explicit_flush_done_ns`=flush 返回且结果与 INPUT_CLOSED 均发布到 ASR 输入队列；`pipeline_input_close_ns`=ASR collector 收到 INPUT_CLOSED；`full_input_ready_ns`(A)=最后 chunk 发布后完整数组可交 full ASR；断言 `A.asr_start_ns ≥ A.full_input_ready_ns ≥ A.feed_end_ns`；schema 不再保留 `endpoint_decision_or_flush_ns`。
6. **流式句末检测定稿**：基于累计 token IDs 重解码累计文本（不拼接 per-token decode，规避 BPE 接缝）；`.` 使用一字符 lookahead（下一 token 未达则 pending）；EOS/max_tokens 时裁决末尾 pending 句点；边界时刻=文本冻结时刻；标点保留在 TTS 文本内；无句末仅在 EOS/max_tokens 后 fallback capped full response；缩写（如 "Mr."）不做完整处理、列为限制。self-test 覆盖：`3`+`.`+`5`、`Mr`+`.`+` Smith`、token 内句点、EOS 前末尾句点、空 decoded token、中文标点、同 token 标点后还有文本。
7. **PCM 读取定稿**：512-byte 应用粒度；requests connect/read timeout + 外层 `perf_counter_ns` total deadline；超时/取消主动关闭 response；**TTS 探活在冒烟前单独执行**并保存 status/Content-Type/Content-Encoding/magic/服务 PCM 配置，若服务无稳定 Content-Type 则探活后固定允许策略、正式运行不临时放宽；字节连续累积不因奇数读取丢半个 sample；RMS/peak 基于 playable buffer；HTTP/格式/对齐/零内容错误整行 error；主 Table VIII 不混 received/playable 口径。
8. **生成元信息与随机数**：新接口暴露 `token_id/decoded_text/is_eos/token_index/stop_reason`，另记 `first_model_token_ns`（与历史 TTFT 兼容）与 `first_content_token_ns`（首句/TTS 推进用）；零内容回复记 error。**RNG 隔离**：按配对键 canonical JSON+SHA-256 派生 generation seed；同一 `sample_id, repeat_idx` 的 A/B 用同一基础 seed；每次生成前重置对应设备 RNG 或用独立 `torch.Generator`；seed 落盘；不用 Python 内置 `hash()`；TTS 不支持 seed 则标不可控变异。
9. **重复 schedule 定稿**：主键 `sample_id, mode, repeat_idx`；配对键 `sample_id, repeat_idx`；repeat 0 计入三轮、子集只补 1/2；10 条子集 5 条 AB/BA/AB + 5 条 BA/AB/BA，按语言×时长平衡；完整 schedule 预生成存 hash；CV 只用恰三条有效记录。
10. **fail-closed 定稿**：主键终态 `success|error|cancelled|timeout`；checkpoint 原子写入 + flush + fsync；损坏/hash 不匹配立即退出；error key 不静默重跑；pair 一侧失败另一侧有明确终态；TTS 慢流不无限续命；schedule 全部预期键最终恰一条终态记录。

### 冒烟与 self-test 覆盖要求（再审 §6.2）

3 条 GPU 冒烟覆盖：中英文成功路径、A/B 两模式、句末正常路径、至少一个可控故障注入（验证 error 落盘）、schema 有效、事件边非负、TTFA 非负、ns 闭合残差为 0、PCM 达 1324-byte playable、A 未提前启动 ASR。本机 self-test 另覆盖：flush=None、split decimal、EOS-only、checkpoint 损坏、hash mismatch、worker exception、TTS 慢流。

---

## v3 修订节（对应 r2 再审，2026-08-21）

### 0. 重要核实发现：repetition_penalty 从未生效

`stream_llm_inference.py:308 _decode_logits()` 接收 `repetition_penalty` 参数但**函数体内从未使用**——仅有 temperature 除 logits + top_p 截断 + multinomial。即 E1–E6/R2–R6 全部历史生成文本的实际采样配置为 **temperature=0.1、top_p=0.9、无重复惩罚**。处置：

- W1 保持现有采样行为不变（与全部历史数据同引擎同行为，A/B 同码自洽），RUNINFO 同时记录请求参数与**实际生效参数**（repetition_penalty=未应用）；
- 论文方法部分若写有 repetition_penalty=1.1，改稿时必须更正（纳入 W8 旧措辞残留搜索清单）；
- 修复该死参数不在本轮范围（避免引入与历史数据不可比的新变量）。

### 1. W1 实现层冻结（r2 §3.1–3.10 全量落实，Gate 1 验收点）

1. **physical speech end 算法**：源 WAV 与分析波形双 SHA-256；mono/downmix 与重采样实现及版本落盘（不假设源必为 16kHz mono，运行时校验并重采样）；16kHz 分析率、25ms 窗/10ms hop；底噪估计区间/统计量/dB→线性公式、窗尾补零、阈值比较符、sample rounding/clamp 规则全部写成脚本常量；energy 与 Silero 差 ≤200ms 时取 energy 值（>200ms 取 Silero 并标记 `pse_method`）；任一无 speech/失败时 fail-closed（该行 error，禁止猜测）；Silero 包版本/模型 revision/hash 落盘；输出 `estimated_physical_speech_end_sample` 及来源标记。
2. **因果回放**（修正旧脚本"先 put 整块再 sleep"的反因果行为，`run_exp_latency.py:638-642` 已核实）：chunk 只在其末样本计划到达时刻释放给消费者；绝对 deadline 调度（禁相对 sleep 累计漂移，同时**不删除实时模拟节奏**这一红线不变——只是改为因果一致的释放时点）；逐 chunk 保存 planned/actual release 与 scheduler error；`physical_speech_end_ns = playout_start_ns + round(pse_sample × 1e9 / sample_rate)`，全程 `perf_counter_ns`；`feed_end_ns` = 最后 chunk 实际释放完成时刻。
3. **System A 同时间轴**：A/B 共用同一计划回放；**A 在 `feed_end_ns` 前不得启动 full-audio ASR**；A 增加 `asr_start_ns`/`asr_complete_ns`；旧 `audio_load_time` 不作 TTFA 起点（旧行为已核实：非流式直接处理内存音频、无实时回放）。
4. **事件拆分**（弃用含混的 `endpoint_decision_or_flush_ns`）：`endpoint_mode = online_vad | explicit_flush | full_input`；`endpoint_decision_ns` / `explicit_flush_start_ns` / `explicit_flush_done_ns` / `full_input_ready_ns`，不适用置 null。本实验 B=`explicit_flush`、A=`full_input`，不报告 VAD endpoint latency。分列输出 `trailing_feed_wait_ms = feed_end − physical_speech_end` 与 `feed_to_close_wait_ms = pipeline_input_close − feed_end`；源尾静音是输入属性，只作披露性分解，不作为组件重复计入 TTFA。
5. **A/B 组件公式**：`text_ready_ns` = B 找到句末→`first_sentence_boundary_ns`；B fallback→`generation_end_ns`；A→`generation_end_ns`。统一闭合：
   `TTFA_playable = (feed_end−pse) + (pipeline_input_close−feed_end) + (first_token−pipeline_input_close) + (text_ready−first_token) + (tts_request_start−text_ready) + (first_playable_pcm−tts_request_start)`，原始 ns 残差严格为 0，导出 ms 残差 <1ms（仅舍入）。A/B 分别定义合法偏序（**流式 B 的 `first_token_ns` 可早于最后一次 ASR commit**，单调性断言按真实因果边而非字段排列）。
6. **B 句末检测冻结**：复用 `measure_decode_to_first_sentence.py` 的已审查实现（中文 。！？、英文 !?、"." 非数字夹击规则），并明确：标点包含在 TTS 文本内、逐 token 累积检测、`first_sentence_boundary_ns` = 待发送文本冻结时刻、空串/special token 处理、无句末时仅在 EOS/max_tokens 后 fallback 到 capped full response。
7. **PCM 读取粒度**：应用层读取 ≤30ms PCM（512 bytes 粒度）；playable 阈值 1324 bytes（22050Hz×16bit×30ms，计算已复核）；`first_pcm_byte_ns` = 首次读到有效 PCM body；`first_playable_pcm_ns` = 累计完整 sample 首达 1324 bytes；校验 HTTP status/Content-Type/WAV·JSON magic，格式错误整行 error 不降级；首块字节数/PCM 时长/RMS/peak 落盘。旧客户端 `iter_content(chunk_size=16000)`≈363ms 粒度（已核实 `:65,:96`）不可复用于 playable 判定，仅其格式常量可复用。
8. **生成接口元信息**：W1 使用新增的不破坏兼容的元信息生成路径（`generate()` 旧签名保留），暴露 `token_id / decoded_text / is_eos / token_index / stop_reason(eos|max_tokens|error)`；`first_token_ns` 对应首个非 EOS 模型 token（现有生成器 EOS 时 yield 空串的问题已核实 `:215-219`）；EOS 不计入 `response_token_count`；零内容回复记 error。
9. **轮次与键**：记录主键 `sample_id, mode, repeat_idx`；配对键 `sample_id, repeat_idx`；**repeat_idx=0 为主实验且计入子集三轮**，子集仅额外跑 1、2（节省一轮 GPU）；子集三轮交替顺序 AB/BA/AB 或 BA/AB/BA；完整 schedule 预生成并存 hash；CV 只用同 `sample_id × mode` 恰三条有效记录。
10. **fail-closed 工程**：JSON schema（类型/单位/nullable/枚举）；checkpoint 原子写入含 `schema_version/run_id/config_hash/schedule_hash`，损坏或 hash 不匹配立即退出；配置变化必须新建 run；error 行保留不静默重试（重跑用新 run_id）；worker 异常经共享 channel 上报并触发取消；每 pair 总 deadline；TTS connect/read/total 三级 timeout；正式 QA 按预期唯一键查全部终态行。

### 2. W3–W5 细则（r2 §4）

- **W3** 验收锚点：B mean/median/max ≈ 5.19/4.05/18.96%，A ≈ 5.23/4.65/14.01%（本机两轮独立重算一致）；偏差明显先查轮次匹配/过滤/单位，不直接覆盖旧摘要。
- **W4** 覆盖清单：R2 干净集、R2 增强条件、Table VII A/B/LA 质量格、R4 外部一致性（4.93%/2.69%）、QA 与总册全部残留引用；每个保留数字标注 mean-utterance 或 corpus 口径；无需重跑 ASR。
- **W5** 最小比较族（v3.1 已唯一冻结，见 v3.1 增补 Gate 0 第 3 条）：R2 十二条件 Holm 族、Table VII B-vs-LA 主比较 + A/B 验证性比较、Table III 总体主比较 + 三分组 Holm 族、R5 仅报 B−A 配对均值差 bootstrap CI。固定 SciPy 版本、Wilcoxon `correction`、rank-biserial 符号方向、全零差处理。

### 3. 流程更新（Gate 1）

W1 实现 + self-test → 脚本级审查（对照 r2 §5 Gate 1 十二项）→ GPU 冒烟 3 条（schema/因果顺序/非负 TTFA/ns 闭合/PCM 格式/错误落盘）→ 正式 50×2 + 子集补轮 + 匹配文本控制 → 结果级 QA（r2 §6 清单）→ 总册定稿 → 论文修改（范围按 r2 §7 限定）。

---

## 一、工作项总览（v2）

| # | 内容 | 执行侧 | 对应审计项 |
|---|---|---|---|
| W1 | 配对 A/B 同时间轴 TTFA 直接实测（协议见 §二） | GPU 主机 | P0-1/P0-2 |
| W2 | 环境记录补齐 | GPU 主机 | P1-5 |
| W3 | CV 统一口径重算（ddof=1 全分布） | 本机 | P0-3 |
| W4 | macro/corpus WER/CER 双口径补算 | 本机 | P0-4 |
| W5 | 成对统计推断（协议冻结，见 §五） | 本机 | P1-1 |
| W6 | 语义复现元数据（扩大版） | 本机 | P1-4 |
| W7 | 真人语音 QA（人工试听 + 文档说明） | 用户 + 本机 | P1-2 |
| W8 | 写作总册全链路同步（两阶段） | 本机 | P0-5 |
| W9 | LocalAgreement 方法与排除规则说明 | 本机 | 复审 §5.1 / 初审 P1-3 |

## 二、W1 TTFA 协议（正式运行前冻结，复审 §3 全量落实）

### 2.1 实验结构与顺序

- 表述统一为：**同一批样本、相同输入音频与配置下，分别执行的配对 A/B 独立请求**（A/B 为互斥管线，不存在"同一请求"）。
- 样本：E4 同 50 样本清单；锁定配置不变（Whisper turbo cuda:0 / Qwen2-7B cuda:1 / chunk 500ms / prefix=1 / suffix=0 / threshold 2.0s / max_tokens=128）。
- **AB/BA 平衡顺序**：25 条 A→B、25 条 B→A，按语言×时长分层平衡，确定性 seed 预生成顺序表；顺序表、seed、SHA-256 落盘。
- 唯一键 `sample_id × mode × repeat_idx`；每条键的 A/B 输入 WAV SHA-256 一致。
- **重复性子集**：分层抽 10 条（语言×时长），A/B 各 3 轮；报告子集 CV（median/mean）与顺序效应。主实验 50 条 ×A/B 各 1 轮。论文表述据此为"主实验单轮配对 + 10 条子集三轮重复"，不把 E1 的 TTFT 重复性外推到 TTFA。

### 2.2 三个概念分开记录（不包装、不混名）

1. `source_trailing_silence_ms`：源 WAV 物理语音结束后的自带尾静音（离线分析得出）；
2. `trailing_feed_wait_ms = feed_end_ns − physical_speech_end_ns`：可超过 500ms，不再设"必然≤500ms"验收，源尾静音/chunk 量化/调度误差分别报告；
3. 本运行不实现在线 VAD endpoint decision（E4 式 feed 结束立即 flush），如实记录 `explicit_flush_time`；**不声称测量了 VAD endpoint latency**。E5 负值问题以"该字段为异时间轴量、不入新表"方式处理，不以改名宣称"修复"。

### 2.3 physical speech end 冻结定义

- 分析波形统一重采样为 16kHz mono（源格式运行时校验，不假设；细则以 v3.1 增补第 1 条为准）；能量法：25ms 窗 / 10ms hop，RMS 门限 = max(全文件 RMS 的 −40dB 相对值, 底噪估计 +6dB)，末段语音终点 = 最后一个超门限窗的右边界；
- Silero VAD 复核：固定版本与参数（threshold=0.5、min_speech=250ms、min_silence=100ms、padding=30ms，以脚本常量落盘），取最后一个 speech 段的 end；
- 裁决规则：两法差 >200ms 时取 Silero 值并标记 `pse_method=silero_fallback`，逐样本记录两法原始值；
- 输出 `estimated_physical_speech_end_sample`、源 WAV SHA-256、sample→单调时钟映射公式（逐 chunk 推送时间戳线性内插）；
- 原始区间一律 `time.perf_counter_ns()`；UTC 墙钟仅入 RUNINFO，不参与延迟计算。

### 2.4 逐条原始事件字段（A/B 均存，N/A 显式置空）

`clock_type, playout_start_ns, physical_speech_end_sample, physical_speech_end_ns, last_input_sample_ns, feed_end_ns, explicit_flush_start_ns, explicit_flush_done_ns, pipeline_input_close_ns, full_input_ready_ns(A), last_asr_commit_ns, asr_processing_done_ns, first_model_token_ns, first_content_token_ns, first_sentence_boundary_ns, generation_end_ns, tts_request_start_ns, tts_response_headers_ns, first_pcm_byte_ns, first_playable_pcm_ns, tts_done_ns`（v3.1 定稿字段，`endpoint_decision_or_flush_ns` 已弃用），另存 `response_token_count, generation_stop_reason(eos|max_tokens|error), sentence_end_found, sentence_fallback`。句末检测与 TTS 请求发出是两个独立时间戳，不得合并。"完整回复"注明为 max_tokens=128 capped response。

### 2.5 直接指标与组件闭合

- `TTFA_received = first_pcm_byte_ns − physical_speech_end_ns`；`TTFA_playable = first_playable_pcm_ns − physical_speech_end_ns`（**主表只用 playable 口径，不混排**；HTTP/格式/对齐/零内容错误整行 error、不降级；playable 语义=已收齐最低播放缓冲，**不称 first audible**，received 仅作辅助披露口径单列）。
- `first_playable_pcm_ns`：确认响应为预期裸 PCM（非 WAV 头/错误 JSON）、字节按 2 字节对齐、累计达 ≥30ms（20–40ms 带内取 30ms 固定值）可播放缓冲的时刻；记录首块字节数、对应音频时长、RMS、peak。
- 组件分解：输入尾部等待 → flush→first token → first token→TTS 文本就绪 → 文本就绪→请求发出 → 请求→first playable PCM。全部由同一组连续 `*_ns` 差分构成，**闭合残差应为数值舍入级（<1ms）**；存在无法归入的异步间隔时单列为命名调度项，不留残差口。取消 v1 的固定 |Δ|≤50ms。
- 单调性断言：同一记录内事件时间戳按物理因果序非递减，违例即该行 error。

### 2.6 TTS 策略与匹配文本控制

- 主协议：B 首句就绪即调 TTS；A capped full response 结束后调 TTS（保持系统定义）。逐条保存 `tts_text, tts_text_sha256, tts_n_chars, tts_n_bytes_utf8, tts_text_source(first_sentence|capped_full_response)`。
- 论文表述：A/B TTFA 差异为**两套系统策略的总体差异**，不全归因于 ASR/KV。
- **匹配文本 TTS 控制**（分层 ≥10 条子集，仅 TTS 调用，不重跑管线）：每样本测 (i) B 首句文本重测（与主实验对照得 TTS 运行方差）、(ii) A 回复首句文本、(iii) A 全文重测；另加一条固定校准句（中英各一）。用于分离"早启动策略"与"输入长度/内容"两类影响。
- 未接真实声卡：论文声明未计入声卡/播放器缓冲延迟。

### 2.7 工程保护

`schema_version`、`run_id`；checkpoint 存配置 hash；恢复时校验 commit/dirty、样本清单、音频 hash、模型、TTS、端点算法、顺序表、生成参数；**显式输入路径，不用 glob 猜最新**；timeout/HTTP 非 200/空响应作为失败落盘，不静默重试冒充成功；固定并保存 Python/NumPy/Torch/CUDA seed 与实际生成参数。

### 2.8 产物与 QA

`r7_ttfa_unified/`：per-sample JSONL（全原始时间戳）+ summary CSV（zh/en/ALL × A/B × received/playable，mean/std/P50/P90/P95）+ 重复子集 CV + 顺序表 + RUNINFO + QA 报告（50/50 配对、WAV hash 一致、无负值、闭合残差分布、error=0、单调性断言全过）。

## 三、W2 环境记录（GPU 主机）

`lscpu`、`uname -a`、`free -h`、NUMA/线程设置、CPU governor/frequency、`nvidia-smi`、TTS 服务 commit/模型 revision/运行配置、uv.lock 摘要，存 `r7_ttfa_unified/env/`。

## 四、W3 / W4 口径冻结（本机）

- **W3**：`CV_i = std(x_i1..3, ddof=1)/mean(x_i1..3)`，百分比输出；样本须恰 3 轮、同模式同配置无 error；P90 用线性插值（numpy 默认）并注明；逐样本明细 + mean/median/P90/max + CV>5% 数量与比例；三个输入文件路径 + SHA-256 落盘；**所有论文表格 std 定义统一注明**（不只 CV 摘要）。
- **W4**：逐样本分别保存 `wer_S/D/I/N` 与 `cer_S/D/I/N` 八列；汇总按语言×条件×系统输出 mean-utterance 与 corpus 两套 + ΣS/ΣD/ΣI/ΣN + n + 配对过滤规则；`corpus = Σ(S+D+I)/ΣN` 恒等式自检；A/B 用同一 sample ID 集合并生成 paired filter manifest；en case-fold、zh 去接缝空格、标点处理复用既有归一化实现（不改 `run_exp_quality.normalize_text`）；全文档清除 AISHELL 旧值 6.72%，10.77%/11.80% 标注为 mean-utterance 宏平均口径，corpus 值以新结果为准。

## 五、W5 配对统计协议（冻结）

- bootstrap：以 sample ID 为配对重采样单位，10,000 次，固定 seed（落盘），percentile 95% CI；
- 改善率统一定义 `(mean(A)−mean(B))/mean(A)`，不与逐样本改善率均值混用；
- Wilcoxon：双侧、`zero_method='wilcox'`、`method='auto'`（scipy 默认 exact/asymptotic 切换），报告统计量与原始 p；
- R2 多条件族内 Holm 校正，报告校正 p；
- 效应量：rank-biserial correlation + paired Cohen's dz；
- 输出：n、点估计、CI、统计量、原始 p、校正 p、效应量、过滤规则；
- 文字规则：CI 跨 0 或校正 p 不达标不写"统计显著"；不显著≠等价；R5 单裁判结果即使 p 不显著也不写"statistically equivalent/indistinguishable"。

## 六、W6 / W7 / W8 / W9 范围（按复审扩大）

- **W6**：两组参数分开记——被评 A/B 回复（temperature=0.1, top_p=0.9, `requested_repetition_penalty=1.1` / `effective_repetition_penalty=not_applied`, max_tokens=128）与 judge（temperature=0，judge max_tokens 另记）；BGE-M3 model+tokenizer revision/hash 与 HF snapshot commit；judge 完整模型 ID、供应商/base URL、请求时间；prompt/输入/回复/原始响应 hash；失败与重试记录；样本范围=50 条 Very Long 合成样本；无法回溯的服务版本写 unknown，不事后推断；语义结论保持探索性定位。
- **W7**：试听记录含 5 条 ID（中英均覆盖、含 extra-long）、试听者、日期、截断/错序/爆音/异常静音/音量检查项，结论标 manual spot check（不称 human evaluation）；文档补：同章节/同说话人拼接、人工静音构造、`reference`/`reference_full` 恢复校验过程、babble 空输出率（LibriSpeech 12/30、AISHELL-1 5/30）、error=0 仅表示程序未崩溃。
- **W8**：同步 `PAPER_WRITING_REFERENCE.md`/`PAPER_HANDOFF.md`/`REVISION_CHANGELOG.md`/`experiments/EXPERIMENT_DESIGN.md`/`experiments/CISR_REVISION_PLAN.md`（完成状态或"后续口径以 v3.1 审计方案为准"说明）/最终 CSV·RUNINFO/论文正文与回复信；建立"主张→来源文件→生成脚本→论文位置"清单，覆盖复审 §5.4 所列 12 项及 v3 再审 §7"不得写"清单（含 repetition_penalty 未生效表述）。
- **W9（新增）**：LocalAgreement 说明——命名"项目内实现的 LocalAgreement-2-style baseline"；描述 absolute audio timeline / sentence-boundary-aware trimming / la_max_buffer_s=15.0 / punctuation-robust agreement；披露 505→498（3 运行错误 + 4 条流式 TTFT>10s、成对排除、未过滤失败率 7/505）；汇总脚本白名单最终文件、排除 `invalid_dev3_frame_bug/`；"质量同量级"不写成"统计等价"。

## 七、实施顺序与预估

1. 本方案 v2 + 回复函 → 复审；
2. 通过后：本机实现 W3/W4/W5（含 self-test）并执行；W6/W9 文档；同步实现 W1 脚本（含 self-test 与协议断言）；
3. W1 脚本审查 → GPU handoff（含 W2 清单、TTS 探活前置、冒烟 3 条 → 正式 50×2 + 重复子集 10×2×3 + 匹配文本控制）；
4. 结果返回 → 结果级 QA → 装配新 Table VIII（TTFA_playable 主口径）；
5. W8 阶段 2 → 按初审 §8 验收标准逐项对齐 → 整改回复 → 论文修改。

GPU 预估：主实验 50×2 约 1.5–2.5h + 重复子集约 1–1.5h + 控制测量分钟级，合计约 3–4h。

## 八、论文数字预判（更新）

Table VIII 将被同时间轴实测替换：B 预计与现值同量级；**A 为首次实测，可能与 22.67s 估计差距较大**；新增 TTFA_received/playable 双口径、源尾静音与 feed wait 分列、TTS 策略差异归因与控制实验。核心结论（70%–74% 改善、流式 ASR 主收益、KV 边际收益小、babble 失败边界、TTS 为可听响应主要瓶颈）不变。
