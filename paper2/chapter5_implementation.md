# 第五章 系统实现

## 5.1 模块架构

系统复用一期基于 Whisper[17] 的流式 ASR 与 LLM 增量预填充，并增加对话编排、输出断句、流式 TTS、软件播放状态和实验驱动。表 5-1 将模块与本文的验证对象对应起来。

**表 5-1　主要模块、接口与验证对象**

| 模块 | 主要职责 | 关键验证对象 |
|---|---|---|
| `src/dialogue/timeline.py` | 关联 TTS 文本片段、token 区间、音频块与累计软件采样区间；按 software-consumed cursor 反查 | token/sample 连续性、chunk 唯一归属、乱序拒绝、裁剪端点 |
| `src/llm/stream_llm_inference.py` | 增量预填充、assistant 侧 KV 累积、裁剪、EOT 与角色恢复 | KV/mask/ledger 长度一致，role/end 状态与 position 连续 |
| `src/tts/sentence_chunker.py` | 基于 stream2sentence[21] 将 token 解码流切分为 TTS 文本片段并关联 token 区间 | 非空白字符守恒、空片段处理、末端钳制 |
| `src/tts/streaming_tts.py` | 流式 TTS 接口、Mock TTS 与真实后端适配 | 片段—音频块归属、画像参数读取 |
| `src/player/player.py` | 维护软件已消费采样游标和停止接口 | 采样计数、单调查询与 seek 语义 |
| `src/dialogue/trigger.py` | 计算话轮完成置信度 | 类别 token 配置与阈值触发记录 |
| `src/dialogue/orchestrator.py` | 串联用户输入、候选生成、断句、TTS、打断与状态修正 | 推测作废、同步 oracle 接受、EOT/role 状态和逐轮指标 |
| `experiments/sci34_supplement/` | 固定轨迹 E3、A1/P1、确认性 E1/E2 与 C2 协议 | 配对目标、计时边界、crossed analysis、exact crop/recovery gates |

机制实现与实验实例化相互分离：编排器面向统一接口，实验脚本选择真实后端或画像驱动后端。本文的 playback 状态仅来自软件已消费采样游标，不包含 device-presented sample clock 或 acoustically heard content measurement。

## 5.2 片段时间轴与断句对齐

`PlaybackTimeline` 以 TTS 文本片段为主轴。片段生成后登记文本及 assistant 内容 token 区间；TTS 音频块随后附加到所属片段，并更新累计软件采样区间。播放器维护 software-consumed-sample cursor，打断查询据此定位片段并返回该片段末 token 作为合法保留边界。

采样区间采用 $[0,p)$ 已由软件消费的语义。当 $p=\operatorname{se}(f_k)$ 时，游标恰好覆盖片段 $f_k$，因此保留该片段。实现显式处理片段中部、片段末端、首采样、空时间轴和越过末端等边界。工件中的 `heard_text`、`n_heard` 等字段仅是兼容别名，不表示声学真值。

写入 API 强制生产者合同。`add_fragment()` 要求非空、单调且连续的 token span；重复、重叠、倒退或 gap 均失败。片段 span 冻结后才能 `attach_chunk()`；关闭、丢弃或完成的片段拒绝追加。`chunk_id` 全局唯一且只能归属一个片段，sample range 必须非空并与累计端点连续；软件游标也必须单调且不越过已登记范围。锁只保证短更新的原子性，不替代上述检查。

断句器输出文本，而缓存状态使用 token 端点。系统在逐 token 解码时累计非空白字符数，再以片段的非空白字符长度定位其末 token。该映射不假设 token 边界与标点或词边界一致，但要求全部片段拼接后保持非空白字符序列守恒；纯空白片段被跳过，区间末端被钳制到实际内容 token 数。

## 5.3 KV、token ledger 与角色状态

### 5.3.1 联合状态容器

一期调用方只保留用户侧预填充缓存，assistant 生成 token 的 KV 未持续写回，无法在播放期裁剪。二期 `AccumKVCache` 联合维护 `DynamicCache`、attention mask、完整 `token_ids` ledger、当前 assistant 内容 ledger、`RolePhase`、`GenerationEndReason` 及 assistant role/content 边界。每个稳定操作后均检查

$$
\operatorname{len}(token\_ids)
=\operatorname{len}(pre\_attention\_mask)
=\operatorname{seq\_length}
=\operatorname{DynamicCache.get\_seq\_length}().
$$

同时，assistant 内容 ledger 必须与完整 ledger 中的当前内容 span 一致；裁剪后的 position IDs 按实际 past length 重算。

### 5.3.2 tokenwise append、EOT 与结束原因

`generate_accumulating()` 先选择候选 token。普通内容 token 再经一次 cache-update forward 写入 KV，同步追加 mask 与两个 ledger，随后才由 generator yield。因此首 token 回调表示 candidate selection/internal compute-readiness，而不是 consumer 可见或可交付 token。

结构性 EOT 不作为 assistant 内容写入 KV，不进入内容 ledger、TTS 片段或 timeline。模型选择 EOT 后只进入 `ASSISTANT_EOT_PENDING`，将结束原因记为 `EOS`；`reopen_user_role()` 是唯一 assistant close commit，恰好一次追加从 tokenizer chat template 推导并校验的 assistant-close 与下一 user-open token。max-token、consumer-stop 和 crop 分别记录为 `MAX_TOKENS`、`CONSUMER_STOP` 与 `CROPPED`，避免从生成长度或末 token 反推结束原因。

### 5.3.3 crop 与角色恢复

`crop_to_token(N)` 同步裁剪 K/V、mask、完整 ledger、assistant 内容 ledger 和 span，并由保留序列恢复 role/end state；结构 token 内部不是合法 crop 点。播放期保留零个内容 token 时裁到 assistant content start，保留 assistant header 和 `ASSISTANT_OPEN`，随后正常提交一次 close。整段候选作废则回到推测前的 assistant role start，删除 header 与全部候选内容，并恢复 `USER_OPEN`。

裁剪后的 `CROPPED` 仅保留到下一次合法状态推进；成功追加新 user 内容后恢复 `USER_OPEN + NONE`。`open_assistant_role()`、`prefill_assistant_text()` 与 `reopen_user_role()` 也分别更新 end reason。角色边界由 tokenizer chat template 的规范 tokenization 推导，故当前适用范围限于可可靠提取这些 token 的模板。

## 5.4 编排与事件语义

`DialogueOrchestrator` 提供一次性用户文本路径和增量文本段路径。前者用于 System A，后者用于 B 系列候选生成工作点；输出 token 经断句和 TTS 后登记到时间轴，再按条件执行打断与状态修正。

确认性 E1/E2 runtime 区分以下事件：`last_segment_arrival_ns` 是最后一个预切分文本段到达；legacy `first_token_ready_ns` 是首候选 token 的选择/内部计算就绪回调；`endpoint_accept_ns` 是候选处理后的同步 oracle 接受；`first_deliverable_token_ns` 与 `consumer_delivery_ns` 是同步程序的诊断 marker。后两者受“先处理候选、后接受/消费”的程序顺序支配，不作为生产 deliverability 指标；候选选择到 oracle 接受的间隔也不能解释为自然端点提前量。

Mock TTS 由 CosyVoice2 六句真机画像参数化，以每非空白字符采样数、首块延迟和实时率构造确定性时长近似。该实现便于控制软件游标，但不等价于真实异步合成、应用队列、audio API、OS/驱动/设备缓冲或声学传播；相应 mouth-to-ear 量仅是模型计时与 TTS 画像的组合估计。

## 5.5 实验协议的实现

确认性 E1/E2 使用 100 个唯一 holdout utterances、5 个独立初始化进程 session 和 10 个条件；每个 session 重新加载模型，并在同一组话语上运行 System A、八个数值阈值及 never-speculate B 对照。正式 `analysis_v2.json` 以 crossed/product bootstrap 独立重采样全局 session 与 utterance，再对笛卡尔积重算估计量。C-E1 比较 full-string tokenization/full prefill 与 segment-wise tokenization/incremental forward 两条非 token-equivalent implementation paths；C-E2 则在 token-consistent 的 B@0.92 与 B-never 之间比较。

固定轨迹 E3 先为每条对话生成一次被打断 assistant token 轨迹，再由同一轨迹导出四个注入位置和 playback/generation 两个条件。后续两轮 probe 按 retained-history key 复用；目标字段非空决定 target-specific 资格。正式重分析报告 label-weighted 主 estimand，并以 dialogue-weighted 和 target-specific exact-key deduplication 检查对话权重及重复标签的影响。

A1 在 256–8192 token 上使用 5 次预热、50 次正式重复、固定操作顺序和固定 32-token suffix；crop+role 与 re-prefill 计时均执行 CUDA/GPU 同步。P1 使用 3 个上下文长度、3 个软件游标比例和每单元 20 次重复，共 180 条 prepared-state headless 记录。两者分别是模型侧固定协议微基准与软件控制路径测量，不代表完整 barge-in。

## 5.6 C2 v3 的 exact-only 验证实现

C2 v3 复用冻结的 24-case 网格，包含 27 个 crop event、3 个 no-op 和 60 个后续恢复步骤。fixture 的 308 个 assistant 内容 token 均经 production `generate_accumulating()` 逐 token append。

本文所称 **slicing oracle** 是独立于 production crop 接口、但取自同一 pre-crop snapshot 的逐层切片对照。对目标长度 $N$，oracle 将 snapshot 中每层 key/value 张量沿序列轴直接复制 `[..., :N, :]`，同时复制 `attention_mask[:, :N]` 与 `token_ids[:N]` 并将 sequence length 置为 $N$；它不调用 `crop_to_token()`，也不执行 clean re-prefill。验证器不信任记录中的 stored keep，而根据 case、fragment token partition、role/content boundaries 及第二次裁剪比例独立推导 $N$。

production arm 是对同一 snapshot 唯一调用 `crop_to_token(N)`。每个事件比较 pre-crop retained prefix、production post-crop 与 slicing oracle 的逐层 K/V shape、dtype、device、hash 和运行时 `torch.equal`，并检查 mask、完整 ledger、sequence length 与 KV length。wrong-length disposable control 将 $N$ 改为错误值，验证 shape/manifest gate 能检出偏差。

裁剪后，production 与 oracle 两臂在同一 accepted run 内从精确匹配的保留状态出发，接收相同 token-ID chunks 和相同操作序列；每步比较 K/V、logits、mask、完整 ledger、retained-prefix hash，以及独立按操作序列推导的 role/end/content state。本文将这一性质称为 **within-run matched-arm recovery exactness**。它不表示跨进程、跨设备可重复性，也不表示 clean-reprefill 或不同 forward topology 的数值等价。

## 5.7 环境与复算入口

**表 5-2　正式工件的环境与复算入口**

| 项目 | 冻结信息 |
|---|---|
| 主模型与精度 | Qwen2-7B-Instruct；BF16；SDPA；C2 模型 artifact identity 见 accepted manifest |
| 自动裁判 / 话轮检测 | Mistral-7B-Instruct-v0.3 / TEN Turn Detection；模型 identity 见各 campaign manifest |
| 软件栈 | Python 3.10.18；PyTorch 2.8.0+cu128；Transformers 4.57.1；CUDA 12.8；cuDNN 9.10.2 |
| 设备 | NVIDIA RTX 3090（24 GB）；确认性 E1/E2 使用两张卡，其余 campaign 按各 manifest 记录 |
| 权威工件入口 | 仓库根目录 `REPRODUCIBILITY.md` |
| 关键 accepted analyses | C-E1/C-E2 `analysis_v2.json`；E3 `analysis_weighting_dedup_v2.json`；C2 v3 `analysis_v1.json` 与 `validation.json` |

`REPRODUCIBILITY.md` 是 campaign 状态、正式 run、模型身份、分析文件和复算命令的稳定索引。不同 campaign 不构成可池化的计时总体；分析复算与模型重跑的依赖范围亦以该索引和 campaign manifest 为准。

## 5.8 本章小结

系统将 software-consumed cursor、TTS fragment、assistant token span 和联合缓存解释状态连接起来。时间轴 API 强制连续性、生命周期与唯一归属；缓存容器同步维护 KV、mask、ledger、position、role 与 end reason，并以唯一 EOT close commit 避免结构 token 重复入账。C2 v3 通过同一 pre-crop snapshot 上的 production crop、逐层 slicing oracle、wrong-length control 和 within-run matched-arm recovery 构造可证伪的 exact gate；环境与复算范围由 `REPRODUCIBILITY.md` 统一索引。