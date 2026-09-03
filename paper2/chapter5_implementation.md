# 第五章 系统实现

## 5.1 模块架构

系统复用本项目一期内部实现的、基于 Whisper[17] 的流式 ASR 和 LLM 增量预填充能力，并新增对话编排、输出断句、流式 TTS、软件播放状态和实验驱动程序。表 5-1 给出主要模块及其验证对象。

**表 5-1　主要模块、接口与验证对象**

| 模块 | 主要职责 | 关键验证对象 |
|---|---|---|
| `src/dialogue/timeline.py` | 保存 TTS 文本片段、token 区间、音频块和累计软件采样区间；按 software-consumed cursor 反查 | token/sample 连续性、chunk 唯一归属、乱序拒绝、裁剪端点 |
| `src/llm/stream_llm_inference.py` | 增量预填充、assistant 侧 KV 累积、缓存裁剪、EOT 与角色恢复 | KV/mask/完整 ledger/内容 ledger 长度一致，role/end 状态与 position 连续 |
| `src/tts/sentence_chunker.py` | 基于 stream2sentence[21] 将 token 解码流切分为 TTS 文本片段并关联 token 区间 | 非空白字符守恒、空片段和末端钳制 |
| `src/tts/streaming_tts.py` | 定义流式 TTS 接口、Mock TTS 与真实后端适配 | 片段—音频块归属、画像参数读取 |
| `src/player/player.py` | 维护软件已消费采样游标和停止接口 | 采样计数、单调查询与 seek 语义 |
| `src/dialogue/trigger.py` | 计算话轮完成置信度 | 类别 token 配置、阈值触发记录 |
| `src/dialogue/rewriter.py` | 重写被打断轮的保留前缀 | 调用耗时、输出非空和替换路径 |
| `src/dialogue/orchestrator.py` | 串联用户输入、候选生成、断句、TTS、打断和状态修正 | 推测作废、oracle 接受、EOT/role 状态、逐轮指标 |
| `experiments/scripts/` | 旧二期实验驱动程序及离线重分析 | 旧结果保护、fixture 清洗和统计复算 |
| `experiments/sci34_supplement/` | 固定轨迹 E3、联合 A1、prepared-state P1、确认性 E1/E2、C2 v1/v2/v3 | 配对目标、计时边界、crossed analysis、exact crop/recovery gates、seal |

上述结构把机制实现和实验实例化分开：编排器面向统一接口，实验脚本决定使用真实后端还是画像驱动的模拟后端。本文中的 playback 状态只来自软件已消费采样游标，不包含 device-presented sample clock 或 acoustically heard content measurement。

## 5.2 片段时间轴与断句对齐

`PlaybackTimeline` 以 TTS 文本片段记录为主轴。片段生成后首先登记文本和 assistant 内容 token 区间；TTS 每产生一个音频块，就将其附加到对应片段并更新累计软件采样区间；播放器维护 software-consumed-sample cursor。打断查询读取这一游标，在片段采样区间中定位命中记录，并返回片段末 token 作为保留边界。

计数边界采用 $[0,p)$ 已被软件消费的语义。当 $p=\operatorname{se}(f_k)$ 时，软件游标恰好覆盖片段 $f_k$，应保留该片段而不是落到下一片段。实现分别处理片段中部、片段末端、首采样、空时间轴和越过末端等情况，避免大于/大于等于选择导致一个片段偏移。这里的 `heard_text`、`n_heard` 等 legacy 字段仅为 artifact compatibility alias，不能解释为声学真值。

写入 API 不依赖调用方“自觉有序”，而是显式验证生产者合同。`add_fragment()` 要求 token span 非空、单调且与上一片段连续；重复、重叠、倒退或 gap 均抛错。片段 token span 冻结后才能 `attach_chunk()`；已关闭、已丢弃或已完成的片段拒绝追加。每个 `chunk_id` 必须全局唯一并归属唯一 fragment，chunk 按所属片段和生成次序附加。sample range 必须非空且与全局累计采样端点连续，禁止 gap、overlap 和倒序。软件游标必须单调且不越过已登记范围。测试覆盖错序 fragment、重复/错属 chunk、关闭后追加、sample gap/overlap 与 cursor regression，并要求全部 fail-closed，而不是重排或静默修正。

断句器产出文本字符串，LLM 状态使用 token 端点。系统在逐 token 解码时累计非空白字符前缀和；文本片段产出后，用其非空白字符长度定位最后一个被覆盖的 token。该方法不假设 tokenizer token 边界与标点或词边界一致，但要求片段拼接后的非空白字符序列与原始生成文本守恒。实现跳过纯空白片段并把区间末端钳制到实际内容 token 数，以防空片段错误消费下一 token 或产生越界 crop。

时间轴使用一把锁序列化短小的片段与 chunk 更新。锁只解决并发原子性，不替代 token span、fragment lifecycle、chunk ownership 和 sample ordering 验证。第六章未测量高并发尾延迟，因此本文只主张这些生产者不变式与乱序拒绝行为，不声称任意并发规模下的性能。

## 5.3 KV、token ledger 与角色状态

### 5.3.1 统一缓存容器

一期生成循环的调用方只持有用户侧预填充缓存，assistant 生成 token 的 KV 未持续写回，因而无法支持播放期裁剪。二期 `AccumKVCache` 将以下对象作为一个逻辑状态维护：

- `past_key_values: DynamicCache`；
- `pre_attention_mask`；
- 覆盖完整缓存序列的 `token_ids` ledger；
- 仅含当前 assistant 内容的 `assistant_token_ids` ledger；
- `RolePhase`；
- `GenerationEndReason`；
- assistant role start、assistant content start/end 等边界。

所有 user/assistant 文本、role transition 与恢复 token 都经 token-ID prefill 核心追加。每次稳定操作后强制检查

$$
\operatorname{len}(token\_ids)
=\operatorname{len}(pre\_attention\_mask)
=\operatorname{seq\_length}
=\operatorname{DynamicCache.get\_seq\_length}(),
$$

并要求 `assistant_token_ids` 与 `token_ids` 的当前 assistant content span 完全一致。后续 position IDs 由真实 past length 重算，不沿用 crop 前的位置。

### 5.3.2 tokenwise production append 与 EOT

`generate_accumulating()` 先执行 token selection。普通内容 token 随后单独通过 cache-update forward 写入 KV，并同步追加 mask、完整 ledger 与 assistant 内容 ledger，再由 generator yield。首候选 token 回调发生在 selection 后、这次 forward 与 yield 前，所以记录的是 first-candidate-token selection/internal compute-readiness，不是可交付 token。

结构性 EOT 使用专门语义。生成器选择 EOT 后：

1. 不执行将该 EOT 作为 assistant 内容写入 KV 的 forward；
2. 不追加至 `assistant_token_ids`、TTS 文本片段或 timeline；
3. 将 `RolePhase` 置为 `ASSISTANT_EOT_PENDING`；
4. 将 `GenerationEndReason` 置为 `EOS` 并终止内容生成。

`reopen_user_role()` 是唯一 assistant close commit。它根据 tokenizer chat template 推导并校验 assistant-close 和下一 user-open 的 token IDs，恰好一次写入完整 ledger、mask 与 KV，但不写入 assistant 内容 ledger。这样，预测 EOT 不会与角色恢复重复注入。max-token、consumer-stop 和 crop 分别记录 `MAX_TOKENS`、`CONSUMER_STOP` 和 `CROPPED`，不再通过生成 token 数或 ledger 末 token 推断结束原因。

### 5.3.3 crop 与状态恢复

`crop_to_token(N)` 对 K/V、mask、完整 `token_ids` ledger、assistant 内容 ledger 与 span 同步裁剪，并根据保留序列恢复 `RolePhase` 和 end state。结构 token 内部不是合法 crop 点。播放期保留零个内容 token 时，目标是 assistant content start：assistant header 仍在，角色维持 `ASSISTANT_OPEN`，随后通过唯一 close commit 正常结束。整段候选作废时，目标是 assistant role start 之前的推测快照：assistant header 与全部内容一起删除，角色恢复 `USER_OPEN`。

crop 后 `GenerationEndReason.CROPPED` 在下一状态推进前保留。D-022 修复要求 `prefill_user_text()` 成功追加新 user 内容后立即将其清为 `NONE`；`open_assistant_role()`、`prefill_assistant_text()` 与 `reopen_user_role()` 也按各自的新阶段更新 end reason。编排器在追加 user segment 后 fail-closed 检查 `USER_OPEN + NONE`，避免陈旧 `CROPPED` 穿透到下一轮。

角色转换不硬编码固定 ChatML 字符串，而从 tokenizer chat template 的规范 tokenization 推导并验证。当前适配范围因此限定为能够可靠提取这些边界 token 的模板，不声称适用于任意因果语言模型。

## 5.4 编排与事件记录

`DialogueOrchestrator` 提供一次性用户文本路径和增量文本段路径。前者用于 System A，后者用于 B 系列推测工作点。输出侧逐 token 生成，断句器产出 TTS 文本片段，TTS 后端登记音频块与软件采样区间，再按实验条件注入打断并选择 software-cursor 或 generation 历史策略。

确认性 E1/E2 runtime 将事件明确拆开：

- `last_segment_arrival_ns`：最后预切分文本段进入路径；
- legacy `first_token_ready_ns`：首候选 token 选择回调，属于内部 compute-readiness；
- `endpoint_accept_ns`：候选处理之后的 oracle acceptance；
- `first_deliverable_token_ns`：同步 harness 的诊断 marker；
- `consumer_delivery_ns`：同步 harness 的 consumer 诊断 marker。

后两项受同步程序“先处理候选、后接受/消费”的顺序支配，只用于诊断，不作为生产 deliverability headline。尤其不能把候选选择到 oracle 接受的内部间隔解释为自然 endpoint lead 或用户继续说话时长。

实验中的 Mock TTS 由 CosyVoice2 六句真机画像参数化，使用每非空白字符采样数、首块延迟和实时率构造确定性时长近似。它可以统一控制软件游标比例并保留平均时长尺度，但不等价于真实异步合成、应用队列、audio API、OS/驱动/设备缓冲、线程唤醒或声学传播。mouth-to-ear 因此只是“计算时间与 TTS 画像组合建模”，不是完整音频闭环实测。

## 5.5 实验实现与 versioned analysis

### 5.5.1 E1/E2 交叉重复

确认性受控文本段 run `e1e2c_b8c758b_20260901T173306Z` 使用 100 个唯一 holdout utterances、5 个独立初始化 Python process sessions 与 10 个条件。每 session 重新加载模型，并在同一 100 个 utterances 上运行一次性 System A、八个数值阈值和一个 never-speculate B 对照；因此每条件有 500 个 session×utterance observations，总计 5000 条 records。内容采样单位仍是 100 个 utterances，5 个 session 仅是技术重复。

正式统计使用不覆盖 `analysis_v1` 的 `analysis_v2.json`。crossed/product bootstrap 每次独立抽样全局 session 与全局 utterance，再取笛卡尔积并在 cell 内保持条件配对；10,000 repeats，seed 20260901，报告 percentile 95% interval。该实现匹配“同一批话语跨五个进程重复”的交叉设计，而不是把 dialogue 错当为嵌套于 session。

TEN 置信度通过 222 条目的离线 replay cache 逐段回放，不进入延迟窗口。模型固定 Qwen2-7B-Instruct、greedy、BF16 与 SDPA，holdout 从本地 MultiWOZ 2.1 确定性派生，并与旧 E1/E2 及固定轨迹 E3 的样本 ID 隔离。

C-E1 是非 token-equivalent 的 implementation-path comparison。System A 对 full string 一次 tokenization/full prefill；System B 对 segments 分别 tokenization 并走 incremental forward。正式记录中，A 与 B@0.92 的完整 `output_token_ids` exact 为 280/500，首 token exact 为 465/500，长度/EOS/max-token agreement 为 495/500；44/100 个唯一 utterances 至少一次完整输出分岔，且五个 session 中均为相同 44/100。差异可能来自 segment-wise tokenization、forward topology/shape、role boundary、kernel 与 Python scheduling，不能归因为纯 incremental-prefill effect，也不能按 280 个 matched outputs 过滤主延迟。

C-E2 比较 B@0.92 与 B-never；二者完整 tokens、首 token、长度、EOS/max-token 与文本均为 500/500 一致，因此是 token-consistent B-path comparison。该一致性不改变 C1 只属于同步 candidate-readiness/oracle characterization 的范围。

### 5.5.2 E3、A1 与 P1

E3 accepted run 保存 100 dialogues、400 `(dialogue,injection_label)` pairs、800 condition records 与 1600 judge records。versioned weighting/dedup `analysis_v2` 以 label-weighted estimand 和相同 label-weighted dialogue-cluster bootstrap 为主，另报 dialogue-weighted estimand 与 target-specific unique-semantic-boundary sensitivity；10,000 repeats，seed 20260831。它只测 fixed-detector-conditioned information-reproduction rate，不把规则或 Mistral judge 称作人类真值。

A1 对 256–8192 token contexts 使用 5 warmups、50 repeats，固定 operation order，并固定移除 32-token suffix。joint crop+role 与 re-prefill 的计时均有 CUDA/GPU 同步，但结果只适用于这一固定模型侧协议，不能外推到自然打断位置、其他 crop length 或完整 barge-in。

P1 使用 3 个 context lengths、3 个软件游标比例与每 cell 20 repeats，共 180 条 prepared-state headless records。stop→crop 与 stop→role 是嵌套累计区间。P95 是每 cell 20 个数值形成的 empirical/descriptive order statistic，主要由一至两个上尾观测决定，不作为 production SLO。P1 只测 software cursor 与 headless control path，不测设备呈现、声学停播、在线 TTS 取消或真实服务并发。

## 5.6 C2 v3 exact-only 实现与验收

C2 v3 工件位于 `experiments/sci34_supplement/c2_crop_integrity/`。正式 accepted run 为 `c2crop_82103004_20260903T080512Z`，code commit `8210300`，result commit `7d50624`，manifest hash 前缀 `d8c3db4d`，seal hash 前缀 `e0997d41`。运行固定为 Qwen2-7B artifact、BF16、SDPA、Transformers backend 与严格离线环境。

v3 精确复用冻结的 24-case 网格。24/24 ordered cases 产生 27/27 ordered crop events，其中 3 个 no-op；冻结 fixture 的 308 个 assistant 内容 token 全部经 production `generate_accumulating()` 逐 token append，并检查每 token 一次 `_prefill_ids_p2` forward。三方比较实现如下：

1. crop 前对目标 keep length 的 K/V prefix 生成每层 shape、dtype、device 与 hash manifest；
2. production arm 唯一调用 `crop_to_token()`；
3. oracle arm 从 crop 前 clone 直接独立 slicing，不调用生产 crop；
4. validator 不信任 stored keep，而从 case、fragment token partition、role/content boundaries 和 second-crop fraction 独立推导；
5. 每层要求 pre-prefix、production post-crop 与 slicing oracle 的 K/V `torch.equal` 和 hash exact，并要求 `shape[-2]==keep`；
6. mask、完整 token ledger、seq length 与 KV length exact；27/27 wrong-length disposable negative controls 必须被拒绝。

crop 后两臂以相同 token-ID chunks 执行 60 个 recovery steps。每一步逐层比较 K/V，并比较 logits、mask、完整 ledger、retained-prefix hash，以及从操作序列独立推导的 `RolePhase`、`GenerationEndReason` 与 assistant content state。accepted run 中这些 gate 全部 bitwise/exact 通过；28 层 K/V 均在比较范围内。

v1/v2 是 clean re-prefill 协议，结论保持 rejected。v2 的 24/24 termination probe 和 45/45 token/state/EOT/scenario 检查通过，但单控制 2× numerical gate 只有 42/45。其 control 按语义 seam 分块并强制末 token单独 forward，production 则对初始 512/2048/8192 context 一次 forward，并按实际 API chunk 追加 role/content；两者 forward topology 不匹配。因此三项数值失败既不能识别 crop bug，也不能建立 clean-reprefill equivalence。v3 删除不可识别的 numerical re-prefill 比较，改用 direct pre-prefix/production crop/independent slice 的 exact 问题；它不改变 v1/v2 verdict，也不支持 clean-reprefill numerical equivalence、32-token continuation equivalence、跨模型/backend/硬件或在线生产正确性。

## 5.7 Artifact 与可复算范围

仓库根目录 `REPRODUCIBILITY.md` 是 artifact matrix，映射 campaign、accepted run、代码/结果身份、入口与分析文件。E3 的 exact processed input rescue 位于 `experiments/sci34_supplement/results/e3_exact_rescue/README.md`；它用于闭合 accepted E3 输入工件，不改变原始记录。C2 v3 accepted run、validation、analysis、manifest 与 seal 构成 D-023 接受的 direct crop-integrity 证据。确认性 E1/E2 的正式重分析位于 `experiments/sci34_supplement/results/e1e2_confirmatory/e1e2c_b8c758b_20260901T173306Z/analysis_v2.json`；E3 weighting/dedup 重分析位于 `experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/analysis_weighting_dedup_v2.json`。二者均为 versioned analysis，不得以旧 `analysis_v1` 替代或覆盖旧文件。

不同 campaign 不组成一个可池化的计时总体。确认性 E1/E2、固定轨迹 E3/联合 A1、prepared-state P1 与 C2 v3 各自保留模型、代码、结果和协议身份；Windows checkout 的 CRLF 转换可能使 byte-level C2 seal 在工作树误报，正式复核应使用 LF-preserving checkout、Git blob 或原 tarball。

这些工件支持论文所列的 analysis-only 复算、部分 smoke、accepted-run validation 与 hash/seal 检查，但不能据此声称整个系统已经完整可复现。旧探索性 E1/E2/A2 的部分环境与 exact input 仍不完整，真实 ASR/TTS/设备/声学闭环未实例化。仓库状态也不支持编造或宣称正式 LICENSE、公开/匿名 artifact URL、DOI、作者、基金、利益冲突、伦理或 consent 结论；相关 submission declarations 只在 `paper2/declarations.md` 中作为待作者确认草稿处理。

## 5.8 本章小结

本章给出 software-consumed cursor、片段、token 与 KV 的具体实现，并把 device-presented 与 acoustically heard 层排除在当前观测之外。时间轴 API 强制 token/sample 连续、fragment lifecycle、chunk 唯一归属与乱序拒绝。缓存容器显式维护完整 `token_ids` ledger、assistant 内容 ledger、`RolePhase` 与 `GenerationEndReason`；EOT 不进入内容 KV/ledger，`ASSISTANT_EOT_PENDING` 由 `reopen_user_role()` 唯一提交 close，推测全作废与播放期零内容保留采用不同 crop 语义，user 内容推进会清除陈旧 `CROPPED`。实验实现按 100 utterances×5 sessions 的 crossed `analysis_v2`、非 token-equivalent C-E1 路径、A1/P1 固定协议和 C2 v3 exact-only gate 报告。artifact 章节给出 `REPRODUCIBILITY.md`、E3 rescue 与 C2 accepted run 的可复算入口，但不作完整复现、许可证或公开 URL 主张。