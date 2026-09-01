# E1/E2 确认性 campaign 证据—主张矩阵

本文约束 GPU 验收后允许进入论文的表述。campaign 未通过全部硬性门槛时，只能写“代码已实现，GPU 正式结果待运行/待验收”，不得引用 pilot 或部分 session 数字。

## 1. 证据身份

- 证据类型：确认性、受控、文本段驱动的模型侧 campaign；
- 输入：新的 100 条 MultiWOZ 2.1 派生 holdout，与旧 E1/E2 及 accepted 固定轨迹 E3 disjoint；
- accepted E3 排除源：`experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/manifest.json`；
- 模型：Qwen2-7B-Instruct，greedy，batch 1；
- 触发：真实 TEN confidence 预计算为只读 cache，formal session replay；
- 身份：不可变 `campaign_manifest.json` 冻结 input/cache/TEN/main-model/protocol identity，五个 formal session 强制共享其 SHA-256；
- 重复：5 个独立 Python 进程，`session-index=0..4`；
- 主候选：`0.92`，在新 holdout 可见前预冻结；
- 不确定性：session→dialogue 两层 bootstrap；
- 音频身份：非真实音频实验。

## 2. 时间语义护栏

必须区分：

- `last_segment_arrival`：最后一个受控文本段到达 harness；
- `first_token_ready`：首个最终可用 token 实际准备完成；
- `endpoint_accept`：同步 oracle 接受候选/路径。

实际受控墙钟主指标是：

```text
first_token_ready - last_segment_arrival
```

`endpoint_accept` **不是**最后一段到达瞬间。`TTFT_eff` 在候选已提前准备并于同步 oracle 接受时记为 0，因此是候选准备后同步 oracle 接受的**时延的乐观下界（推测收益的上界）**，不是实际受控墙钟主指标，也不是 mouth-to-ear 或用户感知延迟。

Raw records 已直接保存 `last_segment_arrival_ns`、`first_token_ready_ns` 和 `arrival_to_first_token_ready_ns`；validator 复算差值，analyzer 以 arrival-to-ready 为主指标。不得以 `endpoint_accept_ns` 替代 `last_segment_arrival_ns`，也不得把 `TTFT_eff` 升格为墙钟主结果。

## 3. C-E1 通过验收后

### 可以声称

- 在指定硬件、模型、greedy 解码和受控同步文本段协议下，System A 与 B@0.92 的实际 `last_segment_arrival→first_token_ready` 分布与配对差异，前提是 raw records 可直接审计这两个事件。
- B@0.92 是旧探索 campaign 预选、在新 holdout 上冻结验证的候选，不是根据确认性结果事后选择。
- 可以把 `TTFT_eff` 单列为同步 oracle 接受时延的乐观下界（推测收益的上界）：候选已准备、接受时存活且 ready>0 时为 0。
- consumer delivery latency 可单列为实现诊断。
- 可以报告 5 个独立 session 的配对效应及 session→dialogue 两层 bootstrap CI。
- 若未复现旧方向，应如实报告该预冻结候选未获得预期改善。

### 不可以声称

- `0.92` 是部署最优、全局最优或跨模型/语言最优阈值。
- `endpoint_accept` 就是最后一段到达，或 oracle 没有自身决策/同步语义。
- `TTFT_eff=0` 表示最后一段到达后零计算、零调度、零网络或用户零感知延迟。
- 跳过 raw `last_segment_arrival_ns` / `first_token_ready_ns`，用 `endpoint_accept_ns` 或 `ttft_eff_ns` 替换实际墙钟主指标。
- 新结果“确认”旧具体数字，除非新 analysis 自身给出该值。
- 结果包含真实 ASR、实际 endpoint detector、在线 TTS、播放器、声卡或声学传播。
- 结果是生产端到端语音系统总体延迟。

## 4. C-E2 通过验收后

### 可以声称

- 在冻结的八个数值阈值和 `never_speculate` 上，报告离散首 token 准备延迟—推测浪费工作点。
- 报告 B@0.92 对 `never_speculate` 的预先指定配对比较。
- 浪费率主定义为：
  `sum(wasted_tokens) / sum(wasted_tokens + final_tokens)`。
- 报告 utterance-level `wasted/(wasted+final)`、survival、ready tokens、invalidations、candidate lead、on-demand TTFT 和 `TTFT_eff` 时延的乐观下界（推测收益的上界）诊断。
- 说明各条件共享 holdout、TEN cache、模型配置并使用平衡顺序。
- 将曲线称为“本受控 campaign 的离散工作点”或“样本内 trade-off 形状”。

### 不可以声称

- 用 `wasted/speculative_tokens` 替代正式 pooled waste 主定义；该比值至多是诊断。
- 离散点构成连续、单调、完整或普适 Pareto frontier。
- 根据确认性结果把其他点重命名为 confirmatory 主阈值。
- TEN cache replay 测得在线 TEN runtime，或证明 trigger 端到端零开销。
- pooled token waste 等价于能源、成本、显存或生产吞吐改善。
- `never_speculate` 等价于所有传统语音系统。

## 5. Holdout 与独立重复

### 可以声称

- holdout 按冻结规则从本地 MultiWOZ 2.1 派生，并显式排除旧 E1/E2 与 accepted E3 manifest 中的 ID。
- 5 个 formal session 是 5 次独立 Python 进程/模型加载，session 是 bootstrap 第一层。
- campaign manifest、raw records、session manifests、环境快照和输入/cache/model/config hashes 可审计。
- pilot 使用独立非 formal campaign、`--limit 3` 且不传 formal manifest，不进入 formal analysis。

### 不可以声称

- 100 条话语代表开放域、多语言或生产流量总体。
- 5 个进程等价于 5 台主机、5 个模型族或跨硬件复验。
- pilot 是第六个 formal session。
- 把 `--limit 3` pilot 当作 formal 样本或与 5000 条 formal records 合并。

## 6. Oracle endpoint 与非真实音频边界

### 可以声称

- 同步 oracle 为所有条件提供一致的接受语义。
- segment 序列模拟增量 ASR final 文本流的输入形状。
- `TTFT_eff` 描述 oracle 接受后的策略可交付上界。

### 不可以声称

- oracle endpoint 是实际 endpoint detector 的准确率或延迟结果。
- oracle accept 与最后 segment arrival 是同一时刻或同一事件。
- segment 节奏来自真实录音、真实说话速度或真实流式 ASR。
- 本 campaign 使用、播放或评估真实音频。
- 结果验证真实停顿、噪声、ASR 修订、VAD 错误、在线 TTS cancellation、声卡 buffer 或 barge-in。
- 将本 campaign 称为“真实音频 E1/E2”“全链路确认”或“生产语音端到端验证”。

## 7. TEN cache 边界

### 可以声称

- TEN 对冻结 holdout 的累积文本 prefix 真实前向一次，归档未舍入 confidence、文本/template/model identity。
- replay 使九个 B 条件和五个 session 使用确定性一致 confidence。

### 不可以声称

- replay 包含在线服务调度、并发、排队或 TEN 推理墙钟开销。
- TEN 在自然音频 turn detection 上达到某准确率，除非有独立证据。
- trigger runtime 被“隐藏为零”。

## 8. 统计与报告边界

### 可以声称

- 主统计不做 outlier trimming，报告分布、配对效应和预设两层 bootstrap CI。
- pooled waste 按 `wasted/(wasted+final)` 复算。
- analyzer 当前提供的 `TTFT_eff` 汇总明确标为时延的乐观下界（推测收益的上界）诊断。
- 后续口径修订使用 versioned analysis，并记录 superseded 关系。

### 不可以声称

- CI 跨零证明无差异，或不跨零证明跨设置普适。
- 事后子组、删点、异常值剔除或阈值重选是主分析。
- 把同一 dialogue 的条件或同一 session 内 records 当作完全独立样本。
- 忽略 validator 的 raw 时间恒等式检查或把 analyzer 的 oracle latency lower-bound / speculation-benefit upper-bound 输出当作主指标。

## 9. 论文更新模板

只有在 campaign manifest、raw 时间恒等式、analysis、validation 与其他门槛全部验收通过后，才可使用：

> 在 Qwen2-7B-Instruct、greedy 解码和受控同步文本段协议下，5 个独立进程对新的 100 条 disjoint MultiWOZ 派生 holdout 进行了确认性评测。实际受控墙钟主指标定义为最后一段到达到最终首 token 准备完成；同步 oracle 接受后的 `TTFT_eff` 另作为时延的乐观下界（推测收益的上界）报告。预冻结的 B@0.92 与 System A / never-speculate 的差异为……。浪费率按 wasted/(wasted+final) 定义。输入不是实际音频，且不包含真实 ASR、在线 TEN runtime、TTS、播放器或声学链路。

禁止使用：

> 我们在真实语音系统中证明 0.92 最优，并在用户说完后实现端到端零延迟。
