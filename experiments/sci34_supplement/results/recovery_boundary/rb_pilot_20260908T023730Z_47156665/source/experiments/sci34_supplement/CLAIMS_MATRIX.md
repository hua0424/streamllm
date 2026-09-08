# 证据—主张矩阵

## S-E3 完成后

### 可以声称

- playback 与 generation 条件共享同一条被打断 assistant 生成轨迹。
- 在固定轨迹和 greedy probe 下，两种历史策略对共同差异文本的后续复现率存在某一实测差异，并给出配对 CI。
- E3 的片段目标与 proxy 目标按各自非空目标独立定义分析总体：片段口径仅纳入 `unheard_text.strip()` 非空的 pair，proxy 口径仅纳入 `strict_unheard_text.strip()` 非空的 pair；两者的 `n` 不要求相同。
- playback 片段口径不把完整未播放片段写入历史，这是构造性性质。

### 不可以声称

- 结果代表自然在线用户打断的总体发生率。
- LLM judge 等价于人工真值。
- 字符比例—空白边界 proxy 是物理词/token 播放真值。
- 片段与 proxy 指标共享同一个 eligibility 分母；旧 `eligible_pairs` 字段只能作为片段 eligibility 的兼容别名解读。
- Mock TTS 实验验证了真实播放器或在线 TTS 停止。

## S-A1 完成后

### 可以声称

- 在指定模型、硬件、上下文长度和同步墙钟协议下，joint crop+role 的 median/IQR。
- re-prefill median 与 joint median 的比值。
- 原始 repeats 已归档并可离线复算。

### 不可以声称

- joint A1 是完整 barge-in latency。
- 包含播放器停播、timeline lookup、服务通信或 GPU 并发负载。
- 结果无条件适用于其他模型和推理引擎。

## S-P1 prepared-state v2 已接受

正式证据为 run `sci34_dc52978_20260901_async_prepared_v2`（代码 `dc52978`，结果 `ee1dcc7`）。旧 P1 v1 的 stop→crop/role 联合计时被异步准备态工作污染，只可作为协议审计。

### 可以声称

- 9 个单元、180 条正式事件全部精确命中软件采样目标，停播确认后零软件采样泄漏；其中 120 条为片段内路径、60 条为片段边界路径。
- headless wall-clock-paced software playback 的 stop acknowledgment、post-stop device sync 和 timeline lookup 分布。
- 在该控制路径中，stop→crop 和 stop→role 的软件/模型侧累计延迟；九单元中位数范围分别为 2.44–2.53 ms 和 78.6–80.8 ms，最大单元 P95 分别约为 3.492 ms 和 86.1 ms。
- 播放前 setup 已单独计量并从 stop 路径排除；GPU 服务器无需声卡即可复现实验。

### 不可以声称

- 真实声卡或扬声器何时停止发声。
- 用户实际听到的最后一个 sample。
- 在线 CosyVoice 推理已被取消。
- ASR、LLM、TTS、播放器真实并发竞争已验证。
- 生产级完整端到端 barge-in latency。
- 九个单元范围较窄即可证明上下文长度无关、跨硬件不变或可与 A1 跨 campaign 相减得到固定系统开销。
- 把累计区间与组件中位数相加。

## 不做人工评测的影响

- 不影响片段级历史边界、KV 操作和模型侧性能结论。
- LLM judge 结果只能称为模型代理，不称为人类感知真值。
- 不使用“用户不可感知”“自然度提升”“人工验证风险接近零”等表述。
