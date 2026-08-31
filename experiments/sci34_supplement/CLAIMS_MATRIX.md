# 证据—主张矩阵

## S-E3 完成后

### 可以声称

- playback 与 generation 条件共享同一条被打断 assistant 生成轨迹。
- 在固定轨迹和 greedy probe 下，两种历史策略对共同差异文本的后续复现率存在某一实测差异，并给出配对 CI。
- playback 片段口径不把完整未播放片段写入历史，这是构造性性质。

### 不可以声称

- 结果代表自然在线用户打断的总体发生率。
- LLM judge 等价于人工真值。
- 字符比例—空白边界 proxy 是物理词/token 播放真值。
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

## S-P1 完成后

### 可以声称

- headless wall-clock-paced software playback 的 stop acknowledgment、游标泄漏和 timeline lookup 分布。
- 在该控制路径中，stop→crop 和 stop→role 的软件/模型侧联合延迟。
- GPU 服务器无需声卡即可复现实验。

### 不可以声称

- 真实声卡或扬声器何时停止发声。
- 用户实际听到的最后一个 sample。
- 在线 CosyVoice 推理已被取消。
- ASR、LLM、TTS、播放器真实并发竞争已验证。
- 生产级完整端到端 barge-in latency。

## 不做人工评测的影响

- 不影响片段级历史边界、KV 操作和模型侧性能结论。
- LLM judge 结果只能称为模型代理，不称为人类感知真值。
- 不使用“用户不可感知”“自然度提升”“人工验证风险接近零”等表述。
