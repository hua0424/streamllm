# P1 prepared-state v2 验收补充报告（2026-09-01）

## 1. 验收结论

正式接受 run `sci34_dc52978_20260901_async_prepared_v2` 作为 headless、wall-clock-paced 软件播放控制路径证据。P1 v1 的联合计时继续判为协议无效，仅保留作失败审计；本报告取代 `supplement_acceptance_2026-09-01.md` 中“P1 v2 pending”的当前状态，不改写其历史记录。

本轮必要 GPU 补实验至此全部完成。现有证据不要求继续重跑 E1、E2、E3、A1、A2 或 P1；后续真实声卡、在线 TTS 取消和生产并发闭环属于扩展验证，而非当前论文定稿前置条件。

## 2. 身份与可追溯性

- 实验代码 commit：`dc529788e86ecd3e2e4203ba16b1076d6b231ec1`
- 结果入库 commit：`ee1dcc7`
- run ID：`sci34_dc52978_20260901_async_prepared_v2`
- manifest config hash：`93b7837acdc708ffde48448fc7cb0549475cbf064539d53a5327cda05031e005`
- `records.jsonl` SHA-256：`2dc68896dc52ce2c777b1a6375f1a5c3090f9baffd8f07a6ac1ed0f1769a3b67`
- `analysis.json` SHA-256：`b9705d58f36909604e3e0df94d2190b3a5050c6a62d35fee1c29987fff4db20a`
- `manifest.json` SHA-256：`0358af6cb3a7796d35091322a3075bab322679d65fe4a8a563b278159e7deef9`
- `run_summary.json` SHA-256：`197f2cb86827aa7e355db32cedbde339ff281cb244d50862c3c5991a8238ba4a`
- 回传 tarball SHA-256：`4c6188249f1226e5692a85468cf1e9c3b05e648494a5ce9a6e5a475b264c0bc8`

正式运行使用 clean tree，`runtime=transformers`，Qwen2-7B-Instruct 模型 identity hash 为 `7feb5a62bd0a65d0741eac46fc0ce2a0328aa5e8dec23fec92079346857347bc`。GPU 实验员使用 `git add -f` 将被 `.gitignore` 默认排除的正式 run、日志和快照纳入版本库；该操作不改变实验协议或文件内容，只增加不可变审计副本。

## 3. 数据完整性

- 3 个上下文长度（512、2048、8192）×3 个注入位置（0.25、0.50、0.75）×20 次正式重复，共 9 个单元、180 条唯一记录。
- 每个单元先执行 3 次 warmup；warmup 不写入正式 records。
- 180/180 记录满足 `protocol=async_prepared_v2`、`prepared_state_synchronized=true`、`trial_kind=formal`。
- 180/180 stop request 和 acknowledgement 精确命中目标软件采样位置，`leaked_samples=0`。
- 0.25/0.75 共 120 条 `mid_fragment`，全部 `partial=true`；0.50 共 60 条 `fragment_boundary`，全部 `partial=false`。
- 原始记录中的计时分解逐条满足嵌套恒等式；独立复算与入库 `analysis.json` 完全一致。

## 4. 正式延迟结果

| 区间 | 九个单元中位数范围（ms） | 最大单元 P95（ms） | 计时语义 |
|---|---:|---:|---|
| 软件停播确认 | 0.055–0.062 | 0.077 | stop 请求至播放器线程确认 |
| stop 后设备同步 | 0.167–0.176 | 0.352 | ack 后单独设备同步 |
| 时间轴反查 | 0.47–0.50 | 0.94 | 播放采样到片段/token 端点 |
| stop→crop 完成 | 2.44–2.53 | 3.492 | 累计至同步 KV crop 完成 |
| stop→角色恢复完成 | 78.6–80.8 | 86.1 | 累计至角色边界恢复完成 |

最大单元 P95 的精确值依次为 0.076842、0.351591、0.939422、3.491824 和 86.084611 ms。表中采用适合正文的四舍五入值，不将 0.35 或 3.49 写成严格上界。

准备态 `setup_ms` 的原始范围为 40.499–1722.228 ms，九单元中位数为 41.208–1717.110 ms。它包括 `ensure_full()` 及播放前设备同步，全部在播放器启动前完成，并从 stop 路径排除。上述累计区间相互嵌套，不能与组件中位数相加，也不能通过与另一 campaign 的 A1 数值相减解释系统开销。

## 5. 环境与 campaign 边界

P1 v2 主机环境为：双路 Intel Xeon Gold 6330（2×28 核、112 逻辑 CPU）、约 756 GiB 内存、Ubuntu 22.04.5、Linux 5.15.0-136、Python 3.10.18、PyTorch 2.8.0+cu128、Transformers 4.57.1、CUDA 12.8、cuDNN 9.1、NVIDIA driver 580.105.08、2×RTX 3090 24 GB。运行前后 GPU compute-process 快照均为空。

P1 v2 作为第三个独立 campaign 报告。相同 GPU 型号只能减少一种硬件差异，不能使不同 CPU/OS/调度环境的绝对墙钟值可池化。本文不直接比较或相减旧 E1/E2/A2、固定轨迹 E3/联合 A1 与 P1 v2 的绝对延迟。

## 6. 允许与禁止的主张

允许：在上述主机、模型和 prepared-state 协议下，报告 headless 软件停播确认、零软件采样泄漏、时间轴反查、stop→crop 和 stop→角色恢复的单元级分布。

禁止：把结果称为声卡或扬声器的实际停止时间、用户实际听到的最后采样、在线 CosyVoice 取消、真实 ASR/LLM/TTS/播放器并发，或生产级完整端到端 barge-in latency。九单元延迟范围较窄是本次 campaign 的观察，不证明上下文长度无关或硬件不变性。
