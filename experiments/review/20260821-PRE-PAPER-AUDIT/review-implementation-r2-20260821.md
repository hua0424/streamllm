# Gate 1 整改复核报告 r2（2026-08-21）

- **固定点**：`7323359`
- **整改提交**：`1d81cf1`
- **回复函**：`experiments/review/20260821-PRE-PAPER-AUDIT/reply-review-implementation-r2-20260821.md`
- **前序报告**：`review-reply-implementation-v3.1-r2-20260821.md`

## 1. 总体裁决

本轮整改实质关闭了前序提出的主要问题：final-drain竞态、System A fatal传播、fatal checkpoint恢复、固定Silero注入、smoke样本完整性、TTS文本字段一致性，以及W3/W4/W5统计脚本问题均已有修复和测试证据。

因此：

| 阶段 | 裁决 |
|---|---|
| W3 CV | **放行** |
| W4 WER/CER | **放行** |
| W5 配对统计 | **放行** |
| 独立 TTS 探活 | **放行** |
| 3条 GPU 冒烟 | **有条件放行** |
| 正式50条 A/B 实验 | **暂不放行** |

3条冒烟必须按回复函承诺的顺序执行，并通过结果级 QA；不能跳过冒烟直接进入正式实验。

---

## 2. 已关闭的问题

### 2.1 Final-drain竞态

`run_ttfa_unified.py:698-724,731-769` 已在同一 `state_lock` 内完成：

1. 检查尾段；
2. 将尾段标记为 final；
3. 设置 drain 状态；
4. 最后发布 `pipeline_input_close_ns`。

新增测试强制线程交错，并使用真实 `ASRCache` 协议验证尾文本进入 LLM prompt。该项通过。

### 2.2 System A fatal传播

`run_ttfa_unified.py:879-895,924-1023` 已使 System A 的 ASR、LLM cache 和生成异常进入 `fatal=True`；主循环会停止后续任务并补写 cancelled。新增 ASR和LLM异常测试通过。该项通过。

### 2.3 Fatal checkpoint恢复

`run_ttfa_unified.py:1145-1183,2268-2279` 已恢复历史 `fatal_seen` 状态。重新打开 checkpoint 后会停止未完成任务并补写 `cancelled_after_fatal`。该项通过。

### 2.4 固定 Silero 注入正式流式分段器

`StreamAudioSegmenter` 支持注入固定的 model/utils，正式 runner 使用与 PSE 相同的固定实例，并有 `silero_injected` 断言。该项通过。

注意：`ttfa_local_integration.py` 仍是独立的本地路径检查脚本，不代表正式 runner；正式/冒烟路径的注入已正确。

### 2.5 Smoke样本完整性

`run_ttfa_unified.py:2245-2265` 已拒绝零命中、命中不足、缺少双语覆盖以及错误任务数量。QA要求成功路径和故障路径均实际执行。该项通过。

### 2.6 TTS派生字段一致性

`run_ttfa_unified.py:561-590` 已校验：

```text
tts_n_chars == len(tts_text)
tts_n_bytes_utf8 == len(tts_text.encode("utf-8"))
tts_text_sha256 == sha256(tts_text)
```

并验证 `tts_text_source` 与 mode/fallback 一致。负向测试通过。该项通过。

### 2.7 W3/W4/W5

- W3：逐文件主键唯一、三轮键集一致、配置一致性检查已补；数字与锚点一致。
- W4：共同 sample-ID 过滤、paired manifest、`reference_full` fail-closed、S/D/I/N 已补；当前数据集合一致，旧汇总数字不变。
- W5：重复键、空配对、LA mode、R5唯一性已补；21项结果不变。

三项均放行。

---

## 3. 3条GPU冒烟前的剩余条件

### 3.1 TTS total deadline仍不是严格的内部主动取消

位置：`run_ttfa_unified.py:328-417,864-921,1010-1014`。

当前已实现：

- response句柄可由外层访问；
- pair剩余时间会收紧 read timeout；
- pair超时后主动关闭 response；
- headers-only慢流测试可以返回 timeout；
- 未发现线程泄漏。

但 `tts_measure()` 内部 total deadline 仍只在 `iter_content()` 返回数据后检查。若服务器已返回 headers 后持续不发送 body，最终仍依赖 read timeout 或外层 pair deadline，而非独立精确的 TTS total deadline。

**判断**：对3条冒烟的隔离已基本足够，但执行时必须保存慢流测试证据；若慢流出现 `thread_leak`、无法关闭 response 或污染后续任务，立即停止，不得进入正式实验。

### 3.2 冒烟必须验证真实GPU路径而非只看本机self-test

本机 `run_ttfa_unified --self-test` 已报告69 PASS / 0 FAIL，但仍需注意：大部分 self-test 使用 fake 组件，不能替代 GPU上的真实：

- 固定 Silero PSE及正式 StreamAudioSegmenter；
- Whisper ASR final-drain；
- Qwen生成和TTS调用；
- 双语输入路径；
- 故障后的 cancelled 补写。

本地集成测试可作为路径证据，但不是正式结果证据。尤其真实集成脚本使用噪声输入、可能走 energy fallback，且使用 fake HTTP TTS，不能据此宣称生产协议已完整验证。

### 3.3 冒烟任务必须满足最低覆盖

GPU侧执行时应至少确认：

- 中英文均覆盖；
- A/B两模式均真正执行；
- 成功路径至少有一条；
- 故障注入路径确实产生 error 终态；
- 故障后后续任务被补写 cancelled；
- 任务数量与预期完全一致；
- 不出现 `final_drain_empty`、`thread_leak`、`pair_timeout` 或 schema 错误；
- 固定 Silero artifact hash 与 RUNINFO一致；
- TTS正式请求的 header/payload策略与探活一致。

---

## 4. TTS独立探活放行范围

可以先执行独立 TTS 探活，但必须保存：

- HTTP status；
- Content-Type；
- Content-Encoding；
- payload classifier结果；
- PCM服务采样率、位深、声道和配置；
- 探活时间、服务配置和运行环境。

探活通过只表示 TTS 基本协议可用，不表示：

- ASR路径正确；
- final-drain正确；
- 固定Silero用于正式分段器；
- 线程取消可靠；
- TTFA可用；
- 正式系统输出质量通过。

探活失败时不得临时放宽正式允许策略。

---

## 5. 正式50条实验的放行条件

正式实验仍需等待3条冒烟通过后单独放行。至少满足：

1. 3条冒烟全部通过结果级 QA；
2. 真实 GPU 环境固定 Silero artifact hash 落盘且与正式 StreamAudioSegmenter一致；
3. TTS探活和正式请求 header/payload策略一致；
4. 中英文 A/B 成功路径均通过；
5. 故障路径和 cancelled 补写正确；
6. 慢流/取消没有线程遗留；
7. 所有成功记录通过 schema、事件偏序、TTFA非负和 ns闭合校验；
8. 冒烟结果不存在异常 final-drain、timeout或schema错误。

---

## 6. 最终结论

**本轮整改可以通过代码级复核，独立TTS探活放行，3条GPU冒烟条件放行；正式50条实验暂不放行。**

开发侧现在可以在 GPU 主机上按 handoff 执行：

1. TTS独立探活；
2. 记录探活产物；
3. 运行69项self-test；
4. 运行3条分层 GPU 冒烟（含成功和故障路径）；
5. 提交冒烟 JSONL、RUNINFO、QA和错误/取消记录。

冒烟结果返回后再进行一次结果级复核。只有该复核通过，才可以启动正式50条 A/B、重复子集和匹配文本控制实验。
