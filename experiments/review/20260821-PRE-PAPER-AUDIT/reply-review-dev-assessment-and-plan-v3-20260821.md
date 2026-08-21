# 开发侧回复函 r3（对应 review-dev-assessment-and-plan-v3-20260821.md）

- 日期：2026-08-21
- 方案版本：`dev-assessment-and-plan-20260821.md` v3.1（Gate 0 口径冲突已消除，Gate 1 实现细则已冻结）
- 本轮无不采纳项。

## 一、四条代码断言的核实结果（全部属实）

| 再审断言 | 核实证据 | 结论 |
|---|---|---|
| §4.1 Silero 未固定 revision | `src/asr/streamaudio_segmenter.py:124-126`：`torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', ...)` 无 revision 参数，依赖浮动远端/不明缓存 | ✅ |
| §4.3 生产 System B 生成必在最后 ASR 完成后启动 | `run_exp_latency.py:755-771`：`llm_worker` 仅在收到 `is_end=True`（is_final 段转写完成）后进入 `generate()`；v3 中"first_token 可早于最后 ASR commit"系照抄 r2 §3.5 的表述，对本系统不成立，**接受 r3 纠正**，链条冻结为 `explicit_flush_done ≤ pipeline_input_close ≤ asr_processing_done ≤ first_token` | ✅ |
| §4.4 flush() 返回 None 时 is_final 段永不入队、transcriber 死等 | `run_exp_latency.py:675-678`：flush 结果仅在有音频时入队；`:712-720`：transcriber 退出依赖 `final_received`（仅由 is_final 段触发）——flush=None 时永久阻塞、`transcriber_thread.join()` 不返回。**该机制正是 505→498 排除中"流式 TTFT>10s 挂起"样本（4 条）的来源**，W9 说明将引用此因果链 | ✅ |
| §4.6 现有句末规则在 token 流上无法判小数点 | `measure_decode_to_first_sentence.py:66-76`：`detect_first_sentence_end` 用 `text[i+1]` 双侧上下文，离线全文本可用；流式场景 "." token 到达时下一 token 不存在，必须 lookahead | ✅ |

## 二、Gate 0 处置（§3.1–3.4，已在 v3.1 落实）

1. 方案顶部新增**优先级声明**（v3/v3.1 修订节覆盖全部冲突旧表述），并逐点修订了五处 v2/v3 旧段：16kHz mono 假设、已弃用字段、playable 降级口径、W5"或"表述、W6 的 repetition_penalty 写法；实现 schema 只保留 v3.1 字段；
2. 采样参数双字段：`requested_repetition_penalty=1.1` / `effective_repetition_penalty=not_applied`；论文禁写"1.1 实际生效"（已入 W8 残留搜索清单）；
3. **W5 采用再审建议的唯一冻结族**，不再保留"或"：Table III 总体 A/B 主比较 + Long/VL/EL 三比较 Holm 族；Table VII B vs LA 主比较、A vs B 验证性比较分开标注；R2 十二条件一个 Holm 族；R5 只报 B−A 配对均值差 bootstrap 95% CI；
4. W8 同步清单增补 `experiments/EXPERIMENT_DESIGN.md` 与 `experiments/CISR_REVISION_PLAN.md`（口径状态说明）。

## 三、Gate 1 处置（§4.1–4.10 全量接纳，已冻结为 v3.1 增补实现细则）

要点（全条见方案 v3.1 增补节）：

- PSE 算法定稿：排他右边界 [0,N]、双波形 SHA-256、底噪估计与 fallback、`db_to_amplitude=10**(db/20)`、≤200ms 取 energy；**Silero 固定 commit**（取 GPU 主机现有缓存对应 commit 并落盘 model hash），禁浮动远端；
- 因果回放：`actual_release_ns ≥ planned_release_ns`，提前发布记 error，逐 chunk 保存 scheduler error；论文说明为 500ms chunked real-time replay；
- 新增 `asr_processing_done_ns`；首句冻结后独立 TTS worker、LLM 生成不中断；
- 无条件 `INPUT_CLOSED` sentinel（flush=None 路径转为正常完成或显式 error，不再挂起）；self-test 必覆盖该路径；
- 事件定义五条（feed_end/flush_start/flush_done/input_close/full_input_ready）+ A 侧断言链；
- 流式句末检测：累计 token IDs 重解码、`.` 一字符 lookahead、EOS/max_tokens 裁决 pending、七类 self-test 用例（含 `Mr`+`.`+` Smith` 缩写限制声明）；
- PCM：512-byte 粒度、三级 timeout、探活先行并固定 Content-Type 策略、1324-byte playable、RMS/peak 基于 playable buffer、格式错误整行 error、主表不混口径；
- 生成元信息 + RNG 隔离：配对键 canonical JSON+SHA-256 派生 seed、A/B 同基础 seed、独立 `torch.Generator` 或设备 RNG 重置、不用内置 `hash()`；`first_model_token_ns`（兼容历史 TTFT）与 `first_content_token_ns` 分列；
- schedule：5 条 AB/BA/AB + 5 条 BA/AB/BA，语言×时长平衡，预生存 hash；
- fail-closed：终态 `success|error|cancelled|timeout`、原子写 + fsync、error key 不静默重跑、pair 总 deadline、TTS 慢流不无限续命、预期键恰一条终态。

## 四、下一步（按 §6 门禁执行，不再提交宏观方案）

1. 实现 W1（代码 + JSON schema + RUNINFO + self-test，覆盖 §6.2 全部本机用例）；并行实现 W3/W4/W5 本机脚本并执行；
2. 提交 W1 代码级审查（对照 §6.1 清单），通过后出 GPU handoff（TTS 探活 → 3 条冒烟[含故障注入] → 正式 50×2 + 子集补 1/2 轮 + 匹配文本控制 + W2 环境记录）；
3. 正式结果过 §6.4 结果级 QA 后，执行 W8 全链路同步，总册方可重新标"定稿"。

§7 的"可写/不得写"边界清单已并入 W8 核对项，改稿阶段照单执行。

## 五、实现证据（2026-08-21 补充，代码随函送审）

按 §8"下一轮审查对象应是实际代码、schema、self-test"的要求，实现已完成，提请代码级审查（Gate 1）：

### 5.1 交付清单

| 文件 | 内容 |
|---|---|
| `experiments/scripts/run_ttfa_unified.py`（新增，~1600 行） | W1 统一时间轴 TTFA 实测：PSE 双法裁决、因果回放、A/B 管线、流式句末检测、TTS 客户端、AB/BA 调度、fail-closed checkpoint、schema/偏序/闭合校验、QA 装配 |
| `src/llm/stream_llm_inference.py`（新增方法） | `generate_with_meta()`：暴露 token_id/decoded_text/is_eos/token_index/stop_reason；旧 `generate()` 一行未动 |
| `experiments/scripts/recompute_cv_stats.py`（新增） | W3 CV 重算（ddof=1 全分布） |
| `experiments/scripts/score_wer_offline.py`（扩展） | W4 corpus WER/CER（逐样本 S/D/I/N，DP 回溯） |
| `experiments/scripts/paired_inference.py`（新增） | W5 配对统计（bootstrap/Wilcoxon/效应量/Holm） |
| 文档 | `r5_semantic/REPRO_METADATA.md`（W6）、`r3_baseline_la/LA_METHOD_AND_EXCLUSION.md`（W9）、`r2_real_speech/MANUAL_SPOT_CHECK.md`（W7，试听待人完成） |

### 5.2 self-test 结果

`uv run python -m experiments.scripts.run_ttfa_unified --self-test` → **33 项全 PASS**（exit 0），覆盖再审 §6.2 全部本机用例：

- 成功路径：zh/en × A/B，schema 有效、事件边非负、TTFA 非负、**原始 ns 闭合残差=0**、PCM 达 1324B playable、A 未提前启动 ASR（`asr_start ≥ full_input_ready ≥ feed_end` 断言）；
- 故障注入：**flush()→None 经 INPUT_CLOSED 正常完成不挂起**、小数跨 token（`3`+`.`+`5` 不判句末）、缩写（`Mr`+`.`+` Smith` 判句末，限制已声明）、EOS-only 零内容记 error、WAV magic 格式 error、TTS 慢流超时 error、ASR 异常 fail-closed、checkpoint 损坏/hash 不匹配退出；
- 协议件：AB/BA 25/25 平衡、任务键唯一、子集三轮、schedule hash 稳定、配对 seed 确定性、PSE 裁决三分支（≤200ms 取 energy / 冲突 silero_fallback / 单算法失败 fail-closed）、chunk 因果释放（actual≥planned）、PSE 时钟映射公式。

W3/W4/W5 脚本各自 self-test 亦全过（W3 锚点与再审 §4.1 预期逐位一致：B 5.19/4.05/18.96、A 5.23/4.65/14.01）。

### 5.3 本机已产出的数据（无需 GPU）

- `r1_stats/repeat_cv_summary.csv/md`（W3，ddof=1 全分布 + 逐样本明细）；
- `r2_real_speech/wer_real.csv`、`r3_baseline_la/wer_la_vs_b.csv` 重生成：**宏平均列与定稿版 0 行差异**，新增 corpus 口径与逐样本 S/D/I/N 文件（`*_persample.csv`）；
- `stats_inference/paired_inference.csv/md`（W5，21 个比较；锚点：Table III 改善率 34.6%/65.6%/83.9%、table7 A/B 70.4%、B vs LA 差 541.1ms CI [485.3,599.9] p=1.2e-70、R5 B−A=−0.06 CI [−0.34,0.22] 跨 0）。

### 5.4 实现层须请审查注意的四点

1. **W4 参考文本来源**：本机无 R2 样本 JSON（在 GPU 主机），新增 `--ref-csv` 从 `qa_transcribe.corrected.csv` 补参考；**必须用 `reference_full` 列**——`reference` 列为 QA 展示截断版（90/150 行更短），误用会把 aishell1_clean 非流式 CER 从 0.1077 抬到 0.2009（已实测规避，重生成后宏平均与定稿版逐行一致）；
2. **句末检测的缩写限制**：`Mr.` 类缩写在流式 lookahead 下会判为句末（影响 TTS 文本切分点，不影响 TTFA 计时有效性），已在脚本 docstring 与方案声明；
3. **TTS 首包即断连**：拿到 playable 缓冲后客户端主动关闭响应（`tts_done_ns` 记录关闭时刻），不为测全程而保留连接；
4. **System B 生成起点未动**：保持生产语义（is_end 后才 generate），时间戳公式未反向改变系统行为。

### 5.5 当前状态与下一步

- W8 阶段 1 已执行：`PAPER_WRITING_REFERENCE.md` §十降级"整改中"，五处作废表述行内标记；changelog 已登记全部整改；
- 待 GPU 侧（Gate 1 通过后经 handoff 执行）：TTS 探活 → 3 条冒烟（含故障注入）→ 正式 50×2 + 子集补 repeat 1/2 + 匹配文本 TTS 控制 + W2 环境记录；
- 正式结果经结果级 QA 后执行 W8 阶段 2，总册方可重新标定稿。
