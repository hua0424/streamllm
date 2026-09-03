# 实验设计文档（Code Experiment Plan）

## 文档状态

> 本文件是 2026-05-21 的历史实验预案，不再代表正式结果或最终统计口径。GPU 原始结果见 `experiments/results/exp*.json`；排除开发 fixture 后的离线完整性审计见 `experiments/results/paper2_reanalysis.json`；最终结论与限制以 `paper2/chapter6_experiments.md` 和决策 D-013～D-019 为准。二审后的 C2 EOS/EOT 与 crop/re-prefill 正确性协议见 `experiments/sci34_supplement/c2_equivalence/EXPERIMENT_PLAN.md`（现行为 v2：v1 formal 已 rejected 归档，D-019），GPU 唯一入口见同目录 `GPU_HANDOFF.md`。下文中“单调前沿、完全隐藏、完整 barge-in 亚毫秒”等均是当时的预期或旧口径，不得作为已验证结论引用。

- Origin Mode: plan
- Origin Date: 2026-05-21
- Historical Status: superseded for reporting by D-013 (2026-08-31)
- Version Label: code_plan_v1

---

## 0. plan 模式已定的四项基础决策（2026-05-21）

| # | 决策 | 选择 | 影响 |
|---|---|---|---|
| P1 | 打断产生方式 | **确定性程序注入** | "用户听到的"=注入时刻前已播放音频，ground truth 完全确定、可复现、无需真人/伦理审查 |
| P2 | 打断时机策略 | **混合：固定播放比例 25%/50%/75%（含 mid-fragment）+ 片段边界对照** | 同时覆盖"触发重写(贡献3)"与"干净截断"两类 |
| P3 | 一致性指标 | **客观"未听到内容引用率"为主 + LLM-judge 连贯性为辅 + 人工小样本验证** | 主指标客观自动化、直接对应机制目的 |
| P4 | 语种/数据集 | **英文为主（MultiWOZ 派生 + 自构造英文打断集）；中文（CrossWOZ）为可砍扩展** | 与 HumDial-FDBench 对齐；stream2sentence 用 nltk |

---

## 1. Experiment Overview

- **Title**：播放感知级联语音对话系统的延迟、效率与一致性评测
- **Objective**：验证（i）流式+播放感知系统相对非流式级联的延迟改善；（ii）软触发激进度对"推测浪费率 vs TTFT"的可调权衡；（iii）按播放位置截断相对按生成位置截断在多轮一致性上的改善；并通过消融量化各组件贡献。
- **Type**：simulation（自动化流水线仿真，非模型训练）
- **总体假设**：H1 系统 B 的 mouth-to-ear 延迟显著低于系统 A 且不随语音长度线性增长；H2 存在一条单调的浪费率-TTFT 权衡前沿；H3 播放感知截断的"未听到内容引用率"显著低于生成位置截断。

---

## 2. 系统配置与被测条件（Systems / Conditions）

| 代号 | 系统 | 说明 | 用于 |
|---|---|---|---|
| **A** | 非流式级联 baseline | 完整录音→整段ASR→整段LLM→整段TTS→播放 | E1 |
| **B-ours** | 流式+播放感知（完整方法） | 软触发推测+流式TTS+**按播放位置**截断 | E1/E2/E3/所有A* |
| **B-gen** | 流式+**按生成位置**截断 | 打断时截断到LLM已生成位置（忽略播放进度） | E3（核心对照） |
| **B-syn** | 流式+**按合成位置**截断 | 截断到TTS已合成位置（忽略播放器buffer） | E3（可选第三点） |
| **B-noKV** | 流式但**重新 prefill**（不复用KV） | 每次截断后重新编码上下文 | A1 |
| **B-naive/mark/rewrite** | 三种历史处理策略 | 朴素截断 / 标记法 / 重写法（贡献3） | A2 |

> B-gen 是 E3 的关键对照：它模拟"只看生成、不看播放"的朴素做法，B-ours 与它的差值即本文机制的价值。**注意（D-006）**：B-gen 不等同于商用系统（OpenAI/Azure 也做播放感知），它是"未做播放感知"的学术对照，论文中须如此表述，不得暗示商用系统=B-gen。

---

## 3. 指标定义（Metrics，对应大纲第三章）

| 指标 | 定义 | 单位 |
|---|---|---|
| **TTFT_text** | 软触发提交 → LLM 首 token | ms |
| **mouth-to-ear** | 用户语音结束 → 用户听到首个响应音频 chunk | ms |
| **barge-in 响应延迟** | 打断注入时刻 → TTS 播放停止 | ms |
| **推测浪费率** | 被作废的推测 token 数 / 总生成 token 数 | % |
| **KV 复用率** | 截断后复用的 KV token 数 / (复用 + 重算) | % |
| **未听到内容引用率**（主一致性指标） | 下一轮回复中引用了"生成但未播放内容"的样本数 / 被打断轮次总数 | % |
| **LLM-judge 连贯性**（辅） | judge 对下一轮回复连贯性的成对偏好/评分 | 偏好率 / 1-5 |

---

## 4. 数据与场景（Data & Scenarios）

- **对话源**：MultiWOZ 派生多轮英文对话（复用一期 CosyVoice2/TTS 合成管线生成音频）。中文 CrossWOZ 作可砍扩展。
- **多轮结构**：每个对话 ≥3 轮，其中指定轮次触发打断，测下一轮回复是否受"差异化截断的历史"影响（E3 必需多轮）。
- **打断场景**（P1+P2）：对每个被打断轮次，在播放比例 {25%,50%,75%} 与 {片段边界} 分别注入，形成受控条件。
- **场景类型**（大纲§6.2）：流畅完整句 / 带思考停顿 / **频繁打断（重点）** / 混合。
- **规模建议**（一个月 deadline 下的最小统计可用集）：主实验每条件 ~50-100 段对话；人工验证子样本 ~50 条。最终数值以实验机 7B 跑；验证机 0.5B 先跑通与 smoke。

---

## 5. 各实验规格（Per-Experiment Spec）

### E1 端到端延迟【必做】
- **RQ**：B-ours 相对 A，延迟改善多少、是否摆脱"随语音长度线性增长"。
- **IV**：系统类型(A/B-ours)、输入语音长度分组。**DV**：TTFT_text、mouth-to-ear、barge-in 响应延迟。
- **成功判据**：B-ours 的 mouth-to-ear 显著低于 A（配对检验 p<.05 + 效应量），且对语音长度斜率≈0。

### E2 推测浪费率 vs TTFT trade-off 曲线【必做，核心图】
- **RQ**：调节软触发两阈值，浪费率与 TTFT 如何权衡。
- **IV**：软触发（推测/提交）阈值扫描。**DV**：推测浪费率、TTFT_text/mouth-to-ear。
- **产出**：浪费率-TTFT 前沿曲线（论文核心图）。**成功判据**：曲线单调、可识别拐点。

### E3 播放感知 vs 生成位置截断的一致性【必做，价值核心】
- **RQ**：按播放位置截断相对按生成位置截断，多轮一致性改善多少。
- **IV**：截断依据(B-ours / B-gen[/B-syn])、打断播放比例。**DV**：未听到内容引用率(主)、LLM-judge 连贯性(辅)。
- **成功判据**：B-ours 未听到内容引用率显著低于 B-gen（p<.05 + 效应量）；LLM-judge 同向；人工小样本与 LLM-judge 一致性达标（如 Cohen's κ>0.6）。

### A1 KV 复用 vs 重新 prefill【必做消融】
- **IV**：B-ours vs B-noKV。**DV**：截断→可继续生成的延迟、KV 复用率。**判据**：B-ours 延迟显著更低。

### A2 三种历史处理策略【必做消融】
- **IV**：朴素/标记/重写。**DV**：未听到内容引用率、LLM-judge 连贯性、（重写的）额外延迟是否被隐藏。**判据**：标记/重写相对朴素改善一致性；重写延迟在用户说话期内可隐藏。

### A3 软触发激进度扫描【必做消融】
- 与 E2 共享数据，单独报不同激进度下浪费率/延迟/误触发率。

### E4 buffer 精确映射 vs 实时速度假设【锦上添花，可砍，D-006】
- **RQ**：buffer 精确映射相对"假设实时播放速度"，未听到内容引用率差异。
- **IV**：位置估计方式(测量 buffer / 实时速度假设)。**DV**：未听到内容引用率。
- **前置条件**：需构造"合成速度≠实时"场景（TTS 快于播放/buffer 堆积）。**仅在主 pipeline 跑通且时间有余时做。**

---

## 6. Instrumentation 清单（★ 编码验收标准：代码必须打这些点）

> 这是"实验设计先于编码"的落地物。`src/dialogue/` 各模块实现时必须发出以下埋点，否则实验无法测量、需返工。

**时间戳（单调时钟，相对 start）**：
- [ ] `user_speech_end`（音频输入结束）
- [ ] `trigger_fire`（软触发提交时刻）
- [ ] `first_llm_token`（首 token）
- [ ] `first_tts_chunk`（TTS 首块合成）
- [ ] `first_audio_played`（播放器播出首块）
- [ ] `barge_in_injected`（打断注入时刻，实验注入器写入）
- [ ] `tts_stop`（TTS 播放停止）
- [ ] `kv_crop_done`（KV 截断完成）

**计数器**：
- [ ] `generated_tokens_total`、`discarded_tokens`（每次推测作废累加）→ 推测浪费率
- [ ] `kv_reused_len`、`kv_recomputed_len`（每次截断事件）→ KV 复用率

**反向映射表落盘**（每轮对话记录，供离线核对与"未听到内容引用率"判定）：
- [ ] 每个 LLM token 的 `token_idx` ↔ 其归属 `fragment_id`
- [ ] 每个 stream2sentence 片段 `fragment_id` ↔ 覆盖的 `[token_start, token_end)`
- [ ] 每个 TTS 音频 chunk `chunk_id` ↔ 源 `fragment_id`
- [ ] 播放进度采样 `playback_ms` ↔ 当前 `chunk_id`
- [ ] 打断时：注入 `playback_ms` → 反查得到的 `token_idx`（截断点）与"已播放文本 / 未播放文本"切分

**实验注入器（experiment harness，非系统本体）**：
- [ ] barge-in 注入器：按 {0.25,0.5,0.75}×轮回复播放时长 与 {片段边界} 注入
- [ ] "未听到内容引用率"判定器：输入(已播放文本, 未播放文本, 下一轮回复)，检测下一轮是否引用未播放内容（规则+LLM-judge）
- [ ] 增量保存/断点续传（沿用一期实验脚本约定）

---

## 7. 分析计划（Analysis Plan）

- **主指标**：E1 mouth-to-ear；E2 浪费率-TTFT 曲线；E3 未听到内容引用率。
- **统计**：配对样本（同对话跨系统）用配对检验 + 效应量；多条件用方差分析 + 多重比较校正；报告置信区间而非仅 p 值。
- **对照**：E1 vs 系统 A；E3 vs B-gen；消融 vs B-ours 完整方法。
- **LLM-judge 可靠性**：人工小样本 vs LLM-judge 一致性（Cohen's κ）。

---

## 8. 环境与可行性

- **验证机** 5070 Ti 16GB：0.5B 主 LLM 跑通 pipeline + smoke，验证埋点正确。
- **实验机** 3090×2 48GB：7B 主 LLM 出正式数值（分卡布局见 `docs/paper2_context.md` §3.6）。
- **依赖**：一期栈 + stream2sentence + CosyVoice2 + TEN Turn Detection + Qwen3-0.6B（需 `uv sync` 重建环境，当前 venv 已损坏）。

---

## 9'. Harness 实现状态（2026-07-02，验证机 0.5B 全部跑通自检 PASS）

| 实验 | harness | 本机概念数值 | 实验机待办 |
|---|---|---|---|
| E1 延迟 | `run_exp1_latency.py` | A TTFT 24.8ms vs B 0ms；建模 m2e 2289 vs 45ms | 7B + real CosyVoice2 实测 mouth-to-ear、SYNTH_RTF 实测替换 |
| E2 trade-off（核心图） | `run_exp2_tradeoff.py` | 曲线：th0.02→waste30.4%/TTFT0.5ms … th≥0.12→0%/43-75ms，拐点 0.05-0.08 | 7B + TEN 7B（阈值区间按 TEN 分布重标）+ 真实 MultiWOZ |
| E3 一致性（核心） | `run_exp3_consistency.py` | **loose**（片段级）：B-ours 0%（**构造性保证**，非实验发现）vs B-gen 55.6%；**strict**（P1 严格 GT，含被打断片段未播尾部）：双列报告，playback 的 strict>0 = 片段粒度量化误差 | 真实 MultiWOZ + LLM-judge 交叉验证 + 7B |
| A1 KV 复用 | `run_exp_a1_kvreuse.py` | crop 0.12-0.19ms 近常数；re-prefill 14→63ms 线性；4k 处 3.6x | 7B 重跑（差距更陡）|
| A2 历史策略 | `run_exp_a2_history.py` | 三策略跑通；重写 mean~660ms 可隐藏 | LLM-judge 连贯性评分（judge 字段已预留）|
| A3 激进度扫描 | 与 E2 共享 records（逐阈值分解，无需独立脚本） | 同 E2 | 同 E2 |

**共用组件已验证**：软触发（开发替身 AUC~0.80；TEN 7B 实验机换入，D-011）、推测-作废状态机、
截断模式开关（B-ours/B-gen；**B-syn 在 Mock 同步合成下与 generation 等价，仅接入异步 real TTS
后才可区分，不得称"已验证"**）、Mock TTS TimingProfile（实验机 benchmark 替换）、
规则版未听引用检测器（LLM-judge 交叉验证留实验机）。
**§6 埋点补齐（2026-07-02 审查后）**：TurnMetrics.timestamps 落盘 8 个 §6 时间戳（模拟量已
标注）、`ttft_text_ms`（§3 定义的 trigger_fire→首token）、`kv_reused/recomputed_len` 与
`kv_reuse_rate`（rewrite 策略下 <1）、`TurnResult.timeline_records` 反向映射落盘、
E3 增加 P2 的"boundary"片段边界对照注入与 strict GT 双列。
**barge-in 响应延迟**关键路径=反查+crop（亚 ms、与上下文无关）；role 重建不在关键路径（可延迟）。
**ASR 真实音频链路**：一期已证 TTFT 与语音长度关系；二期 harness 用确定性文本段驱动（P1），
真实音频→流式 ASR 接入在实验机（有数据）时进行，属可选增强而非必需。

## 9. 待确认 / 开放项

1. 数据规模（每条件 50 vs 100 段）最终数值——建议先跑 50 看方差再决定是否加。
2. 是否纳入 B-syn（合成位置截断）作为 E3 第三对照——增强论证但加工作量，可作可砍项。
3. LLM-judge 用哪个模型（建议与主 LLM 不同家族以避免偏袒，如用一个更强的裁判模型）。
4. 中文 CrossWOZ 扩展是否做——归入可砍。
