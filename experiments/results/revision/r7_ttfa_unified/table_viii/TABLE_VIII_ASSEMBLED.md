# Table VIII 装配稿（R7 统一 TTFA，2026-08-22；分项标签修正版）

> **数据源（唯一合法）**：`r7_ttfa_unified/r7_main/`（repeat0，n=50/模式，zh/en 各 25）。
> 主指标 **first_playable_pcm**（speech_end → 首个 ≥1324B 可播 PCM）；单位 ms，1 位小数；
> std=ddof=1，分位数 np.percentile 线性插值。六分项首尾相接恒等闭合（QA-1）。
> 旧 `r6_ttfa/ttfa_budget.csv` 全部行作废，未参与本表。

> **分项标签边界（2026-08-22 复审修正）**：第二分项论文标签为 `t_feed_to_close_wait`（**喂入结束→管线输入关闭**，= pipeline_input_close − feed_end）；
> 源 summary 字段 `t_flush_to_close` 为历史命名。该 ~133ms 是完整等待，**不得归因为 flush 计算开销**——按复审独立复算（review-reply §2）：
> feed_end→explicit_flush_start ≈132.68ms、explicit flush 本身 ≈0.21ms、flush_done→input_close ≈0.12ms（合计 0.33ms 量级为 flush 段自身）。
> 本表保持六段闭合链、仅用准确标签（复审推荐的最简处理）。

## (a) TTFA 总量

| 系统 | 语种 | n | mean | std | P50 | P90 | P95 |
|---|---|---|---|---|---|---|---|
| B（流式） | zh | 25 | 3303.3 | 2681.3 | 2603.0 | 3130.4 | 8299.2 |
| B（流式） | en | 25 | 7660.5 | 2781.2 | 7577.0 | 10940.4 | 11857.2 |
| B（流式） | ALL | 50 | 5481.9 | 3486.1 | 3113.7 | 10506.6 | 11656.3 |
| A（非流式） | zh | 25 | 22616.8 | 2197.2 | 22161.4 | 25459.9 | 26874.8 |
| A（非流式） | en | 25 | 22234.7 | 2850.4 | 22439.6 | 25274.4 | 26487.5 |
| A（非流式） | ALL | 50 | 22425.7 | 2526.1 | 22269.9 | 25588.8 | 26887.4 |

- B vs A（ALL）：mean 降 **75.6%**（4.09×）、P50 降 **86.0%**（7.15×）——两种表述二选一，勿混用。
- B vs A（zh）：mean 降 **85.4%**（6.85×）、P50 降 **88.3%**（8.51×）——两种表述二选一，勿混用。
- B vs A（en）：mean 降 **65.5%**（2.90×）、P50 降 **66.2%**（2.96×）——两种表述二选一，勿混用。

## (b) 组件分解（mean±std，ms）

| 组件 | B zh | B en | B ALL | A zh | A en | A ALL |
|---|---|---|---|---|---|---|
| t_trailing_feed_wait（语音结束→喂入结束） | 0.1±0.0 | 0.1±0.0 | 0.1±0.0 | 0.1±0.0 | 0.1±0.0 | 0.1±0.0 |
| t_feed_to_close_wait（喂入结束→管线输入关闭） | 126.5±71.5 | 139.6±121.1 | 133.0±98.7 | 0.0±0.0 | 0.0±0.0 | 0.0±0.0 |
| t_close_to_first_token（输入关闭→首 token） | 1351.3±222.5 | 1414.8±327.3 | 1383.1±278.8 | 4530.9±860.1 | 3834.9±809.5 | 4182.9±898.3 |
| t_first_token_to_text_ready（首 token→文本就绪(首句/全文)） | 115.7±274.3 | 658.3±384.2 | 387.0±429.2 | 4774.4±330.3 | 4587.7±754.2 | 4681.1±583.9 |
| t_text_ready_to_tts_req（文本就绪→TTS 请求） | 0.4±0.1 | 0.5±0.1 | 0.4±0.1 | 0.6±0.1 | 0.6±0.1 | 0.6±0.1 |
| t_tts_to_playable（TTS 请求→首个可播 PCM） | 1709.3±2319.6 | 5447.3±2451.0 | 3578.3±3023.6 | 13310.8±2461.1 | 13811.4±2590.8 | 13561.1±2513.6 |
| **Σ组件（闭合校验）** | 3303.3 | 7660.5 | 5481.9 | 22616.8 | 22234.7 | 22425.7 |
| **TTFA（表 a 复核）** | 3303.3 | 7660.5 | 5481.9 | 22616.8 | 22234.7 | 22425.7 |

## (c) 稳定性与截断注记

- 子集三轮 CV（10 样本×2 模式，ddof=1）：mean 7.73% / max 20.70%（`ttfa_subset_cv_r7_main.csv`）。
- 生成截断（repeat0，n=100）：84 条 max_tokens 截断 / 16 条 eos（max_tokens=128 对 A/B 同等作用）。
- speaker '晓伊' 由本地服务映射为内置中文女声（非原论文音色）；Triton fallback×4 为平台固定条件——两者均入 RUNINFO 注记，论文按 review §6 声明。

## (d) TTS 控制结果的使用范围（不入表行）

- `r7_tts_control` 32 条：tts_request_start→first_pcm mean **7076ms**；仅用于 TTS 服务延迟归因与审稿回复证据，不作为 Table VIII 行项。
- **引用必带脚注**：`r7_tts_control` was launched after completion of `r7_main` but before the separately required written authorization and reviewer QA sign-off. The run was retained under an explicit procedural-deviation waiver because post-run audit found exact checkpoint/text/hash binding, 32/32 successful calls, and no code or platform divergence affecting measurement validity; this waiver is not retroactive authorization of the original execution.

## 装配 QA

- QA-1 六分项逐记录闭合：100 条最大残差 0.00e+00 ms（恒等式成立）
- QA-2 received→playable 缓冲差：mean 0.1 / p95 0.1 / max 0.2 ms（received 仅作 QA 补充）
- QA-3 与 ttfa_summary_r7_main.csv 双入口对拍：48 行全部一致
- QA-4 输入 checkpoint sha256(LF)=4edcd6ec28189d00…（不可变归档，control_from 同源）

