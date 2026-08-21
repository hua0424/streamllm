# 开发侧回复函（对应 review-dev-assessment-and-plan-20260821.md）

- 日期：2026-08-21
- 方案版本：`dev-assessment-and-plan-20260821.md` v2（已按本复审全量修订）

## 一、总体回应

复审意见**全部接纳，无争议项**。复审的事实性断言经本机独立核实：

1. **被评 A/B 回复的生成参数**：`src/llm/stream_llm_inference.py:195` `generate()` 默认 `temperature=0.1, top_p=0.9, repetition_penalty=1.1`，`run_exp_latency.py:766` 调用未覆盖默认值——复审 §5.3 的 temperature=0.1 断言**属实**（v1 方案误写 temperature=0，v2 已更正为两组参数分列）；
2. judge 为 `temperature=0.0`（`semantic_consistency.py:124`），属实；
3. TTS 客户端（`measure_tts_first_chunk.py`）确为 CosyVoice 式 HTTP 流式裸 PCM（22050Hz/16bit/mono），复审 §3.6 要求的格式校验、字节对齐、最小可播放缓冲判定均可复用该客户端实现。

对 v1 的两处表述问题自我更正：

- v1"同一请求、同一时间轴"系指"每请求单一时间轴"，表述不当；v2 统一为复审 §3.1 的"同一批样本、相同输入音频和配置下，分别执行的配对 A/B 请求"；
- v1 把 `feed_end − physical_speech_end` 命名端点分项并设"必然≤500ms"验收，确属把指标改名后宣称非负；v2 按 §3.2/§3.3 改为三量分列、如实命名 `explicit_flush_time`、取消该验收条件。

## 二、对复审 §8 十二项的逐项确认

1. **endpoint 三量区分**：接纳。v2 §2.2：`source_trailing_silence_ms` / `trailing_feed_wait_ms` / `explicit_flush_time` 分别记录分别报告；本运行无在线 VAD endpoint decision，不声称测量 VAD endpoint latency；E5 负值字段为异时间轴量，不入新表。
2. **physical speech end 固定参数**：接纳。v2 §2.3：16kHz、25ms 窗/10ms hop、能量门限与 Silero 参数全固定、冲突裁决规则（差 >200ms 取 Silero 并标记 `pse_method`）、输出 sample index + WAV SHA-256 + 映射公式、全程 `perf_counter_ns`。
3. **原始事件字段/分项公式/单调性断言**：接纳。v2 §2.4 全字段清单（含 `generation_end_ns`、`tts_request_start_ns` 独立时间戳、`generation_stop_reason`）；§2.5 组件由同一组连续 `*_ns` 差分构成，闭合残差为舍入级（<1ms），异步间隔单列为命名调度项；同记录时间戳按因果序非递减断言。
4. **first received / first playable PCM**：接纳。v2 §2.5：裸 PCM 格式校验（非 WAV 头/错误 JSON）、2 字节对齐、≥30ms 可播放缓冲判定，首块字节数/时长/RMS/peak 落盘；主表用 `TTFA_playable`，降级时称 speech-end-to-first-received-PCM，不称 first audible；未接声卡的限制声明入论文。
5. **配对独立请求与 AB/BA 平衡**：接纳。v2 §2.1：25 条 A→B + 25 条 B→A，语言×时长分层，确定性 seed，顺序表+seed+hash 落盘；唯一键 `sample_id × mode × repeat_idx`，配对 WAV SHA-256 一致为验收项。
6. **System A 直接 TTS 测量**：接纳。A 在 capped full response（注明 max_tokens=128 上限）结束后调 TTS，`generation_end_ns`/`tts_request_start_ns` 分列；0.09 s/字符估计彻底废弃。
7. **重复性子集**：**执行**。10 条分层子集 × A/B × 3 轮（约 +1–1.5h GPU），报告子集 median/mean CV 与顺序效应；论文表述为"主实验单轮配对 + 子集三轮重复"，不外推 E1 结论。
8. **CV 公式与 std 口径**：接纳。v2 §四：`ddof=1` 公式、P90 线性插值注明、逐样本明细、输入 SHA-256、**所有论文表格 std 定义统一注明**。
9. **WER/CER S/D/I/N**：接纳。v2 §四：WER 与 CER 八列分列、corpus 恒等式自检、paired filter manifest、归一化实现复用、6.72% 旧值清除、10.77%/11.80% 标注宏平均口径。
10. **统计协议冻结**：接纳。v2 §五：paired bootstrap 10,000 次固定 seed、percentile 95% CI、改善率 `(mean(A)−mean(B))/mean(A)`、Wilcoxon 双侧 `zero_method='wilcox'` `method='auto'`、R2 族内 Holm 校正、rank-biserial + paired dz、三条文字规则照录。
11. **LocalAgreement 工作项**：接纳，新增 W9（v2 §六），含命名、四要素描述、505→498 排除披露、白名单汇总、"同量级≠等价"。
12. **扩大的 QA/元数据/总册同步**：接纳。W6 增列 judge 供应商/base URL/请求时间与 unknown 标注规则；W7 增缺陷检查清单与 manual spot check 命名；W8 建"主张→来源→脚本→论文位置"清单覆盖 §5.4 全部 12 项。

## 三、对复审 §6 建议项的处置

1. 全 50 条 3 轮：不做，以 10 条分层子集 3 轮替代（复审认可的最低方案）；
2. 第二 judge/judge 重复：不做，语义结果明确降级为探索性；
3. 真实声卡：不做，指标命名与限制声明按 §2.5；
4. 增强条件同一样本集重跑：不做，论文只做条件内 A/B 比较，不做跨条件因果排序；
5. GPU 竞争：不做，limitations 声明独占 GPU；
6. 原平台 TTFA：不做，新 TTFA 仅绑定第二平台。

## 四、后续流程

方案 v2 经复审通过后：本机先实现 W3/W4/W5（含 self-test）并执行，W6/W9 文档同步；W1 脚本完成后送审，审查通过即出 GPU handoff（含 W2 环境清单、TTS 探活前置、冒烟→正式顺序）；正式结果经结果级 QA 后装配新 Table VIII，再执行 W8 阶段 2 并按初审 §8 验收标准写整改回复。
