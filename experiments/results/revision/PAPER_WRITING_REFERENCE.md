# CISR 修订论文写作参考（数据与口径总册）

> **读者**：论文编写人。**用途**：按审稿意见逐条给出回应要点、论文修改位置、可直接引用的数字、
> 数据文件位置与写作注意事项。所有数字均经独立复核（审查 20260821-PAPER-DATA 三轮闭环，r3 放行）。
> **配套**：`PAPER_HANDOFF.md`（逐实验产物）、`REVISION_CHANGELOG.md`（执行流水与裁决记录）、
> `CISR_REVISION_PLAN.md`（原方案 §8 改稿清单）、`experiments/review/`（各轮评审与回复）。

---

## 〇、全局铁律（违反会被审稿人抓，写作前必读）

1. **平台绑定**：原平台（2025-12 机器）数字 = Table III、Table IV（498 重算亦来自原 exp2 归档）、
   Table V、Fig.6；第二平台（KVM Xeon Gold 6133 @2.5GHz + 2×RTX 3090）数字 = Table VI（E2）、
   Table VII（E3 三方）、Table VIII（E5/E6/补测）、E1 绝对值。**两平台不得混排同栏**；
   Table IV 与 Table VII 并存时各加平台脚注。
2. **禁缩放红线**：不对第二平台数字做任何缩放去对齐原平台绝对值，所有数字如实呈现。
3. **同机对比**：每个结论只用同机产生的对比数字。
4. **max_tokens 口径**：Table III/IV/V = 50；E4/R5/R6-decode/TTFC = 128。新增小节须写明，勿混淆。
5. **LA 方法描述**必须写修复后语义（绝对时间轴提交 + 句界裁剪 + la_max_buffer_s=15.0），
   不得只写 "LocalAgreement-2"（评审保留项；修复过程文档可应询出示，见 §四）。
6. **摘要改善率**：统一写 **"70%–74%（两平台复现）"**（原平台 74.3%/65.6%，第二平台 70.4%/64.7%）。
7. Table VIII 脚注注明各分项来源（端点=E5、post=E4、decode=补测、TTFC=E6，同机同码同 50 样本；
   A 行 decode/TTFC 为估计，表格保留"全实测/估计"标注）。
   **（2026-08-22 起：Table VIII 数据源已整体切换为 R7 统一实测，本条装配口径随 §二历史稿一并作废。）**
8. **R7 唯一数据源（2026-08-22 审查终裁）**：TTFA/Table VIII 只可引用 `r7_ttfa_unified/`
   已过审数字（r7_main 140/140 + tts_control 32 偏差豁免采信）；不得引用作废装配稿
   （ttfa_budget.csv）或未 QA 估计值，不得新旧混排。引用 tts_control 数字必须附带
   流程偏差豁免披露（原文见 `review/…/deviation-waiver-r7-tts-control-20260822.md` §3）；
   不得写"控制实验已获单独放行后执行"或"全部实验严格按预授权顺序执行"。

---

## 一、意见1：真实语音验证 → 新 Table VI + §V 新小节 + limitations

**回应要点**：自建 LibriSpeech/AISHELL-1 长语音集（可确定性重建），干净集流式优势与原合成集同量级；
增强集 10/12 条件改善 +22.8%~+31.6%；babble 为诚实披露的边界条件（进 limitations）。

**数据集描述（§V-A 用）**：LibriSpeech test-clean 按章节 / AISHELL-1 按说话人顺序拼接，
句间静音 U(0.2,1.0)s，seed=42 字节级可重建；各 75 条（long 30 / very_long 30 / extra_long 15）；
12 个增强变体（SNR 20/15/10dB、变速 0.9×/1.1×、babble 15dB）各 30 条。
转写 sanity：librispeech WER 2.98%；aishell1 CER 10.73%（修正口径；数字写法失配见下脚注）。

**干净集 TTFT（ms，mean）**：

| 数据集 | 系统 | long | very_long | extra_long | extra_long 改善率 |
|---|---|---|---|---|---|
| librispeech | A / B | 1655 / 1773 | 2957 / 1628 | 4805 / **1559** | **67.5%** |
| aishell1 | A / B | 1627 / 1659 | 2982 / 1707 | 5140 / **1763** | **65.7%** |

**转写质量（干净集，离线同口径终版；英文大小写折叠、中文去接缝空格）**：
librispeech_clean：A WER 2.97% / CER 1.05%；B WER 5.65% / CER 1.36%。
aishell1_clean：A WER 10.77% / CER 10.77%；B WER 11.80% / CER 11.80%
（中文 WER/CER 同为字级粒度，数值一致属正常）。

**写作注意**：
- **zh CER 脚注（必加）**：aishell1 参考用中文数字、Whisper 输出阿拉伯数字（"百分之二十二点六" vs
  "22.6%"），影响 49/75 样本（受影响 mean 0.1476，未受影响 0.0313）；不注明会被质疑与 librispeech 反差。
- **逐条件行用 `overall` 行**（speed 变体时长重判产生 medium 子组：librispeech 5 条、aishell1 3 条）。
- **babble（limitations）**：librispeech_babble streaming 3425ms vs A 2326ms（−47.3%），
  aishell1_babble +6.5%；归因链 = VAD 对 babble 过度触发 → 段积压（最慢样本音频结束后 21s 排空）
  + 噪声段空输出（12/30、5/30）→ 长尾（median 2020ms vs mean 3425ms）。不得写"各种噪声下均优"。
- **"near-zero error"限定合成集**：真实语音 System A 上界 librispeech WER 2.97% / aishell1 CER 10.8%。
- 增强集 10 条件改善率 +22.8%~+31.6%（Table VI 建议 12 条件全列，用户已裁决"要严谨"）。
- 空输出样本 `asr_time`/`llm_prefill_time` 为哨兵值，统计时剔除；`ttft` 不受影响。

**数据文件**：`r2_real_speech/wer_real.csv`、`ttft_real.csv`（逐条件 × 模式 × 分组）、
构建/增强 manifest、`qa_static.csv`、`qa_transcribe.corrected.csv`；机制数字见 changelog 2026-08-19 E2b 条目。

---

## 二、意见2：TTFA 端到端预算 → 新 Table VIII + §I/§III-C 口径句

> **⚠️ 数据源已切换（2026-08-22 审查终裁）**：下表跨运行装配口径（含 ALL 行与 A 行估计）
> **整表作废**，不得以任何形式入论文；新 Table VIII 唯一合法数据源为 **R7 统一时间轴实测**
> （`r7_ttfa_unified/`，r7_main 140/140 + r7_tts_control 32，审查通过/偏差豁免采信）。
> 装配与写作口径在 W8 阶段 2 定稿；在此之前本节仅保留历史装配稿供对照。

**R7 实测主数字（repeat0，n=50，ms；`ttfa_summary_r7_main.csv`）**：

| TTFA = first_playable_pcm − speech_end | mean | p50 | p90 | p95 |
|---|---|---|---|---|
| B streaming ALL | 5481.9 | **3113.7** | 10506.6 | 11656.3 |
| B streaming zh / en | 3303.3 / 7660.5 | 2603.0 / 7577.0 | — | — |
| A non-streaming ALL | 22425.7 | **22269.9** | 25588.8 | 26887.4 |

子集三轮 CV mean 7.73% / max 20.70%；组件分项（flush/close→first_token→text_ready→
tts_req→first_pcm→playable）齐全可分解。匹配文本 TTS 控制：tts req→first_pcm
mean 7076ms（32 条复算一致）。**引用控制数据必带流程披露**（偏差豁免，非事后追认；
原文见 `review/…/deviation-waiver-r7-tts-control-20260822.md` §3）。

**历史装配稿（⚠️整表作废，仅供对照）**（ms，mean；两轮 P0 裁决最终口径）：

| 系统 | 语种 | T_endpoint | T_post（E4 TTFT） | T_decode首句 | T_TTFC | **TTFA** |
|---|---|---|---|---|---|---|
| B（作废） | zh | 52.7 | 1398.9 | 142.3 | 13985.7 | **15579.6** |
| B（作废） | en | 53.5 | 1447.0 | 635.7 | 11858.9 | **13995.0** |
| B（作废） | ALL | 53.1 | 1422.9 | 389.0 | 12922.3 | **14787.3** |
| A（作废，decode/TTFC 估计） | ALL | 53.1 | 3928.9 | 389.0* | 18302.4* | **22673.4** |

（A 分语种行见 CSV；* = 估计项：decode 用 B 同语种均值代理、TTFC=0.09s/字符 × A 回复均长 zh 203.8 字符。）

**写作注意**：
- B 行 post 口径经两轮裁决：最终取 **E4 streaming TTFT 直接实测**（端点触发 flush 尾延迟）；
  E5 的 2s 追加静音是测量装置属性（其中 ~410ms 为端点积压真实排空，故不取 E5 窗后 1012.5ms）。
  正文一句话说明即可，细节备询（changelog 2026-08-21 两条 P0 条目）。
- E5 端点分布：mean 53ms / median 109ms / p90 208ms；`final_enqueue_wait≈2s` 是装置属性勿当系统等待。
- decode 分语种差异（en 635.7 vs zh 142.3ms）全部来自首句 token 数（速率相同 25.9/26.0 tok/s）。
- **TTFC 必须带长度关系**（用户裁决）：~0.09s/字符（3 字符→1.4s，45→7.6s，200→17.9s），
  该部署为句段级流式；可给短回复场景数字。本机 CPU 放大 TTFC（平台披露覆盖；原机补测已裁决不需要）。
- 网络延迟：单机部署线程队列通信，T_Net≈0，正文说明界定理由。

**数据文件**：`r6_ttfa/ttfa_budget.csv`、`endpoint/`、`decode_to_first_sentence.csv` + `.summary.txt`、
`tts_first_chunk.csv`（+ 冷启动对照 `tts_first_chunk_run1_cold.csv`）。

---

## 三、意见3：统计显著性补强 → Table III/IV/V 加列 + Fig.6 + E1 小节

**回应要点**：三张主表补 mean±std/P95/P99；50 样本 × 3 轮重复测量 ~~CV<5%~~（⚠️作废，ddof=1 新口径见 §十）；平台稳定性表述换可辩护口径。

- **Table III**（原平台，已逐字核验与原文一致）：streaming Long/Very/Extra = 1126.63/1099.16/1087.70ms；
  baseline extra_long 6745.57ms（成对排除 1 条运行错误样本，规则见 `table3_filter_manifest.json`）。
  加列数据：`r1_stats/table3_latency_percentiles.csv`。
- **连锁修正**：baseline extra_long 换 6745.57 后，"最长分组平均减少 5.67 秒"→ **5.66 秒**（两处）；
  "34.6%–83.9%" 区间复算不变（83.88%），保留。
- **平台稳定性表述**（替换"P99≤1.5×mean"旧口径）：System B 流式 P99 有界（1979/2174/2605ms），
  Long→Extra Long 仅增 1.32×（baseline 4.96×）；Extra Long 流式 P99 为 baseline 的 0.21 倍。
- **E1 重复测量**（第二平台）：三轮 streaming mean 1434.0/1482.1/1454.9ms（极差/mean 3.3%），
  逐样本 CV ~~mean 4.2% / median 3.3%~~（⚠️ddof=0 旧口径；新口径 mean 5.19%/median 4.05%，见 §十与 repeat_cv_summary.csv）；
  non-streaming 4063.3/4244.2/4118.5ms。结论：~~"3 轮 CV<5%"~~（⚠️作废，改为 ddof=1 全分布表述）。
- **Fig.6 重绘**：`results/revision/fig/Fig6.pdf`（12 等频分箱 mean 折线 + P5–P95 阴影带，
  分箱数据 `Fig6.bins.csv`）。

---

## 四、意见4：LocalAgreement-2 基线 → 新 Table VII + §V 新小节

**回应要点**：同权重同引擎同分段器，唯一变量是 ASR 上下文/提交策略；同机三方对比无跨机污染。

**Table VII（ms，mean；498 条干净成对清单，第二平台）**：

| 系统 | mean TTFT | long | very_long | extra_long | WER(ALL) | zh CER | en WER |
|---|---|---|---|---|---|---|---|
| System A | 5310.8 | 1958 | 3906 | 7698 | 0.0951 | 0.0674 | 0.1527 |
| **System B** | **1573.9** | 1464 | 1551 | 1638 | 0.1047 | 0.0732 | 0.1702 |
| LA-2 基线 | 2115.0 | 1741 | 2200 | 2230 | 0.1073 | 0.0835 | 0.1566 |

（WER/CER 为离线同口径重算：英文折叠、中文去接缝空格；见 `r3_baseline_la/wer_la_vs_b.csv`。）

**写作注意**：
- 统一表述："**LA-2 基线比 System B 慢约 34%**"（或"B 比 LA 低约 26%"），**二者只取其一**。
- 质量：三系统同引擎同量级；LA/SysB 转写长度比 mean 0.98 / median 0.99；divergence mean 1.0 / max 7。
- 机制归因一句话：LA 全缓冲重解码开销 + 长音频退化（very_long 以上 LA 2230 vs B 1551–1638）。
- **方法描述必须写修复后语义**（铁律 5）；实现经历一次 bug 修复重跑（错帧 + 裁剪幻听双机制），
  过程文档在 `r3_baseline_la/handoff/` 与 `experiments/review/20260820-E3LA/`，被追问可出示。
- 中间消融臂 `streaming_asr_only`（原平台 1171.0ms）：用户裁决**只加脚注标明为原平台数字，不补跑**。
- 样本量表述：498 条成对干净子集（排除：运行错误 3 条 + 流式 TTFT>10s 挂起 4 条）；
  论文"50 utterances per group"表述同步改为实际样本量（108/150/240）。

**数据文件**：`r3_baseline_la/la_results_*` 三件套、`system_ab_rerun/`、`wer_la_vs_b.csv`、`ttft_la_vs_b.csv`。

---

## 五、意见5：提交正确性与下游语义 → §IV 新小节（机制）+ §V 新小节（语义）

**回应要点（机制，R4）**：回滚下发 = 0（构造保证）；内部重识别漂移实测存在但下游不可见；
接缝分歧常见但仅为 BPE 重切分，token 序列解码文本逐样本恒等。

- **提交分歧**（`r4_commit/commit_divergence.json`）：375 commit / 224 correction；
  涉及段 224/425（**52.7%**）、涉及样本 49/50；漂移幅度编辑距离 **mean 2.3 字符 / p90 6 / max 16**
  （归一化 mean 5.6% / max 47.1%，含实词级漂移，top 示例见 JSON）。
  外部一致性：streaming 拼接 vs System A 全量转写 **WER 4.93% / CER 2.69%**（max 14.2%）。
- **分词接缝**（`r4_commit/tokenizer_seams.csv`）：25/50 样本存在接缝分歧（50.0%）；
  分歧块 mean 4.36 处（max 8）、oneshot 侧 mean 5.60 token（max 12）；
  **两条 token 序列解码文本 50/50 完全一致**（分歧均为 '.'+'Is'→'.Is' 型跨缝合并）。
- **append-only 表述（用户已裁决软化）**："对已提交输出无回滚（下游不可变）；内部重识别存在漂移
  （实测 224 次/50 样本、涉及段 52.7%，幅度见分布），不影响下游"——不得写"内部从不变化"或"同音字/标点级"。

**回应要点（语义，R5 三轨）**：

| 轨道 | 数字 | 用法 |
|---|---|---|
| A：bge-m3 余弦 | **mean 0.8832**（std 0.0755，min 0.6232，p10 0.7951） | 主证据 |
| B1：成对等价 judge | mean 2.96/5，≥4 分 40.0% | 如实报告 + 归因（两臂独立采样推荐内容不同 + 128 token 截断） |
| B2：独立意图满足盲评 | **A 3.10 vs B 3.04（差 +0.06）**；≥4 分 30.0% vs 26.0% | 主证据（~~统计不可区分~~⚠️删除；只报差值与 CI，见 §十） |

- **写作约束**：以轨道 A + B2 为主证据；不得单独引用 B1 低分；不得把绝对分 ~3/5 写成"两模式质量都差"
  （截断上限对两臂同等作用）；披露 judge 型号（DeepSeek deepseek-v4-flash、temperature 0、A/B 顺序随机化）。
- 定性 case：高一致 `crosswoz_8717_turn3`（cosine 0.9784，judge 4）；差异最大 `crosswoz_7196_turn3`
  （cosine 0.6232，judge 1 但 B 独立分 4 > A 的 2——成对低分不代表 B 退化）。
- 数据文件：`r5_semantic/semantic_consistency.csv` + `.summary.txt` + `judge/`、`judge_solo/` 逐样本 JSON。

---

## 六、Table IV 全表替换（498 重算口径，P1-2 改稿清单）

| 量 | Long | Very Long | Extra Long |
|---|---|---|---|
| Baseline TTFT | 1698.88 → **1690.81** | 3300.58 → **3307.96** | 6518.40 → **6515.67** |
| ASR-only TTFT | 1064.18 → **1064.67** | 1171.02 → **1155.09** | 1228.77 → **1228.82** |
| Full TTFT | 1084.17 → **1087.69** | 1154.06 → **1152.36** | 1114.57 → **1188.00** |
| ASR 增益 ms | 634.70 → **626.13** | 2129.56 → **2152.87** | 5289.63 → **5286.86** |
| ASR 降幅 % | 37.4 → **37.0** | 64.5 → **65.1** | 81.1 → **81.1** |
| KV 增益 ms | −19.99 → **−23.01** | +16.96 → **+2.73** | +114.20 → **+40.82** |
| Full vs Baseline % | 36.2 → **35.7** | 65.0 → **65.2** | 82.9 → **81.8** |

**连锁重写**：① "KV 收益占比约 9.3%" → **3.3%**（40.82/1228.82），"收益愈加显著"递进叙述删除重写；
② Very Long KV 增益 +2.73 几乎归零，原句不能保留；③ "1.1s 左右" → "约 1.1–1.2s"或分组值；
④ 样本量表述 150 → 498（含排除规则）；⑤ 全文检索旧数字残留（摘要/正文/结论）。
数据：`r1_stats/table4_ablation_percentiles.csv`；分位数加列同文件。

---

## 七、回复信证据链（意见 → 修改位置 → 数据文件 → 关键数字）

| 意见 | 论文修改位置 | 数据文件 | 关键数字 |
|---|---|---|---|
| 1 真实语音 | Table VI + §V 新小节 + limitations | `r2_real_speech/wer_real.csv`、`ttft_real.csv` | extra_long 改善 67.5%/65.7%；10 条件 +22.8~31.6%；babble −47.3% |
| 2 TTFA | Table VIII + §I/§III-C | `r7_ttfa_unified/r7_main/`、`tts_control/`（R7 唯一数据源；旧 `ttfa_budget.csv` 作废） | B streaming p50 3113.7ms vs A 22269.9ms；组件分项齐全；TTS 控制 7076ms（须带偏差豁免披露） |
| 3 显著性 | Table III/IV/V 加列 + Fig.6 + E1 小节 | `r1_stats/`、`fig/Fig6.pdf` | CV mean 4.2%/median 3.3%；P99 有界 |
| 4 LA 基线 | Table VII + §V 新小节 | `r3_baseline_la/wer_la_vs_b.csv`、`ttft_la_vs_b.csv` | LA 比 B 慢 34%；质量同量级 |
| 5 机制+语义 | §IV + §V 新小节 | `r4_commit/commit_divergence.json`、`tokenizer_seams.csv`、`r5_semantic/` | 回滚 0；漂移 224 次 mean 2.3 字符；余弦 0.883；A−B +0.06 |

## 八、Limitations 素材（用户裁决口径）

1. 独占 GPU 环境、单机部署（GPU 竞争实验不做）；
2. babble 多人噪声下流式退化（完整归因链，见 §一）；
3. TTS 为句段级流式部署，TTFC 与回复长度近似线性，且本机 CPU 放大绝对值（平台披露）；
4. append-only 内部重识别漂移（分布见 §五）；
5. 真实语音集为拼接长语音（构建方法与 seed=42 可重建性在 §V-A 说明）。

## 九、待决清单最终状态（全部已裁决）

1. 中间消融臂 → 脚注标原平台数字，不补跑；2. babble → 12 条件全列；
3. 摘要改善率 → "70%–74%（两平台复现）"；4. E6 原机对照 → 不做；
5. append-only → 两层精确表述；6. TTFC → 长度关系式 + 短回复示例。
（TTFA 口径经两轮 P0 裁决定稿=方案 (a)，见 §二。）

## 十、数据状态：⚠️ P0 整改中（2026-08-21 PRE-PAPER-AUDIT 后降级）

**本册"全部定稿可直接动笔"状态已撤销。** 撰稿前审计（`experiments/review/20260821-PRE-PAPER-AUDIT/`）
发现 5 项 P0，整改方案 v3.1 已冻结（Gate 0/1），以下条目在对应整改落地前**不得引用**：

1. **Table VIII（§二）**：跨运行装配口径作废（含"B 全实测"标注与 A 行 0.09s/字符 TTFC 估计）；
   ~~以 W1 统一时间轴实测（`run_ttfa_unified.py`，GPU 待跑）替换后方可入论文~~
   **→ 已完成（2026-08-22）**：W1 正式数据 r7_main 140/140 + tts_control 32 审查通过
   （控制侧为偏差豁免采信），见 §二 R7 主数字；装配（W8 阶段 2）为唯一剩余动作；
2. **"3 轮 CV<5%"**：作废；新口径（ddof=1）已出：`r1_stats/repeat_cv_summary.csv`
   （B mean 5.19%/median 4.05%/P90 10.73%/max 18.96%，19/50>5%；A 5.23%/4.65%/9.92%/14.01%，23/50>5%）；
3. **WER/CER 表头**：宏平均口径须标注 mean-utterance；corpus 口径已补
   （`wer_real.csv`/`wer_la_vs_b.csv` 新增 wer_corpus/cer_corpus 及 S/D/I/N 列）；
4. **"统计不可区分"（§五 B2）**：删除；只报 B−A=−0.06、bootstrap 95% CI [−0.34, 0.22]
   （`stats_inference/paired_inference.csv`），语义为探索性证据；
5. **"端点等待约 0.05s"物理解读**：作废（E5 该字段 8/50 负值，异时间轴量）。

已完成的整改产物（本机，无需 GPU）：W3 CV（上）、W4 corpus WER/CER（上）、
W5 成对统计（21 个比较，含 Holm 校正）、W6 语义元数据（`r5_semantic/REPRO_METADATA.md`）、
W9 LA 方法说明（`r3_baseline_la/LA_METHOD_AND_EXCLUSION.md`）、W7 抽检模板
（`r2_real_speech/MANUAL_SPOT_CHECK.md`，试听待人完成）。
**W1 统一 TTFA：已完成并过审（2026-08-22 终裁）**——r7_main 140/140（QA 0，结果级 47/47）、
r7_tts_control 32（数据级通过，流程偏差豁免采信，不构成追认；裁定量与登记见
`review/…/review-results-qa-r7-main-20260822.md`、`…/deviation-waiver-r7-tts-control-20260822.md`）。
**W2 环境记录：已完成**（`r7_ttfa_unified/env/platform_conditions.txt`，hash `a4c40057…` 入全量绑定）。
**剩余**：W8 阶段 2（新 Table VIII 装配，机械工作）→ 论文修订。

历史快照（整改前口径，仅供对照，不得直接入论文）：最后一次口径刷新 2026-08-21
中文 CER 去接缝空格（主机重跑 4b9587b，12 行 aishell1 CER 回落、英文 WER 与 TTFT 逐字节不变）。
