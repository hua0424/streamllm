# 修订论文数据就绪度审查报告（2026-08-21）

审查对象：

- 路线图：`experiments/CISR_REVISION_PLAN.md`（R7 §8.1 图表清单、§8.3 文字修改点）
- 决策与证据链：`experiments/results/revision/REVISION_CHANGELOG.md`、`PAPER_HANDOFF.md`、`PAPER_IMPACT_NOTES.md`
- 数据产物：`r1_stats/`、`r2_real_speech/`、`r3_baseline_la/`、`r4_commit/`、`r5_semantic/`、`r6_ttfa/`、`fig/`
- 论文原文（对照基线）：`paper/tougao_new/3次调整/latex/main.tex`

审查目标：论文编写前逐项核验数据是否满足论文要求，防止改稿时引用错误数字、或论文论点缺乏数据支撑。

## 一、独立复核结果（本机重算，非转录自 changelog）

| 项 | 复核内容 | 结果 |
|---|---|---|
| Table III | r1 重算 streaming 三组均值 1126.63/1099.16/1087.70 vs 论文原文 | **逐字一致**；baseline extra_long 6753.43→6745.57（成对排除 1 条，规则在 `table3_filter_manifest.json`） |
| Table V | 六格 WER/CER + 三档 asr_time vs 论文原文 | **逐字一致**（0.0895/0.0196/0.0083/0.0203/1327.48 等全部相同） |
| Table IV | 498 清单重算值（108/150/240） | 与 `table4_ablation_percentiles.csv` 一致；**与论文原文数字不同**（见 P1-2） |
| E1 重复测量 | 三轮 streaming 均值、逐样本 CV 独立重算 | 1434.0/1482.1/1454.9 ✓；CV mean 4.2%/median 3.3% ✓（p90 我算 8.7%，changelog 记 8.8%，分位取法差异，见 P2-6） |
| E2 真实语音 | 逐条件改善率重算 | clean extra_long 67.5%/65.7% ✓；snr/speed 十条件 +22.8%~+31.6% 逐条复算全落在带内 ✓；babble −47.3%/+6.5% ✓；wer_real 30 行/ttft_real 102 行 ✓ |
| E3 三方 | la_results + system_ab_rerun 源 JSON 重算 | A 5310.8 / LA 2115.0 / B 1573.9（n=498×3）✓；分组 1741/2200/2230 vs 1464/1551/1638 ✓ |
| R4 | commit_divergence.json + tokenizer_seams | 375 commit/224 correction、涉及段 52.7%、49/50 样本、edit mean 2.32、外部 WER 4.93% ✓；接缝 25/50 有分歧、解码文本 50/50 恒等 ✓ |
| R5 | semantic_consistency.csv 重算 | cosine mean 0.8832 ✓；case crosswoz_8717_turn3=0.9784、crosswoz_7196_turn3=0.6232（judge 4/1）✓ |
| R6 decode | CSV + summary + GPU 侧 RUNINFO QA | 50/50 error 0、mean 389.0ms、sentence_end_found 100%、QA 7/7、输入 sha256 与 E4 产物一致 ✓ |
| E6 TTS | tts_first_chunk.csv 重算 | zh 13985.7 / en 11858.9 ✓ |
| Table VIII | ttfa_budget.csv 逐行求和 | zh/en/ALL 六行分量和=总计全部 ✓；A 行 TTFC 估计复核（zh 回复均长 203.8 字符 × 0.09s/字符 = 18342ms）✓ |
| Fig.6 | Fig6.bins.csv | 24 行（2 模式 × 12 等频分箱），含 p5/p95 列 ✓ |

**总判定：数据链完整、可复核、可追溯（RUNINFO/QA/manifest 齐全），批准进入论文修改阶段。** 但下面 1 项 P0 口径矛盾必须在 Table VIII 数字写进论文前裁决，P1×3 必须在成稿前处理，否则会产生"论文引用错误数字"或"数据不支持论点"的问题。

## 二、P0：Table VIII 两系统行对 2 秒追加静音的处理不对称，且与 PAPER_HANDOFF 自述矛盾

### 证据（本机从 E5/E4 源 JSON 重算）

- System B 行的 `t_post_endpoint`（3011.9ms）= `first_token − speech_end`（3065.1ms）− 端点等待（53.1ms）。其中 **[53.1ms, 2052.6ms] 的 1999.5ms 是等追加静音推完的装置等待**（`final_is_final_segment_enqueue_time` 均值 2052.6ms），**真实端点后处理只有 1012.5ms**（首 token − final enqueue）。
- System A 行的 `t_post_endpoint`（3928.9ms）取 E5 non-streaming 的 `ttft` 字段——**不含这 2 秒**：E4（无追加静音）同 50 样本 non-streaming ttft 均值 3984.0ms，与 E5 的 3928.9ms 同量级；且 E5 non-streaming 记录中 `audio_end − speech_end = 0`。
- 即：**B 行含 ~2.0s 装置等待，A 行不含**。当前表因此把 B−A 的 TTFA 差距从对称口径的 ~8.3s（双方剔除 2s）或 ~10.3s（双方计入 2s）压到 6.3s——方向上对作者保守（不利），不是结果美化，但属于口径不一致，审稿人拿 E5 的 RUNINFO 一对就能发现。
- 同时 `PAPER_HANDOFF.md` 写"final_enqueue_wait≈2s 为测量装置属性，**不进预算表**"——这与装配结果**事实不符**：2s 就在 B 行 post 分项里。论文若照抄这句话再引用 16.4s 总计，构成自相矛盾。

### 需求方/开发需裁决（三选一，均为只改口径或文档，不必重跑）

1. **维持 E5 链条（当前 CSV 口径）**：论文如实写明"post-endpoint 项含 2s 静音窗（端点策略保守值），其中纯处理 ~1.0s"，并同步修正 PAPER_HANDOFF 的"不进预算"表述；A 行是否对称加 2s 需一并说明；
2. **对称剔除 2s**：B 行 post 改用"首 token − final_enqueue"（1012.5ms）或 E4 同期 flush 口径，总计 ~14.4s；
3. **改用 E4 TTFT 口径**（53 + 1423 + 389 + TTFC ≈ 15.2s）——这正是 GPU 侧 `r6_ttfa/RUNINFO.md` 移交说明里给的公式，与本机装配 CSV（用户 2026-08-21 裁决的 E5 链条）**不一致**，两份文档目前各说各话，必须统一。

无论选哪个：`PAPER_HANDOFF.md` §TTFA 的公式行（当前写 `TTFT(E4 streaming mean 1423ms)`，与 CSV 的 3012ms 冲突）、GPU 侧 RUNINFO 移交说明、changelog 三处要对齐到同一口径。

## 三、P1（成稿前必须处理）

### P1-1 PAPER_HANDOFF 的 E4 漂移表述过轻，与实测分布不符（论文编写人主读该文档）

`PAPER_HANDOFF.md` E4 条目仍写漂移"属同音字/标点级"。2026-08-21 `commit_divergence.json` 完整统计：涉及段 52.7%、edit mean 2.32 字符、归一化比率 mean 5.6%、**max 47.1%**，top 示例含实词级漂移（电话号码串整体消失、"药油腐火锅洋镇店"→"要约复火锅、洋阵点"）。changelog 已自我修正该表述（"多数为小改但存在实词级漂移"），但 PAPER_HANDOFF 未同步——照抄风险高。论文 §IV 新小节措辞应以实测分布为准：**"回滚下发 = 0（构造保证）；内部重识别漂移实测存在（224 次 / 49~50 样本，涉及段 52.7%），编辑距离 mean 2.3 字符、p90 6、max 16；下游不可见"**，不得写"同音字/标点级"或"极低频"。

### P1-2 Table IV 换用 498 清单重算值后，正文 §IV-B 的连锁数字全部要更新

计划 §4.2 已决定 Table IV 按 498 干净成对清单重算（旧数字含不可复现的手工修复）。新旧对照（旧 = 论文原文，新 = `table4_ablation_percentiles.csv`）：

| 量 | Long | Very Long | Extra Long |
|---|---|---|---|
| Baseline TTFT | 1698.88 → **1690.81** | 3300.58 → **3307.96** | 6518.40 → **6515.67** |
| ASR-only TTFT | 1064.18 → **1064.67** | 1171.02 → **1155.09** | 1228.77 → **1228.82** |
| Full TTFT | 1084.17 → **1087.69** | 1154.06 → **1152.36** | 1114.57 → **1188.00** |
| ASR 增益 ms | 634.70 → **626.13** | 2129.56 → **2152.87** | 5289.63 → **5286.86** |
| ASR 降幅 % | 37.4 → **37.0** | 64.5 → **65.1** | 81.1 → **81.1** |
| KV 增益 ms | −19.99 → **−23.01** | +16.96 → **+2.73** | +114.20 → **+40.82** |
| Full vs Baseline % | 36.2 → **35.7** | 65.0 → **65.2** | 82.9 → **81.8** |

特别注意三点论点弱化/重写：

1. 原文"KV Cache 相对整体的收益占比达到约 9.3%（114.20/1228.77）"→ 新口径 **3.3%（40.82/1228.82）**，"收益愈加显著"的递进叙述要重写；
2. Very Long 的 KV 增益从 +16.96 掉到 **+2.73（几乎归零）**，原文"在 Very Long 与 Extra Long 组 KV 增益分别为…"的句子不能保留；
3. "Full Streaming 将 TTFT 均值控制在 1.1s…左右"→ 新值 1087.69/1152.36/1188.00，Extra Long 1188ms 更接近 1.2s，建议改"约 1.1–1.2s"或直接报分组值。

另外原文两处样本量表述要改："挑选了 Long 及以上长度的各 50 个样本"（150 条）→ 实际 **108/150/240 共 498 条**（成对干净子集，需给出排除规则：运行错误 + 流式 TTFT>10s 挂起）。若旧数字任何一处残留（含摘要、正文、结论），审稿人对照表格即发现前后矛盾。

### P1-3 LA 对比优势的百分比表述歧义（34% vs 25.6%）

实测 LA 2115.0 vs B 1573.9：**B 比 LA 低 25.6%**；**LA 比 B 高 34.4%**。`PAPER_IMPACT_NOTES.md` 影响项 4 写"LA 比 System B 慢约 34%"（正确口径），但 `PAPER_HANDOFF.md` 写"TTFT 优于 LA-2 基线约 34%"——按常规读法（B=LA×0.66）是错的。论文统一用"**LA-2 基线比 System B 慢约 34%**"或"**System B 比 LA-2 低约 26%**"，二选一，不得混用。

## 四、P2（写作注意事项，changelog 已有依据，防止误用）

1. **Table III 连锁**：baseline extra_long 用 6745.57 后，摘要与正文两处"最长分组平均减少 5.67 秒"应为 **5.66 秒**（6745.57−1087.70=5657.9ms）；"34.6%–83.9%"区间复算后不变（83.88%），可保留。
2. **Table VI zh CER 口径脚注**：aishell1 参考文本中文数字 vs Whisper 阿拉伯数字写法失配影响 49/75 样本（受影响 mean CER 0.1476，未受影响仅 0.0313）。引用 zh CER（如 streaming 0.1652）必须加脚注，否则与 librispeech WER 5.65% 的反差会被质疑。（changelog 2026-08-21 已登记，PAPER_HANDOFF 未提，建议补。）
3. **speed 变体分组漂移**：变速后时长重判分组，speed11 条件出现 medium 子组（librispeech 5 条、aishell1 3 条）。Table VI 逐条件行建议用 `overall` 行，或注明分组按变速后时长重判。
4. **R5 三轨用法约束**：以轨道 A（cosine 0.88）+ B2（独立意图满足 A 3.10 vs B 3.04，差 0.06 不可区分）为主证据；B1 成对 judge 2.96/5 如实报告并归因（两臂独立采样导致推荐内容不同 + 128 token 截断），需披露 judge 型号（DeepSeek deepseek-v4-flash、顺序随机化）；不得单独引用 B1 低分，也不得把"绝对分 ~3/5"写成两模式质量都差（截断上限对两臂同等作用）。
5. **decode 分语种差异归因**：en 635.7 vs zh 142.3ms 全部来自首句 token 数（两语种解码速率相同 25.9/26.0 tok/s），论文引用分语种数字时按此归因。
6. **E1 CV p90**：changelog 记 8.8%，我按 0 基下标第 45 位算 8.7%，属分位约定差异，无实质影响；论文引用时建议只报 mean/median（4.2%/3.3%）。
7. **"near-zero error"表述**：原文 §IV-C"非流式 System A…WER/CER 近似为 0"仅对合成集成立；R2 实测真实语音 System A 上界为 librispeech WER 2.97% / aishell1 CER 10.8%（含数字写法因素），修改稿应把 near-zero 限定在合成集表述内（这正是指见 1 要求的真实上界数据）。
8. **babble 进 limitations 的口径**：数据支持"VAD 对 babble 过度触发→段积压 + 空输出（12/30、5/30），median 2020ms vs mean 3425ms 长尾"的完整归因链；不得写"各种噪声条件下均优于非流式"。空输出样本的 asr_time/llm_prefill_time 为哨兵值，勿纳入统计。

## 五、INFO：平台绑定与混排禁区（裁决 A/B/D 的落实清单）

- **原平台数字**（2025-12 机器）：Table III、Table IV（重算亦来自原 exp2 归档）、Table V、Fig.6。
- **第二平台数字**（KVM Xeon 6133 + 2×3090）：Table VI（E2）、Table VII（E3 三方，含 A/B 本机重跑）、Table VIII（E5/E6/decode）、E1 重复测量绝对值。
- 两平台同名字段差异巨大（如 extra_long baseline TTFT：原平台 6515.67 vs 第二平台 7697.5；streaming mean：1114.57 vs 1637.5），**任何表格/正文不得混排**；Table IV 与 Table VII 并存时各加平台脚注。
- 摘要改善率按裁决 D："70%–74%（两平台复现）"或绑定平台（原平台 74.3%/65.6%，第二平台 70.4%/64.7%）。
- max_tokens 适用范围：Table III/IV/V 均为 50；E4/R5/R6（decode、TTFC、语义）为 128，新增小节描述时须写明，勿与 §V-A 的"50 token"设置混淆。
- Table VII 方法描述必须写修复后 LA 语义（绝对时间轴提交 + 句界裁剪 + la_max_buffer_s=15.0），不得只写"LocalAgreement-2"（评审 R2 保留项）。

## 六、结论

数据侧就绪：Table III–VIII、Fig.6、§IV/§V 新小节、limitations 素材全部有实测支撑且本机复核通过，changelog 证据链完整。放行进入论文修改，前置条件按优先级：

1. **（阻塞 Table VIII）** P0 口径三选一裁决，并对齐 PAPER_HANDOFF / GPU RUNINFO / changelog 三处表述；
2. **（成稿前）** P1-1 更新 PAPER_HANDOFF E4 表述；P1-2 按 498 重算值全面替换 §IV-B 数字与样本量表述；P1-3 统一 LA 优势口径；
3. P2 各项在对应小节写作时照单执行。

本报告只登记问题与证据，未改动任何程序与数据产物；整改方案由开发判断后决定。
