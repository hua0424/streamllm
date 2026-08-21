# REVISION_CHANGELOG — CISR 修订补充实验执行记录

## 2026-08-21 R2 终口径刷新完成（4b9587b）——论文数据全部定稿

- 主机重跑任务 1（中文 CER 去接缝空格终口径）：wer_real.csv 仅 12 行 aishell1 的 CER 两列变化，
  逐行回落到与同行 WER 一致（aishell1_clean streaming 0.1652→0.1180；非流式 0.1081→0.1077 几乎不动，
  侧面印证接缝空格假设）；librispeech 15 行与 ttft_real.csv 逐字节不变；带引号 glob 行为验证正确。
- `PAPER_WRITING_REFERENCE.md` §一数字已更新为终版、§十 标记"全部定稿"。
- **至此论文修改的数据准备全部完成**（Table III–VIII、Fig.6、§IV/§V 新小节、limitations、
  回复信证据链），无待刷新项。

## 2026-08-21 中文 CER 接缝空格口径修正（影响 Table VI/VII 的 zh CER 列）

- 发现：结果 JSON 的 `transcribed_text` 为 `" ".join(fragments)` 展示重构，接缝空格被中文 CER
  计为删除错误（E3 crosswoz 抽查：含空格 0.1040 vs 去空格 0.0692，抬高 ~3.5pt）。
- 修正：`score_wer_offline.score_pair` 中文分支 CER 前双方去空格（zh WER 经 zh_to_word_seq
  本就去空格，不受影响；英文不受影响；qa_transcribe 的 hypothesis 无拼接空格，不受影响）。
- 重出：E3 `wer_la_vs_b.csv`（LA/B/A 的 zh CER 0.0835/0.0732/0.0674，质量差回到真实小幅度）；
  R4 外部一致性 CER 0.0460→**0.0269**。R2 的 aishell1 各行 CER 待主机重跑任务 1 刷新
  （`OFFLINE_SCORING_HANDOFF.md` 三次追加，约 1 分钟）。
- 论文写作注意：Table VI/VII 的 zh CER 以新口径为准；若引旧数（如 streaming 0.1652）系接缝空格虚高。

## 2026-08-21 审查 r2 处置：P0 二次裁决=方案 (a)（E4 TTFT 口径），Table VIII 定稿

- r2 新证据独立核验成立：E4 TTFT − E5 post-flush 逐样本 50/50 为正（mean 410.5ms、min 139.0、
  max 1257.9）——方案2 的 1012.5ms 把静音窗内真实排水工作一并剔除，低于同物理量的 E4 直接实测
  （1476.0 vs 1065.6，差恰为排水项）；守恒分解 3065.1 = 53.1 + 1999.5 + 1012.5 成立。
- **需求方二次裁决=方案 (a)**：B 行 post 改用 E4 同 50 样本 streaming TTFT（ALL 1422.9，
  zh 1398.9 / en 1447.0）。装配脚本已修（self-test 5/5），Table VIII 定稿：
  **B ALL 14.79s（zh 15.58 / en 13.99）/ A ALL 22.67s**（差距 7.9s）。
  PAPER_HANDOFF §TTFA、r6_ttfa/RUNINFO 公式已同步为最终口径。
- 说明：此为本机推荐口径的纠错闭环——首轮方案2 的"或"并列选项把两个不等价子口径并置
  （审查方已在 r2 中自认首轮选项表述责任），开发按裁决字面实现无误；本轮证据化修正后，
  B 行 pipeline 分项回到直接实测，符合禁美化红线（修正方向使 B 数字变大）。

## 2026-08-21 论文数据就绪度审查（20260821-PAPER-DATA）处置：P0 口径裁决 + 三处文档对齐 + P1 修正

- 审查逐项独立复核全部通过（Table III/V 逐字一致、E1–E6 数字吻合、R4/R5/R6 产物吻合）。
- **P0（Table VIII 装置等待不对称）成立**：B 行 post 原含 ~2.0s 实时喂追加静音的装置等待
  （first_token−speech_end=3065ms 中，final 段入队在 +2053ms），A 行 ttft 从 audio_end 起算本就不含。
  **需求方裁决=方案2（对称剔除）**：B 行 post 改为 final 段入队→首 token（1012.5ms）。
  `assemble_ttfa_budget.py` 已修正（self-test 5/5），重装配：**B ALL 14.38s / A ALL 22.67s**（差距 8.3s）。
  PAPER_HANDOFF §TTFA 公式、r6_ttfa/RUNINFO 移交说明公式已对齐到同一口径（原文档写"2s 不进预算表"
  与旧装配结果矛盾，修正后该表述成立）。
- **P1-1** PAPER_HANDOFF E4 漂移表述更新为实测分布（mean 2.3 字符/p90 6/max 16、归一化 max 47.1%、
  涉及段 52.7%，含实词级漂移，不得再写"同音字/标点级"）。
- **P1-2** Table IV 498 清单重算值已核验与 table4_ablation_percentiles.csv 一致（Extra Long KV 增益
  40.82ms=3.3%、Very Long 2.73ms≈0 等），论文 §IV-B 连锁数字替换列入改稿清单（含样本量 108/150/240
  表述与排除规则）。
- **P1-3** LA 优势口径统一为"LA-2 基线比 System B 慢约 34%"（B 比 LA 低约 26%），PAPER_HANDOFF 已改。
- P2 八条写作注意事项登记备查（含 zh CER 脚注已补入 PAPER_HANDOFF E2 条目、speed 变体 medium 子组说明、
  R5 三轨用法约束、"near-zero"限定合成集、babble 归因链口径等）。
- 审查回复：`experiments/review/20260821-PAPER-DATA/reply-20260821-PAPER-DATA.md`。

## 2026-08-21 R2 重跑产物核验（8036780）+ glob 取样缺陷修复闭环

- 任务 1 重跑产物核验通过：wer_real.csv 30 行 / ttft_real.csv 102 行，14 个条件目录逐行齐全，
  数值合理（librispeech_clean A/B WER 2.97%/5.65%；babble streaming 43.7% 含 12 空转写，与 E2b 归因一致）。
- 主机指出的 glob 引号偏差属实：带引号单 glob 在脚本内展开成 14 目录后旧 `files[-1]` 只读最后一目录。
  已修复为**按目录分组取最新**（带不带引号行为一致），self-test 新增两条多目录取样断言（共 11 项全过）；
  E3 两 CSV 用修复后脚本重出，与已提交版本逐字节一致。
- 任务 2（CER 勘误）定性维持 e7ddbcb 登记结论（数字写法失配），无需主机侧进一步操作。

## 2026-08-21 R2 主机侧离线评分产物核验（d88c917）+ CER 勘误定性 + scope 口径修正

- 主机侧完成 OFFLINE_SCORING_HANDOFF 两任务。核验结论：
  1. 主机对 `qa_real_speech.py` 的 BOM 读取修复（utf-8-sig）正确；
  2. 英文 WER 交叉验证通过（librispeech 新旧 mean 均 0.0298，口径未变部分不受影响）；
  3. **中文 CER 勘误定性**：修正后 aishell1 mean 0.1073（旧污染口径 0.0672），超 10% 验收线，
     但逐样本定性为**数字写法失配**——参考文本中文数字（"百分之二十二点六"）vs Whisper 阿拉伯数字
     （"22.6%"），49/75 样本含此失配（其 mean CER 0.1476），**无失配的 26 样本 mean 仅 0.0313**；
     非构建错位（错位会 50%+），E2-0 构建 sanity 结论维持有效。Table VI 引用 zh CER 时需注明该口径因素。
- **scope 口径修正（本机发现）**：score_wer_offline 首轮按 sample_id 前缀分组，把 12 个变体并入
  干净集、丢失 Table VI 逐条件行；已改为 `--scope-by dir`（默认，按结果目录分组）/ `prefix`（E3 用）。
  E3 两 CSV 以 prefix 口径重出、与已提交版本逐字节一致；R2 两 CSV 待主机按交接文档重跑任务 1（约 1 分钟）。

## 2026-08-21 R5 语义一致性三轨完成（意见5 质量部分）

- 脚本 `semantic_consistency.py`（self-test 6/6）；产物 `r5_semantic/semantic_consistency.csv` + `.summary.txt` + judge/judge_solo 逐样本 JSON（审计可溯）。
- **轨道 A（bge-m3 嵌入余弦）**：mean **0.8832** / std 0.0755 / min 0.6232 / p10 0.7951（n=50）。
- **轨道 B1（成对等价 judge，DeepSeek deepseek-v4-flash，顺序随机化）**：mean 2.96/5，≥4 分 40.0%。
  判语分析：低分主因是两模式各自采样导致**推荐内容不同**（不同餐馆/景点）+ 128 token 截断造成
  "信息缺失"，属生成发散而非管线退化——成对等价口径无法分离这两者。
- **轨道 B2（独立意图满足盲评，2026-08-21 增设，分离采样噪声）**：A 3.10/5 vs B 3.04/5，
  **A−B 差 +0.06**（B≥4 分 26.0% vs A 30.0%）——独立口径下两模式下游意图满足度统计上不可区分，
  与轨道 A 高余弦互证。绝对分 ~3/5 反映截断上限对两臂同等影响。
- **论文表述建议**：以轨道 A + B2 为主证据（"嵌入余弦 0.88、独立意图满足差 0.06/5 不可区分"），
  B1 成对分如实报告并归因（采样发散 + 截断）；定性 case：高一致 crosswoz_8717_turn3
  （cosine 0.9784）、差异最大 crosswoz_7196_turn3（cosine 0.6232，judge 1 分但 B 独立分 4>A 的 2——
  成对低分不代表 B 退化）。
- 环境备注：bge-m3 经三次断流后 curl 断点续传下载（仓内权重为 pytorch_model.bin，torch 2.5.1
  拒载 .bin（CVE-2025-32434），已转 model.safetensors 存 C:/Users/hua/.cache/models/bge-m3）；
  DeepSeek 网关需非 python UA，judge max_tokens 提至 2048 后 3 条 reasoning 耗尽样本补齐（50/50）。

## 2026-08-21 本机离线分析批：分词接缝 / 提交分歧 / TTFA 预算装配 / Fig.6 重绘

- **分词接缝**（R4 §5.2，`check_tokenizer_seams.py`，self-test 7/7）：复现生产增量分词路径 vs 一次性分词。
  50 样本中 25（50.0%）存在接缝分歧，但**逐样本解码文本完全一致（50/50）**——分歧均为片段接缝处的
  BPE 跨缝重切（'.'+'Is'→'.Is' 型），分歧块 mean 4.36 处 / oneshot 侧 mean 5.60 token（max 12）。
  产物 `r4_commit/tokenizer_seams.csv` + `.summary.txt`。
  注：接缝无空格是生产真实行为（raw fragment 直拼）；论文表述应为"接缝分歧常见但仅为重切分、
  文本恒等，语义影响由 R5 证据兜底"，不得写"不匹配率极低"。
- **提交分歧完整统计**（R4 §5.3，`analyze_commit_divergence.py`，self-test 6/6）：
  375 commit / 224 correction（涉及段 224/425=52.7%，涉及样本 49/50）；
  correction 编辑距离 mean 2.3 字符、p90 6、max 16（归一化比率 mean 5.6%、max 47.1%）——
  **修正 PAPER_HANDOFF"属同音字/标点级"的过轻表述**：多数为小改但存在实词级漂移（如 "Inak."→". Enough!"），
  论文措辞以实测分布为准。外部一致性：streaming 拼接 vs System A 全量转写 WER mean 4.93%（max 14.2%）。
  产物 `r4_commit/commit_divergence.json`（含 top-5 漂移示例，供定性分析）。
- **TTFA 预算装配**（R6 §7.3，`assemble_ttfa_budget.py`，self-test 5/5；口径=用户裁决的 E5 链条）：
  System B TTFA mean **zh 17.2s / en 15.5s / ALL 16.4s**（端点 53ms + 后端 3012ms + 解码首句 389ms
  + TTFC zh 13.99s/en 11.86s，四项全实测）；System A ALL 22.7s（pipeline 实测，decode/TTFC 为估计项，
  CSV source 列已标注）。产物 `r6_ttfa/ttfa_budget.csv`。
- **Fig.6 重绘**（R7 §8.1，`plot_fig6_trend.py`，self-test 4/4）：exp1 归档逐样本数据，12 等频分箱，
  mean 折线 + P5–P95 阴影带。产物 `results/revision/fig/Fig6.pdf/.png/.bins.csv`。

## 2026-08-21 离线评分脚本 score_wer_offline 完成（E3 已出数）+ 中文 CER 口径勘误

- 产物：`experiments/scripts/score_wer_offline.py`（self-test 9/9）；E3 三系统已出数
  `r3_baseline_la/{wer,ttft}_la_vs_b.csv`（498 样本：LA WER 0.1073 / B 0.1047 / A 0.0951，
  同引擎同量级，支撑"同等质量下 B TTFT 优约 34%"）；英文统一大小写折叠（计划 §3.4），
  与 la_results 内联无折叠口径（LA 0.130）差异源于 multiwoz 混合大小写，以离线折叠版为准。
- **勘误（指标口径，非数据问题）**：`qa_real_speech.py` 原中文分支误用
  `cer(zh_to_word_seq(...), zh_to_word_seq(...))`，逐字空格污染 CER 分母——E2-0 登记的
  "aishell1 CER 6.72%" 口径有误；exp3 原生口径为 `cer(ref, hyp)` 直接吃原文
  （`run_exp_quality.py:606`）。已修正脚本并加 `--recompute-from-csv` 免模型重算模式；
  修正后数值待主机侧重算（`r2_real_speech/OFFLINE_SCORING_HANDOFF.md` 任务 2），若超 10% 验收线另行上报。
- R2 的 wer_real/ttft_real 需主机侧执行（样本 JSON 只在主机），交接同上文档任务 1。

## 2026-08-21 TTFA 预算缺项补测脚本完成（待审查后 GPU 主机执行）

- 起因：E1–E6 验收完成后本机离线核对产物，发现计划 §7.3 预算表分项 `T_decode_to_first_sentence`
  无法从既有产物恢复（E4 未记录逐 token 时刻，`generate()` 无逐 token 日志）。
- 产物：`experiments/scripts/measure_decode_to_first_sentence.py`（独立 LLM 解码段测量，
  输入 E4 的 50 条 streaming transcribed_text，生产同款 cache_prompt+generate 路径，
  逐 token 计时 + 句末标点检测含小数豁免）；`--self-test` 本机 12/12 断言通过。
- 前因后果与方法学论证：`r3_baseline_la/handoff/E6_TTFA_DECODE_FIRST_SENTENCE_HANDOFF.md`（供审查）。
- 待办：审查确认 → GPU 主机冒烟（3 条）→ 正式 50 条（约 0.5 GPU 小时）→ 本机装配 ttfa_budget.csv。

## 2026-08-19 E1 重复测量（R1.2，意见3）3 轮全部完成

- 命令：`uv run python -m experiments.scripts.run_exp_latency --dataset all --sample-list $REV/r1_stats/repeat_subset_ids.json --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 --output-dir $REV/r1_stats/repeat_r{1,2,3} --no-resume`（逐轮串行）
- 产物：`r1_stats/repeat_r{1,2,3}/`（各含 exp1_results/exp1_summary/exp1_statistics 三件套 + checkpoint + run.log + RUNINFO.md）
- 关键数字：每轮 50 样本 × 2 模式 = 100 条，error 0；streaming TTFT mean = 1434.0 / 1482.1 / 1454.9 ms（三轮极差/mean = 3.3%）；non-streaming = 4063.3 / 4244.2 / 4118.5 ms；逐样本跨轮 CV：mean 4.2%、median 3.3%、p90 8.8%；config 块三轮均与 §1.3 锁定表一致（turbo / Qwen2-7B / cuda:0 / cuda:1 / suffix=0 / max_tokens=50）
- 异常与处理：**绝对水平偏差待需求方裁决**——repeat 子集 50 条全为 very_long 组，本机 System B mean ≈1.43–1.48s，超出 §5.2 sanity 带 0.9–1.3s；同样本在原实验机（exp1_latency 历史结果）为 981ms。已核对非配置错误（锁定表逐项一致、0 error、双卡分工正确），初步判断为机器级差异（本机 Xeon Gold 6133 @2.50GHz 低主频服务器 CPU，流式管线 CPU 敏感）。同机 A/B 对比内部有效性不受影响。E2–E5 是否继续，等待需求方决定（2026-08-19 已上报）。
- 备注：round 3 前 ~10 样本曾与 E2-0 aishell1 构建短暂并行（后发现即中止构建链），round 3 mean 落于 r1/r2 之间，未见可见污染；如需求方要求可干净重跑 round 3（约 49 GPU 分钟）。

## 2026-08-19 E2-0 真实语料下载完成

- 命令：`wget -c` 从 openslr 下载 test-clean / test-other / data_aishell / musan（共约 26 GB），`tar -xzf musan.tar.gz`
- 产物：`experiments/datasets/raw_data/`（librispeech 645MB、aishell1 15GB、musan.tar.gz 11GB + musan/ 解压目录）
- 关键数字：4 个包全部 exit=0，总耗时约 95 min（~5MB/s）
- 异常与处理：无

## 2026-08-19 E2-0 librispeech 干净集构建完成（aishell1 及后续阶段进行中）

- 命令：`uv run python -m experiments.scripts.build_real_speech_set --source librispeech`
- 产物：`experiments/datasets/processed/{json,audio}/librispeech/` + `r2_real_speech/librispeech_build_manifest.json`
- 关键数字：75 条（long 30 / very_long 30 / extra_long 15），总时长 44.6 min，配额与验收线一致
- 异常与处理：构建链曾为保护 E1 round 3 的 CPU 环境主动暂停，E1 完成后从 aishell1 阶段续跑

## 2026-08-19 E2-0 aishell1 构建 + 全部增强变体 + 变体静态 QA 完成

- 命令：`build_real_speech_set --source aishell1`；`build_augmented_variants --dataset {librispeech,aishell1} --variants snr20 snr15 snr10 speed09 speed11 babble`；`qa_real_speech --datasets <12 变体>`
- 产物：`processed/{json,audio}/aishell1`（75 条）+ 12 个变体目录（各 30 条，含可选 babble）+ `r2_real_speech/{aishell1_build_manifest,{librispeech,aishell1}_augment_manifest}.json` + `qa_static.csv`
- 关键数字：aishell1 配额 30/30/15 达标（总时长 40.8 min）；变体静态 QA 全部通过（exit=0）
- 异常与处理：无

## 2026-08-19 E2-0 转写 QA 通过（E2-0 全部验收线达成）

- 命令：`uv run python -m experiments.scripts.qa_real_speech --datasets librispeech,aishell1 --transcribe --asr-model-size turbo --device cuda:0`
- 产物：`r2_real_speech/qa_transcribe.csv` + `qa_transcribe.log`
- 关键数字：librispeech mean WER = 2.98%（≤10% ✓，max 14.7%）；aishell1 mean CER = 6.72%（≤10% ✓，max 33.3%）；退出码 0。至此 E2-0 验收线全部达成（配额 30/30/15 ✓、变体各 30 条 ✓、静态 QA ✓、转写 sanity ✓、manifest 齐全 ✓）
- 异常与处理：无

## 2026-08-19 E1 sanity 偏差归因完成：非代码改动，定位于 CPU 单线程机器差异

- 方法：同机三版本代码 A/B（V0=2f9f481 实验期 / V1=13dfcf7 pre-DEV / 当前 DEV 代码）+ uv.lock 版本比对 + 50 共同样本分量归因 + CPU/虚拟化环境检查
- 产物：`r1_stats/attribution/`（README.md 完整证据链 + v0/v1 结果 JSON 与日志）
- 关键数字：三版本代码本机 TTFT 互差 <10%（代码无回归）；软件栈版本逐一致；LLM 预填两机 +9%（GPU 一致），ASR 尾处理 +50%（+445ms，占全部增量 98%）；本机为 KVM 虚拟机、Xeon Gold 6133 @2.50GHz 固定无 turbo
- 异常与处理：结论——非 DEV 代码导致，为机器级 CPU 单线程差异；同机 A/B 对比有效性不受影响。E2–E5 是否继续待需求方裁决
- 附带修正：env_versions.txt 误采系统 anaconda 环境（`uv run pip list` 指向 conda pip），已用 `uv pip list --python .venv/bin/python` 重采项目 venv 真实版本

## 2026-08-19 需求方裁决：E3 补跑 System A/B，禁缩放红线，sanity 带更新，继续 E2–E5

- 裁决（需求方原话要点）：
  1. **必须重做（唯一）**：E3 的 System A/B 在本机按 498 清单重跑，与 LA 同机三方对比（跨机污染且偏向作者有利方向，必须堵）；
  2. 不重做：exp1/exp2 全量、E2、E4/E5（本就同机内部有效；exp1 不重做的硬证据=改进率 65.6%→64.7% 稳定不变）；
  3. **红线**：不对本机数字做任何缩放去对齐旧机绝对值；
  4. 论文侧落实：A 披露覆盖 Table IV、B 正式裁决禁止缩放、C 更新 sanity 带、D 摘要绝对数绑定平台。
- 本机 QA sanity 带调整（落实 C，仅作异常检测、不用于数据修正）：System B long+ mean 带 0.9–1.3s → **1.2–1.7s**（以 E1 实测 very_long 1.43–1.48s 为锚 ±15%）；System A extra_long 6–8s → **7.5–10.5s**（按 System A 实测 +28% 偏移等比放宽）。所有数字一律如实记录，不做对齐。
- 异常与处理：无

## 2026-08-19 E2a 真实语音干净集运行完成（R2，意见1）

- 命令：`uv run python -m experiments.scripts.run_exp_latency --dataset {librispeech,aishell1} --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 --output-dir $REV/r2_real_speech/{librispeech,aishell1}_clean`
- 产物：`r2_real_speech/{librispeech,aishell1}_clean/`（三件套 + checkpoint + run.log + RUNINFO.md）
- 关键数字：各 75 样本 × 2 模式，error 0；librispeech streaming TTFT mean long/very_long/extra_long = 1773/1628/1559ms vs non-streaming 1655/2957/4805ms（extra_long 改善 67.5%）；aishell1 = 1659/1707/1763ms vs 1627/2982/5140ms（extra_long 改善 65.7%）；改善率与原论文 65.6% 同量级 ✓；config 块与锁定表一致 ✓
- 异常与处理：无（System B long+ 1.56–1.77s 落在更新后本机 sanity 带 1.2–1.7s 边界附近，属机器级偏移预期范围；System A 数值随真实语音时长等比，未见异常）

## 2026-08-19 E2b 增强变体 12 组全部运行完成（R2，意见1）

- 命令：`for v in <12 变体>; do uv run python -m experiments.scripts.run_exp_latency --dataset $v --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 --output-dir $REV/r2_real_speech/$v; done`
- 产物：`r2_real_speech/{librispeech,aishell1}_{snr20,snr15,snr10,speed09,speed11,babble}/`（三件套 + run.log + RUNINFO.md ×12）
- 关键数字：每组 30 样本 × 2 模式，error 全 0，config 全 ✓；snr/speed 十组 streaming 改善率 +22.8%~+31.6%；**babble 两组显著退化：librispeech_babble streaming mean 3425ms vs non-streaming 2326ms（−47.3%），aishell1_babble +6.5%**
- 异常与处理：babble 退化已归因，判定为真实管线行为而非缺陷——VAD 对多人 babble 噪声过度触发（段积压，最慢样本 audio_end 后 21s 才排完 ASR 队列），whisper 对噪声段大量空输出（librispeech streaming 空转写 12/30、aishell1 5/30，no_speech 门控）；分布为长尾（median 2020ms vs mean 3425ms，11/30 >3s）。该结果对作者不利方向，如实登记（禁缩放红线）。**下游分析注意**：空输出样本的 `asr_time`/`llm_prefill_time` 为哨兵值（last_text_time=0 导致的负/正纪元数），统计这两个字段时应剔除零提交样本；`ttft` 不受影响。

## 2026-08-20 E3-AB System A/B 本机重跑完成（R3 前置，需求方裁决项）

- 命令：`uv run python -m experiments.scripts.run_exp_latency --dataset all --sample-list $REV/r3_baseline_la/exp2_ablation_sample_list.json --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 --output-dir $REV/r3_baseline_la/system_ab_rerun`
- 产物：`r3_baseline_la/system_ab_rerun/`（三件套 + checkpoint + run.log + RUNINFO.md）
- 关键数字：498 样本 × 2 模式 = 996 条，error 0；streaming TTFT mean 1573.9ms（long/very_long/extra_long = 1464/1551/1638ms，均在更新后带 1.2–1.7s 内 ✓）；non-streaming = 5310.8ms（三组 1958/3906/7698ms，extra_long 落在更新后带 7.5–10.5s 内 ✓）；config 锁定 ✓
- 异常与处理：无

## 2026-08-20 E3-LA 完成但 QA 发现 DEV-3 实现 bug：结果标记无效，待修复重跑

- 命令：`uv run python -m experiments.scripts.run_exp_baseline_la --dataset all --sample-list $REV/r3_baseline_la/exp2_ablation_sample_list.json --asr-device cuda:0 --llm-device cuda:1 --output-dir $REV/r3_baseline_la`
- 运行本身：498/498 完成，error 0，exit=0，无死锁
- QA 异常：LA WER mean=0.545（异常高），79% 样本转写长度 < System B 的 70%
- 归因（逐轮重放 crosswoz_10296_turn2 实证）：`_trim_buffer()` 裁剪 buffer 后，`n_committed`/`prev_words` 仍停留在裁剪前"全序列帧"，下一轮假设是裁剪后"尾段帧"的词序列——下标错帧使提交跳过新假设前 n_committed 个词（实证：提交1='你好…住宿'，提交2 直接跳到 '600元有什么合适的吗?'，中间'我希望酒店的最低价格是500到'被静默丢弃），flush 同样丢尾。
- 处置：结果 JSON 重命名为 `la_results_*.json.INVALID_dev3_frame_bug` 保留现场；按"发现缺漏停止上报，不自行修改实现"规则，已上报需求方待授权修复（建议：裁剪后重置 prev_words/n_committed 参考帧，提交条件改为时间下限 end > last_committed_end 叠加尾随保护）；修复后需重跑 E3-LA（约 6h）。
- 注：E4/E5 走 System B 路径、与 LA 组件无关，经评估不受影响，继续按计划推进。


## 2026-08-20 E4 插桩+完整回复复跑完成（R4+R5，意见5）

- 命令：`uv run python -m experiments.scripts.run_exp_latency --dataset all --sample-list $REV/r1_stats/repeat_subset_ids.json --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 --max-tokens 128 --save-full-response --save-fragments --output-dir $REV/r4_commit --no-resume`
- 产物：`r4_commit/`（三件套 + commit_log.jsonl + checkpoint + run.log + RUNINFO.md）
- 关键数字：50 样本 × 2 模式，error 0；full_response 50/50 非空（mean 208 字符）；committed_fragments 50/50；commit_log.jsonl 599 行（375 commit + 224 correction）覆盖 50/50；streaming TTFT mean 1423ms（与 E1 同口径一致）
- 异常与处理：无。备注：turbo 下 50 样本观测到 224 条 correction（已提交段重识别漂移），供分词接缝/语义一致性分析；结果 JSON config 块未含 save_full_response/save_fragments 开关位（在 checkpoint config 中有），属记录完整性小瑕疵，不影响数据。

## 2026-08-20 E5 端点等待测量完成（R6.1，意见2）

- 命令：`uv run python -m experiments.scripts.run_exp_latency --dataset all --sample-list $REV/r1_stats/repeat_subset_ids.json --asr-device cuda:0 --llm-device cuda:1 --suffix-segments 0 --append-silence-ms 2000 --output-dir $REV/r6_ttfa/endpoint --no-resume`
- 产物：`r6_ttfa/endpoint/`（三件套 + checkpoint + run.log + RUNINFO.md）
- 关键数字：50 样本 × 2 模式，error 0；QA 四项全过：(a) audio_end−speech_end = 2.002s±0.001 ✓；(b) asr_no_speech 0 个 ✓；(c) final_speech ≤ final_is_final 无违例 ✓；(d) 无异常样本。端点指标（50/50 有效）：endpoint_detection_wait mean 53ms / median 109ms / p90 208ms；final_enqueue_wait mean 2053ms；post_endpoint_ttft mean 3012ms；total（speech_end→首 token）mean 3065ms
- 异常与处理：无

## 2026-08-20 DEV-3 修复完成（commit `6d74c1c`）：错帧 + 裁剪幻听双重修复，评审门槛中本机侧全部达成

- 评审：`experiments/review/20260820-E3LA/review-20260820-E3LA.md` 判定 bug 定位成立（P0）、结果无效处置正确、修复方向原则上通过，要求补齐跨帧/无重复/边界/flush/空识别/生产路径回归与真实样本回放后方可重跑。
- 修复内容（`src/asr/local_agreement_streamer.py` 重写状态机）：
  1. **错帧（原 bug）**：提交状态改为绝对音频时间轴（`committed_words`/`committed_end_abs`/`buffer_start_abs`），提交判定用时间下界 `end > committed_end_abs + eps` 而非跨帧词数下标；`prev_words` 保留未提交尾部作下一轮比较基线（两轮确认不削弱）。
  2. **裁剪幻听（修复过程中新发现，同一样本本机回放暴露）**：原"末词 end−0.1s"裁剪把缓冲切成以句末残片（'宿。'）开头，Whisper turbo 对此坍缩为训练集水印幻听（'请不吝点赞订阅转发打赏支持明镜与点点栏目'）且连续三轮文本稳定一致，绕过 LA-2 两轮确认。实测：句末残片开头→幻听；干净句首/句中词边界切开→正常。修复：裁剪与提交解耦，优先裁到最后一个句末已提交词的 end（无回退），无句界锚点且缓冲超 15s（ufal buffer_trimming_sec 对齐）才强制裁到已提交边界。
  3. **标点抖动卡死（同场暴露）**：Whisper 对同一音频的标点附着/分词跨轮不稳（'宿。'↔'宿'+'，'），逐字 LCP 会永远停在原地。修复：一致比较只取实质词、按去首尾标点规范化文本比较；纯标点词透明（不作锚点，提交时随区间带出）。
- 回归：`test_revision_regressions` **16/16**（A1-A3 按新语义重写 + 新增 D1 跨帧跳段 / D2 强制裁剪多周期无重复 / D3 边界 ±0.1s 不重复不跳过 / D4 flush 幂等 / D5 生产路径 run_single_sample 全链路无缺口且 LLM 收到文本==提交文本 / D6 标点抖动不卡死）。
- 真实样本回放（turbo，本机）：`crosswoz_10296_turn2` 修复前 WER 0.8796、24 字符、中段丢失；修复后 **WER 0.0185 / CER 0.0727**、116 字符、5 次提交覆盖全文（证据 `r3_baseline_la/handoff/replay_crosswoz_10296_turn2_fixed.json`）。
- 附带硬化：`run_exp_baseline_la` 清单缺失样本改为硬失败（评审要求"停止而不是静默缩减"）；`la_max_buffer_s` 入结果 config 块。
- 待办（GPU 机侧）：E0 冒烟 `--max-samples 2` → 隔离旧 checkpoint/INVALID 现场 → 全量重跑 E3-LA（命令与 QA 清单见 GPU_EXPERIMENT_HANDOFF §E3"2026-08-20 重跑前置"）。

## 2026-08-20 E6 TTS 首包延迟测量完成（R6.2，意见2）

- 命令：`uv run python -m experiments.scripts.measure_tts_first_chunk --from-e4 $REV/r4_commit --n-zh 25 --n-en 25 --url http://127.0.0.1:20401 --output $REV/r6_ttfa/tts_first_chunk.csv`
- 产物：`r6_ttfa/tts_first_chunk.csv`（50 条，error 0）+ 自动 `.runinfo.md`；冷启动对照轮 `tts_first_chunk_run1_cold.csv`（9 条，手动中止归档）
- 关键数字：zh(n=25) TTFC mean 13.99s / en(n=25) 11.86s；total mean 32.8–34.5s；RTF mean 0.71–0.72
- 异常与处理：首轮冷启动 TTFC 偏高（mean 14.84s），按需求方建议做 3 次预热后重测——**冷热两轮无显著差异（14.84s vs 12.92s，RTF 0.74 vs 0.71），排除预热因素，确认为稳态行为**。
- 附加诊断（TTFC×文本长度，单次测量）：3 字符→1.37s / 17 字符→3.77s / 45 字符→7.64s / 200 字符→17.93s —— **TTFC 与文本长度近似线性**，该部署为句段级流式（句内不流式：短句首包≈全程合成完毕才到）。
- 环境注记：TTS 前端文本处理 CPU 敏感，本机 KVM Xeon Gold 6133@2.5GHz 会放大 TTFC/RTF；按裁决 D 绑定平台披露。spk_id 晓伊→中文女 别名补丁仅影响音色，不影响延迟指标（见 TTS_SERVICE_HANDOFF §三）。


## 2026-08-21 E3-LA 修复后全量重跑完成，评审 R2 七项 QA 全过（R3，意见4）

- 命令：`uv run python -m experiments.scripts.run_exp_baseline_la --dataset all --sample-list $REV/r3_baseline_la/exp2_ablation_sample_list.json --asr-device cuda:0 --llm-device cuda:1 --output-dir $REV/r3_baseline_la --no-resume`（代码 c965240，含修复 6d74c1c）
- 产物：`r3_baseline_la/la_results/la_summary/la_statistics` 三件套 + la_run.log + RUNINFO.md（七项 QA 结论）；E0 冒烟在 `r3_baseline_la/e0_smoke/`；旧无效现场在 `invalid_dev3_frame_bug/`
- 关键数字：498/498 error 0；WER mean 0.130 / CER 0.118（修复前 0.545）；LA/SysB 长度比 0.98/0.99；divergence mean 1.0 max 7；LA TTFT mean 2115ms vs System B 1574ms（LA 全缓冲重解码开销，量级可解释）；回放抽查 3 样本无跳段无重复
- 异常与处理：无。E3 三方同机对比齐备：System A 5310.8 / System B 1573.9 / LA 2115.0（mean TTFT ms）


## 2026-08-21 PRE-PAPER-AUDIT 整改：方案 v3.1 冻结 + W1/W3/W4/W5 实现（总册降级整改中）

- 背景：撰稿前审计（`experiments/review/20260821-PRE-PAPER-AUDIT/`）5 项 P0 全部经本机独立核验属实
  （TTFA 跨运行装配非闭合、E5 端点 8/50 负值、0.09s/字符不被正式 50 条支持[slope=−13.8ms/字符,R²=0.017]、
  CV 口径 ddof 依赖、WER 宏平均未标注）；方案经三轮复审冻结为 v3.1（Gate 0 口径统一 + Gate 1 实现细则）。
- **新发现**：`_decode_logits` 的 `repetition_penalty` 为死参数从未生效（`stream_llm_inference.py`）——
  全部历史生成的实际采样为 temperature=0.1/top_p=0.9/无重复惩罚；E6 changelog 的"TTFC×长度近似线性"
  来自单次长度扫描（3→200 字符），与正式 50 条（99–260 字符段内无相关性）不矛盾，0.09 系该扫描斜率。
- W3（本机）：`recompute_cv_stats.py` → `r1_stats/repeat_cv_summary.csv/md`（ddof=1：
  B 5.19/4.05/10.73/18.96%、19/50>5%；A 5.23/4.65/9.92/14.01%、23/50>5%），与审计锚点逐位一致。
- W4（本机）：`score_wer_offline.py` 扩展 corpus 口径（逐样本 WER/CER 各 S/D/I/N，DP 回溯与
  _levenshtein 距离断言一致；corpus=Σ(S+D+I)/ΣN）；`wer_real.csv`/`wer_la_vs_b.csv` 重生成，
  宏平均列与定稿版逐字节一致（0 行差异）。**本机无 R2 样本 JSON，新增 --ref-csv 用
  qa_transcribe.corrected.csv 的 reference_full 列（reference 为截断展示版，误用会把
  aishell1_clean ns CER 0.1077 抬到 0.2009，已核实规避）**。
- W5（本机）：`paired_inference.py` → `stats_inference/paired_inference.csv/md`（21 比较；
  bootstrap 10k seed=20260821 percentile CI；Wilcoxon 双侧 wilcox/auto/无校正；Holm 族：
  Table III 三分组、R2 十二增强条件）。锚点核对：table3 extra_long 差 5657.9ms、
  改善率 34.6%/65.6%/83.9%、table7 A/B 70.4%、B/LA 25.6%（CI [485.3,599.9]ms,p=1.2e-70）、
  R5 B−A=−0.06 CI [−0.34,0.22] 跨 0。
- W1（待 GPU）：`run_ttfa_unified.py` 统一时间轴 TTFA 实测脚本完成（PSE 双法裁决+固定 Silero ref、
  因果回放 planned/actual、A 等待 feed_end、无条件 INPUT_CLOSED sentinel[修复 flush=None 死锁=
  历史 4 条挂起样本机制]、流式句末 lookahead、PCM 512B/1324B playable、配对 seed、fail-closed
  checkpoint），self-test 33 项全过；`stream_llm_inference.py` 新增 generate_with_meta（旧接口不动）。
- W6/W7/W9 文档：`r5_semantic/REPRO_METADATA.md`、`r2_real_speech/MANUAL_SPOT_CHECK.md`
  （试听待人完成）、`r3_baseline_la/LA_METHOD_AND_EXCLUSION.md`。
- W8 阶段 1：`PAPER_WRITING_REFERENCE.md` §十降级"整改中"，五处作废表述行内标记。
- 下一步：代码级审查（Gate 1）→ GPU handoff（TTS 探活→冒烟 3 条→正式 50×2+子集补轮+匹配文本控制+W2 环境）。

## 2026-08-21 Gate1 实现审查整改（review-implementation-v3.1 全量接纳）

- P0-1 flush=None 显式 final drain（drain 无输出记 error，不静默截断）+ 真实 ASRCache 协议测试；
- P0-2 pair 全程绝对 deadline + 线程终止确认 + fatal fail-stop（fatal 后任务补 cancelled 终态）；
- P0-3 checkpoint 整文件原子快照 + 完整恢复绑定（git/env/模型/Silero/TTS/清单 hash），四负向用例；
- P0-4 正式模式强制固定 Silero（ref/dir + artifact SHA-256，缺失拒启动）；
- P1-1 EOS 优先于 first_model_token；P1-2 请求级 torch.Generator（multinomial 隔离）；
  P1-3 语言×时长分层调度（stratum ≤1，全局 25/25）；P1-4/5 TTS 跨 read 格式校验+重切粒度+
  读到自然结束+对齐校验；P1-6 schema/QA 扩项；P1-7 冒烟分层选取+可控故障注入（仅 smoke）；
- W3 键集合/config 一致性；W4 配对交集过滤+paired_filter_manifest+reference_full 强制
  （重生成 0 差异）；W5 重复/空配对/LA 模式限定/R5 唯一性（重跑数字不变）；
- self-test：W1 56 项全过；W3/W4/W5 全过；新增本机真实组件集成测试
  `ttfa_local_integration.py`（3060 + whisper-tiny + Qwen2-0.5B 本地目录）ALL PASS（仅路径验证）；
- 模型/设备改 CLI（默认取 src/config）；EXPERIMENT_DESIGN.md 同步排结果 QA 后。

## 2026-08-21 Gate1 r2 复核整改（review-reply-implementation-v3.1-r2 全量接纳）

- P0-1 final-drain 竞态：state_lock 内先 final 化后发布 close；真实 ASRCache + 放慢转写
  的确定性交错回归×3；
- P0-2 双 runner 主线程 ASR/LLM 异常无条件 fatal；`_backfill_cancelled()` 可测回填；
- P0-3 Checkpoint.fatal_seen 恢复 run 级 fail-stop（剩余任务只补 cancelled）；
- P0-4 StreamAudioSegmenter 支持注入固定 silero_model/utils（注入不触 hub）；W1 断言
  PSE 与分段器同一 artifact；RUNINFO 双侧 meta；
- P0-5 `_select_smoke()` 精确命中/双语种/N×2 校验；QA 断言成功+故障路径均执行；
- P1：TTS 派生字段一致性校验+负向×3；resp_holder+动态 read timeout+外层主动 close+
  headers-only 慢流测试；集成测试表述降级并 CLI 化；self-test 显式计数；
- self-test 69 PASS / 0 FAIL；W3-W5 不变全过；集成检查复跑 ALL PASS；
  .gitattributes 显式 CSV whitespace 规则。

## 2026-08-21 Gate1 r2 复核通过：探活+3条冒烟放行，GPU 冒烟 handoff 交付

- 审查 `review-implementation-r2-20260821.md`：W3/W4/W5/独立探活/3 条冒烟（有条件）放行，
  正式 50×2 仍待冒烟结果级复核；
- 剩余条件登记：TTS total deadline 内部主动取消仅在 chunk 间生效（headers-后停发场景由
  动态 read timeout/pair deadline 兜底，冒烟需保存慢流证据）；本机 self-test 不替代 GPU
  真实路径验证；
- 交付 `r7_ttfa_unified/R7_GPU_SMOKE_HANDOFF.md`：固定 Silero 目录确定（--silero-dir 优先）→
  探活 → self-test 69 项 → smoke 3（2 zh+1 en，含 asr_error 注入）→ 8 项验收清单 →
  W2 环境记录 → 产物提交与禁止事项。

## 2026-08-21 TTS 客户端契约错配修复（GPU 冒烟现场反馈，两处回归）

- 现场（R7_TTS_CLIENT_CONTRACT_BUG_HANDOFF.md）：探活 404——根路径 + json= 编码与真实
  CosyVoice 契约（POST /inference_sft + form）不符；对照 E6 measure_tts_first_chunk.py 核实属实；
- 修复：_tts_endpoint() 幂等拼后缀 + _tts_form_body() 全字符串 form 编码（探活/正式共用）；
  假 TTS 服务改契约严格路由（错路径 404/非 form 422）堵住自测覆盖缺口；
- self-test 69 → 75 PASS / 0 FAIL；3060 真实组件集成检查复跑 ALL PASS；
- 次要定夺：Content-Type 缺头按 None 原样固定为允许策略（probe 加 policy_note）；Silero
  缓存非 git checkout 时 repo_commit=None+注记，锁定依据 artifact sha256；
- 交付 R7_GPU_SMOKE_HANDOFF_R2.md（执行版 handoff）+ 现场回复函。

## 2026-08-21 PSE Silero 签名错配修复（GPU 冒烟现场反馈第二轮）

- 现场（R7_PSE_SILERO_SIGNATURE_BUG_HANDOFF.md）：任务 1/2 过，任务 3 PSE 预扫描
  fail-closed——silero_pse_sample 漏传真实 get_speech_timestamps 的必填位置参数 model，
  单算法失败拦停；现场已闭环验证补 model 后 n_segments=14、last_end 与能量法一致；
- 修复：model 位置参数透传；analyze_pse 缺 model 显式拒止（pse_missing_model）；
  self-test 假 Silero 改签名严格（漏传即断言失败）+ 缺 model 用例；本地集成检查加
  "契约错误不许能量法兜底掩盖"防护（该兜底此前掩盖了本 bug）；
- self-test 75 → 76 PASS / 0 FAIL；集成检查复跑 ALL PASS（真实 Silero 路径实际跑通）；
- 现场产物收妥：tts_probe.json（ok/pcm，任务 1 免重跑）、env/cpu_gpu.txt、pip_freeze.txt；
- 交付 R7_GPU_SMOKE_HANDOFF_R3.md（任务 2 期望 76；任务 3 命令不变）+ 现场回复函。

## 2026-08-21 TTS 裸 PCM 误判 JSON/HTML 修复（GPU 冒烟现场反馈第三轮）

- 现场（R7_TTS_PCM_JSON_MISCLASSIFY_BUG_HANDOFF.md）：任务 3 冒烟在正式路径探活被
  fail-closed——裸 PCM 首字节恰为 0x7b('{') 被 classify_payload 单字节判 json；
  200 次实测 198 pcm/1 json/1 html（首字节均匀），证实概率性误判，探活与正式请求共用
  分类器 → 不修则正式 50×2 每条 TTS 请求 ~1.6% 假失败；
- 修复：JSON 须整段前缀严格 json.loads 通过；HTML 须 <!doctype/<html 特征；
  Content-Type 显式声明 json/html/xml 以头为准；WAV 保持 RIFF。残余风险（无头且长于
  前缀的截断 JSON 判 pcm）已登记，真实服务不构成该场景；
- 新增 10 项回归（含现场首样本 (-133,-68,-108,-119) 还原与 measure 级 0x7b 首块流）；
  self-test 76 → 86 PASS / 0 FAIL；集成检查复跑 ALL PASS；
- 交付 R7_GPU_SMOKE_HANDOFF_R4.md（任务 2 期望 86；任务 3 命令不变）+ 现场回复函。

## 2026-08-21 R7 冒烟通过 + 本机结果级核验（GPU 主机 commit b1e1206）

- 冒烟产物：QA 0 问题 / 6 记录（5 成功 + 1 asr_error 注入，fatal 语义正确）；
  B TTFA zh 2555–2610ms / en 3045ms；A 22.3–23.4s；A−B 差 19.6–20.4s（全文 TTS 策略主导）；
- 本机独立核验（dev-smoke-verification-20260821.md）：八项验收逐项过——validate 重跑 0
  违规、闭合残差 0、无 final_drain_empty=True/thread_leak/pair_timeout、配对 seed 一致、
  双语双模式齐、Silero artifact e1122837… 双侧一致、探活 ok/pcm；
- 首扫误报登记：核验脚本曾把字段名 final_drain_empty 当命中，修正后 0 问题；
- -lcuda 链接噪声：ctranslate2 JIT 探测 32 位 libcuda，非门禁，登记沿用；
- 待审查复核通过后发正式实验 handoff（50×2 + 子集补轮 + 匹配文本控制）。

## 2026-08-22 冒烟复核意见（provenance Gate）全量采纳落实

- 五项阻塞全部成立并处置：dirty 树→Gate 版 handoff G1（clean 前置）；commit 混淆→核验报告
  addendum 拆分 code(1a0ddc8)/artifact(b1e1206)/verification(cdeb927) 三元；TTS 服务
  provenance→handoff G7 采集命令（commit+diff/镜像 digest/模型/spk2info hash）；
  --tts-control-only→已实现（10 配对×3 调用+中英校准句=32 调用，独立 binding 含
  control-from hash，配对不足 fail-closed）+4 项 self-test；平台条件→
  --platform-conditions-file hash 入 config/binding；
- 晓伊→内置中文女映射固化为 SPEAKER_MAPPING_NOTE 常量，自动入 RUNINFO/binding（G10）；
- self-test 86 → 90 PASS / 0 FAIL；正式 handoff 重写为 Gate 版（G1-G8 + 加严验收）；
- 待审查方复核回复函后出具书面放行记录（Gate 第 11 项）。

## 2026-08-22 冒烟复核二轮意见（Gate 未完成项）全量采纳

- 两处真代码缺口修复：run_tts_control 显式计算并绑定 platform_conditions_sha256
  （审查 §4 指出的 control 读取路径）；新增 --inject-fault-index（默认末位，非末位可指定）
  支撑"非末位 fatal → cancelled"运行级证据；
- handoff（Gate 版）新增 2b 非末位 fatal 小 smoke（r7_smoke_fatal，验收=1 success+1 fault
  error+fatal+4 cancelled_after_fatal，独立 run 不入正式）与 2c self-test 归档
  （GPU clean 树复跑），并明确顺序：G1-G8→2b→2c→书面放行→r7_main；
- 本机 self-test 不可变归档生成：selftest_archive/selftest_20260822.md/.log
  （命令/exit 0/HEAD/环境/90 项输出/输出 sha256）；
- self-test 90 PASS / 0 FAIL 复跑一致；§3.1-3.3 属现场采集项，已固化 handoff 待执行。

## 2026-08-22 终裁整改：handoff 流程循环修复（Gate 版 r2）

- 审查终裁确认实现级全过；8 项缺失产物中除流程循环外全部为 GPU 现场执行项；
- 修复循环：handoff 头部明确执行权限划分——G1-G8 采集/2b fatal 小 smoke/2c GPU 自测
  归档为放行前允许；仅 r7_main（140 任务）与 tts-control 需书面放行；新增 §0 六步
  待执行清单（步骤→产物→核验→最终放行复核）；§2 节头注明放行后执行；
- 回复函（reply-review-final-gate-20260822.md）逐项归属 8 项缺失产物
  （#2/#7/#8 属放行后事项，非前置）。

## 2026-08-22 最终 Gate 复核意见落地：handoff 增补 0b 最小放行材料包

- 评审结论：实现级全过，7 项放行前材料全部为 GPU 现场产物（本机归档不接受）；
- handoff §0b 与评审材料包命名逐项对应（gate_clean_git/gate_selftest_gpu/
  r7_smoke_fatal 三件套/platform_conditions/tts_provenance/tts_probe_new/
  GATE_MANIFEST），附 manifest 生成命令（材料 hash 绑定拟批准 code_commit）；
- 本机无剩余整改项；等 GPU 主机执行 §0→§0b 后核验并申请最终书面放行。

## 2026-08-22 §2b 目录守卫冲突定夺（方案 A）+ GPU Gate 产物首批判收

- 冲突（R7_FORMAL_GATE_CHECKPOINT_DIR_CONFLICT_HANDOFF.md）：三 run 同目录与"一目录一
  checkpoint"守卫冲突；定夺方案 A 且一次改到位——fatal_smoke/ r7_main/ tts_control/ 三
  子目录，守卫零改动（方案 B 需重新过审，弃）；
- handoff 更新：§0c 目录说明、§0b 材料包与 manifest 路径、§2/§2b/§3 命令、§5 产物说明；
- GPU 首批 Gate 产物收妥核验：clean 树（HEAD=2e54ac2）、GPU self-test 90 PASS、
  platform_conditions（双 3090/驱动 550.127.05/CUDA 12.4/Triton fallback×4）、
  TTS provenance（commit 8555549e+163 行 diff/image digest/依赖）与新探活（ok/pcm）；
- 关键发现：spk2info.pt 中"晓伊"与"中文女"embedding 完全相等（diff=0.0）——speaker
  映射注记获模型级证据；
- 待 GPU：§2b 新目录重跑 + manifest 重生成 → 本机核验 → 最终书面放行复核。

## 2026-08-22 Gate 材料包核验完成（GPU commit 51f5d8f）

- 独立核验 gate material：clean/code provenance、GPU selftest 90/0、CosyVoice provenance、
  platform conditions、新探活、manifest 8 项全部齐备；Git blob 原始内容重算 manifest 8/8 hash 一致；
- 非末位 fatal smoke 独立重算：success→fault error+fatal→4×cancelled_after_fatal，
  cancelled 无事件污染，QA 0 问题；
- 两处非阻塞瑕疵登记：selftest 归档路径复制对齐（原始保留）；fatal smoke run.log 因 tee 先于目录创建
  未落盘（控制台后台日志+checkpoint/RUNINFO/QA/summary/CV 完整）；handoff 已修先 mkdir；
- 新增 `gate-material-verification-20260822.md`，建议提交最终书面放行复核（r7_main + tts_control）。

## 2026-08-22 Gate材料复核整改 r2（唯一基线+原子材料包）

- 审查意见全部采纳：版本混用/clean记录矛盾/platform 未绑定/manifest 不完整+hash语义未明/材料 modified 均属实；
- handoff 升级 r3：唯一 code_commit clean checkout；clean证明在材料生成前取得；fatal binding 绑定同一 platform hash；
  manifest 明确 LF-normalized 内容 SHA-256，覆盖 selftest log+md/fatal checkpoint+RUNINFO+QA+run.log/platform/probe/全部 provenance；
  材料提交后再写 artifact commit+porcelain；
- §2c selftest 直接写 env/gate；§2b 先 mkdir 后 tee run.log；子目录守卫保持；
- 正式 r7_main/control 仍不启动，待 GPU 重建最小材料包后最终书面复核。

## 2026-08-22 Gate 材料包 r3 原子重建核验通过（GPU a1fbb82/6aaf356，code_commit b8893d6）

- GPU 在唯一 clean 基线 `b8893d6` 重建最小材料包：G1 clean 证明（porcelain 真正为空，材料生成前采集）、
  G7 TTS provenance、G8 platform、§2c GPU self-test 90/0、§2b 非末位 fatal smoke、§0b 完整 manifest；
- 上一轮 5 项阻塞全部关闭：唯一 code_commit（四处引用统一 b8893d6）；fatal smoke 绑定
  platform_conditions_sha256=a4c40057…（不再 null）；manifest 12 项覆盖 + 显式 LF-normalized 语义；
  材料全落 artifact commit、提交后 porcelain 空；self-test HEAD 统一；
- 本机以 Git blob + LF-normalized 内容 SHA-256 重算 manifest 12/12 一致；
- fatal smoke 6 记录（success→fault error+fatal→4×cancelled_after_fatal）QA 0，run.log 完整落盘；
- 新增 `gate-material-verification-r3-20260822.md`，建议审查方出具正式书面放行
  （r7_main 主实验 → 结果级 QA 通过后 tts_control）。

## 2026-08-22 Gate r3 复核 r2：修正 gate_artifact_commit.txt 自引用时序语义

- 审查判定「条件通过、修正一处后最终复核」：`gate_artifact_commit.txt` 原 porcelain 记录的是
  该文件自身未跟踪，不能声称「含自身提交后 porcelain 空」——证据自引用/时序问题，非数据/基线错误；
- 采纳方案 A：文件改为准确历史记录（`porcelain_at_capture` + `note=` 明确「采集时仅本文件待提交，
  提交后 clean 由当前 checkout 独立验证」），不再自证提交后 clean；
- r3 核验清单 §5 同步修正提交后 clean 的独立证据表述（`e424eed` 上 `git status --porcelain` 空）；
  §4 补 manifest 生成时点说明（不含登记文件，因其在 artifact commit 之后才提交）；
- 代码基线不变（b8893d6）；a1fbb82/6aaf356/e424eed 均非 formal code commit。

## 2026-08-22 r7_main 书面放行后：修正 handoff 任务数 120→140 并写明版本操作

- 审查方已书面授权 r7_main（50 条 A/B 主实验 + 10 条子集补轮；tts_control 仍须 r7_main 结果级 QA 后另行复核）；
- 修正 handoff §2/§4 的「120 任务」笔误：`build_schedule` 实际产出 **140**（repeat 0=50×2=100，
  repeat 1/2=10 子集×2 模式×2 轮=40）；QA 用 tasks 动态算预期、不硬编码，RUNINFO 打印实际任务数；
- 写清版本操作：脚本/src/sample-list 在 b8893d6 与 origin/main 逐字节一致，RUNINFO 的 code_commit 仍记 b8893d6；
  但放行版 platform_conditions.txt（a4c40057…）仅在 a1fbb82 及之后存在，b8893d6 中是旧版 6b0a2fcd…，
  故正式 run 必须在 origin/main（含 Gate 材料包）上执行，--platform-conditions-file 才指向放行版；
- 正式 run 边界重申：新 r7_main/ 目录、新 checkpoint、新 run_id、启动重新探活；不从 smoke 续跑、不跑真实 tts_control。
