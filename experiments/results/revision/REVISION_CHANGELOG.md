# REVISION_CHANGELOG — CISR 修订补充实验执行记录

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
