# GPU 主机侧实现核对与 E0 冒烟复验报告（2026-08-19）

> 任务来源：需求方指示"按 `experiments/GPU_EXPERIMENT_HANDOFF.md` 完成实验编码，编码后先评审，
> 不直接运行正式实验程序，可进行程序测试与小规模验证"。
> 核对基准：任务书 §三（DEV-1~DEV-5 规格）、§四-E0（冒烟验收点）、§1.3（配置锁定表）。
> 核对对象：本机 `main@cbfd01d` 已随 git 到位的全部 DEV 代码。
> **结论：DEV-1~DEV-5 全部到位且与规格一致，E0 冒烟全部通过，未运行任何正式实验，具备按任务书 §四 执行的条件。**

---

## 一、规格逐项核对（代码审查）

### DEV-1 `experiments/scripts/run_exp_latency.py` 扩展 —— ✅ 6/6 符合

| # | 规格要求 | 实现核对 |
|---|---|---|
| 1 | `--sample-list` 过滤 | ✅ main() 支持数组与 `{"sample_ids": [...]}` 两种格式，过滤后报告 missing |
| 2 | 数据集扩展（librispeech/aishell1/变体，`all`=全扫描） | ✅ `--dataset` 接受任意子目录名，`all` 扫描 `processed/json/` 全部子目录 |
| 3 | `--append-silence-ms` + 三个新计时点 | ✅ `speech_end_time`（最后真实块推完）、`final_speech_segment_commit_time`（VAD 闭段或含语音 flush 段入队，判 `contains_speech` 显式字段）、`final_is_final_segment_enqueue_time`；`audio_end_time` 语义不变；合法性判定抽为纯函数 `classify_endpoint_times`（可回归） |
| 4 | `--save-full-response` → `full_response` | ✅ 流式/非流式两路径均实现，`response_preview` 保留 |
| 5 | `--save-fragments` → `committed_fragments` | ✅ 在 `text_queue.put((output_text, False))` 处同步收集 |
| 6 | `--asr-model-size` choices 加 `turbo` | ✅ |

`ExperimentResult` 新字段默认值均为空（`full_response=""`、`committed_fragments=[]`、三个时间点 `0.0`），旧 checkpoint 可加载 ✅。

### DEV-2 `src/asr/faster_whisper_streamer.py` 提交分歧插桩 —— ✅ 符合（纯观测，不改行为）

- `ASRAudioSegment` 新增 `committed: bool = False`、`committed_text: str | None = None` ✅
- `_extract_output_text()` 置 committed 标记+快照，非空输出时追加 `commit_log`（`t/text/segment_ids`）✅
- `transcribe_audio_segment()` 重识别循环中检测 `committed and text != committed_text` → 追加 `correction_events`（`segment_id/old/new/t`）✅
- `reset_commit_tracking(sample_id)` 每样本清空两个列表并记录 sample_id ✅
- 实验脚本侧 `_write_commit_log()` 按规格 JSONL 格式追加写 `<output_dir>/commit_log.jsonl`（commit 行含 `sample_id/type/round/text/segment_ids/t`，correction 行含 `sample_id/type/segment_id/old/new/t`）✅

### DEV-3 `src/asr/local_agreement_streamer.py`（206 行）—— ✅ 符合

- 模型加载复用 `_load_whisper_model_offline_first()` ✅；转录参数逐项复用 `DEFAULT_*` 常量，与 `_transcribe_segments` 完全一致（beam_size=5, word_timestamps, temperature=0.0, compression_ratio/logprob/no_speech 阈值, condition_on_previous_text=False）✅
- 分段沿用 `StreamAudioSegmenter`（由 DEV-4 保证）✅
- LA-2 算法步骤 1–8 全部落实：触发阈值 `decode_trigger_s=2.0`（对齐 recognition_threshold）、按词文本最长公共前缀、`trailing_margin` 提交线（锁定配置 suffix=0 → 0.0，由 DEV-4 显式传入）、buffer 裁剪（最后提交词 end−0.1s，词时间戳前移保持相对轴）、`flush()` 提交全部剩余 ✅
- 超规格的稳健性处理（前两轮本机评审整改产物）：空识别轮保留 `prev_words`/`n_committed` 防文本静默丢失；`divergence_events` 观测已提交位置前分歧（append-only 不回退）✅

### DEV-4 `experiments/scripts/run_exp_baseline_la.py`（472 行）—— ✅ 符合

- 仅 `la_streaming` 单模式；四 worker 结构复用，ASR 侧 `feed_segment` → 片段 `text_queue.put((frag, False))`，队列排空后 `(flush(), True)` ✅（`("", True)` 安全性有注释论证：is_end=True 走 generation_prompt 路径）
- TTFT 定义一致（`first_token_time − audio_end_time`）✅
- 继承 `--sample-list`、断点续传、检查点（直接复用 DEV-1 的 `load_samples/load_checkpoint/save_checkpoint`）✅
- 3 轮预热不跳过 ✅；WER/CER 复用 `run_exp_quality` 同口径归一化 ✅；输出 `la_results/la_summary/la_statistics` 三件套 ✅；config 块含 `trailing_margin_s: 0.0` 与双设备字段 ✅

### DEV-5 `experiments/scripts/measure_tts_first_chunk.py`（291 行）—— ✅ 符合

- 复用 CosyVoice 流式 HTTP 口径（`/inference_sft`，spk_id=晓伊，speed=0.8，22050Hz/16bit/mono PCM）✅
- `--input` JSONL 与 `--from-e4` 自动抽取（streaming 模式 + 无 error + 按 sample_id 前缀判语言，seed=42）✅
- TTFC/total/audio_sec/rtf 测量与 CSV 格式 ✅；请求间隔 ≥1s ✅；失败标记 error 继续 ✅；服务不可达停止（exit 2，不写 CSV）✅
- 防误用设计：`full_response` 缺失/为空默认拒绝退出，仅显式 `--allow-preview-fallback` 才用截断预览（CSV 标记 `text_source=preview`）✅；自动写 `.runinfo.md` ✅

### 配套（E2-0 链路）

`build_real_speech_set.py` / `build_augmented_variants.py` / `qa_real_speech.py` / `test_r2_build_smoke.py` 已随 git 到位（本机冒烟 16/16 + test-clean 真实小配额 WER 2.75% 已验证，见 commit 60a6890）。

---

## 二、程序测试与 E0 小规模验证证据（GPU 主机实测）

| 验证项 | 命令/方式 | 结果 |
|---|---|---|
| 修订回归套件 | `uv run python -m experiments.scripts.test_revision_regressions` | ✅ **10/10 passed** |
| R2 构建冒烟 | `uv run python -m experiments.scripts.test_r2_build_smoke` | ✅ **16/16 通过** |
| E0：DEV-1/2 冒烟 | `run_exp_latency --dataset crosswoz --max-samples 2 --asr-model-size tiny --llm-model-name Qwen/Qwen2.5-0.5B-Instruct --asr-device cuda:0 --llm-device cuda:1 --append-silence-ms 2000 --save-full-response --save-fragments` | ✅ 2 样本 × 2 模式无 error；`audio_end−speech_end` = 2.003s/2.002s（≈2s ✅）；`endpoint_detection_wait` = 0.199s/0.175s（与本机 0.18s 同量级 ✅）；`final_speech_commit ≤ final_is_final_enqueue` ✅；`full_response`（87/73 字符）与 `committed_fragments`（1/23 段）非空 ✅ |
| E0：DEV-2 commit_log | 同上产物 `/tmp/dev_smoke/latency/commit_log.jsonl` | ✅ 46 行：每样本有 commit 记录（格式与 DEV-2 规格逐项一致）；correction 22 条（见发现项 O2） |
| E0：DEV-3/4 冒烟 | `run_exp_baseline_la --dataset crosswoz --max-samples 2 --asr-model-size tiny --llm-model-name Qwen/Qwen2.5-0.5B-Instruct --asr-device cuda:0 --llm-device cuda:1 --save-fragments` | ✅ 2 样本 0 失败；LA 有文本提交（2/4 片段）、LLM 正常生成、la_ 三件套齐全；TTFT 1396ms/2894ms；无队列死锁 |
| DEV-5 逻辑 | 伪造 E4 JSON 单测 `load_from_e4` + 服务不可达实测 | ✅ zh/en 各 25 条抽取正确、seed 确定性复现；空/缺 `full_response` 默认拒绝、显式 fallback 生效；服务不可达 exit 2 且不写 CSV |

---

## 三、发现项（均不阻塞正式实验，供执行阶段与离线分析注意）

- **O1（LA 指标语义）**：LA 冒烟的 150s 样本 `asr_time = −18.97s`（负值）。原因：tiny 模型下 divergence=14，尾部多轮无新提交且 `flush()` 返回空，`last_text_time` 停留在音频结束前约 19s。`asr_time = last_text_time − audio_end_time` 定义本身允许负值（含义：最后一次文本提交早于音频结束），E3 用 turbo 后分歧应大幅减少；离线统计 LA 的 asr_time 时建议说明负值口径，勿直接当异常剔除。
- **O2（correction 非零）**：tiny 冒烟 2 样本产生 22 条 correction（prefix 保留段在上下文增长后重识别漂移，如 '点我希望评分是4.5分'→'我希望評分是4.5分'）。规格称"0 条属正常"——非零同样是**有效观测**（这正是 E4 要测量的现象），tiny 模型漂移频繁符合预期，turbo 下应显著减少。不视为缺陷。
- **O3（E6 阻塞项）**：CosyVoice TTS 服务未部署（`host.docker.internal:20401` DNS 不可达）。DEV-5 脚本行为正确（探活失败即停止）。E6 执行前需按 `experiments/datasets/tools/doc/TTS_USAGE.md` 部署服务并探活；若原镜像无法恢复，按任务书 §1.0-8 记录现象回告需求方，不自行换 TTS。
- **O4（RUNINFO.md）**：§5.1 要求"每次运行后写 RUNINFO.md"——DEV-5 已自动化（`.runinfo.md`），其余脚本未自动写，正式运行每个 E 任务后需人工补写（完整命令行、起止时间、样本数、error 数）。
- **O5（噪音日志）**：冒烟日志中 `ld: cannot find -lcuda` 为 triton 编译探测噪音（i386 目录下的 32 位 stub），不影响运行与结果（G3/G4 及本次冒烟均验证）。
- **O6（文档表述，上轮已报）**：`GPU_HOST_SETUP.md` G3/G4 期望"3 个模式"为过时表述，实际 `run_exp_latency.py` 为 2 模式（streaming/non-streaming），LA 模式在 DEV-4 独立脚本。建议回本机修订手册。
- **O7（环境遗留提醒）**：`.env` 中 HF_TOKEN 仍为失效 token（按手册 Step D 本机适配、未提交）；本机所有下载命令以 `HF_TOKEN=` 前缀置空绕过。后续任何新增模型下载须保留此前缀或清空 .env 中 token。

## 四、本次未执行事项（按指示）

- 未运行任务书 §四 任何正式实验（E1/E2/E3/E4/E5/E6），等待需求方另行安排；
- 未 commit/push（任务书规定 GPU 侧全程不需要）；
- 冒烟产物在 `/tmp/dev_smoke/`（临时目录，不污染 `experiments/results/`，未触碰 exp1/exp2/exp3 只读目录）。
