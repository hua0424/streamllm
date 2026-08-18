# CISR 审稿意见响应：补充实验方案

本文档规划针对 CISR 审稿意见（5 条）所需的全部补充实验与前置准备工作。
**目标：按本方案执行完毕后，即可直接进入论文修改与回复信撰写阶段。**

配套阅读：`experiments/EXPERIMENT_DESIGN.md`（原实验设计）、审稿意见逐条分析（会话记录）。
本方案执行完成后，应将新实验的方法学描述回填到 `EXPERIMENT_DESIGN.md`。

---

## 〇、总览：审稿意见 → 实验 → 论文修改 映射

| 实验编号 | 对应意见 | 内容 | 是否需要 GPU 重跑 | 预计工期 |
|---|---|---|---|---|
| R0 | — | 前置准备：环境核对、原始数据恢复 | — | 0.5 天 |
| R1 | 意见3 | 统计显著性补强（Std/P95/P99 + 重复测量） | 大部分否（仅子集复跑） | 1 天 |
| R2 | 意见1 | 真实语音验证（LibriSpeech + AISHELL-1 + 噪声/变速增强） | 是 | 2–3 天 |
| R3 | 意见4 | LocalAgreement 流式基线（同权重同引擎） | 是 | 2–3 天 |
| R4 | 意见5(机制) | 提交分歧率 + 分词接缝不匹配率测量 | 是（小插桩复跑）+ 离线分析 | 1–2 天 |
| R5 | 意见5(质量) | 下游回复语义一致性（嵌入相似度 + LLM 裁判） | 是 + API | 1–2 天 |
| R6 | 意见2 | TTFA 端到端预算（端点等待 + TTFT + TTS 首包） | 部分是 | 1–2 天 |
| R7 | — | 汇总：论文图表、回信证据清单、修改点清单 | 否 | 1 天 |

**关键既有事实（代码核查结论）：**

- 现有结果 JSON（`experiments/results/exp1_latency/exp1_results_20251210_024430.json` 及 exp2/exp3 对应文件）**逐样本存储了 ttft / asr_time / llm_prefill_time**，因此 P95/P99 可直接离线重算，无需重跑（R1 主体）。
- exp1 结果中 `transcribed_text` 是多片段拼接后的整串（`" ".join(...)`），**未保存逐片段边界**；`response_preview` 只存前 100 字符。因此 R4（逐片段插桩）与 R5（完整回复）**必须小改代码后复跑子集**。
- ASR 后端为 openai-whisper 原生 PyTorch（`src/asr/faster_whisper_streamer.py` 文件名易误导，实际 `whisper.load_model`），LLM 为 HF Transformers（`src/llm/stream_llm_inference.py`）。R3 基线必须复用同一加载路径以保证 matched models/engine。
- exp3 已有 `normalize_text / cer / wer`（`experiments/scripts/run_exp_quality.py` 第 99–196 行），真实语音评测直接复用。
- TTS 客户端（`experiments/datasets/tools/tts.py`）本身就是流式 HTTP（`stream=True` 逐块读 PCM），**测量 TTS 首包延迟无需改协议，只需加计时**。
- `run_exp_latency.py` 的 `--asr-model-size` choices 未含 `turbo`（靠 .env 默认值绕过校验），修改脚本时一并补上。

---

## 工作分工：GPU 实验机 vs 本机

自 2026-08-18 起，本方案按两台机器分工执行。**GPU 机的详细执行任务书见 `experiments/GPU_EXPERIMENT_HANDOFF.md`**（自包含，可直接交给 GPU 机执行）。

**本机硬件说明**：本机为 Windows + RTX 3060 Laptop（6 GB 显存，CUDA 可用，`.venv` 已与 `uv.lock` 同步）。可用于：小模型冒烟测试（`.env` 默认 whisper-tiny + Qwen2.5-0.5B）、Whisper-Turbo 级别的 ASR 质量校验（fp16 约 2 GB，可装下）、bge-m3 嵌入评估。**不可用于**：Qwen2-7B 推理（fp16 约 15 GB 装不下）、任何要写进论文的延迟数字（延迟结论必须与论文声明的 2×RTX 3090 平台一致，本机跑出的 TTFT 一律作废，仅用于验证逻辑正确性）。

**GPU 机情况**：全新安装的 Ubuntu 22.04（系统重装，盘内无历史数据），硬件同为 2×RTX 3090。按 handoff §1.0 从零初始化环境（驱动/uv/clone/依赖/模型预下载），**全新 clone 即正确做法**（git 中不含数据，数据见下行）。

**代码与数据流向**：代码经 git（GitHub `origin/main`）流转——本机开发并推送，GPU 机 `git pull` 获取；数据（`processed/`、`raw_data/` 已被 .gitignore 排除）走 git 之外的文件传输。**原始 1,132 条实验数据完整保存在本机**（已核验：multiwoz 630 + crosswoz 503，JSON 与 WAV 一一对应，3.4 GB），由本机打包传给 GPU 机。

| 工作项 | 归属 | 说明 |
|---|---|---|
| R0 环境初始化（全新 Ubuntu）+ 数据核验、版本存档 | **GPU 机 + 本机** | GPU 机按 handoff §1.0 从零初始化（可立即开始）；原始数据包由本机打包传出 |
| DEV-1/2/3/4/5 程序开发 + 小模型冒烟 | **本机** | 用 .env 默认小模型在本机验证逻辑后推送到 main；GPU 机 pull 后按 handoff §3 复验 |
| R1.1 离线重算分位数 | **本机** | 结果 JSON 已在本机，纯 CPU |
| R1.2 重复测量（50 样本 × 3 轮） | **GPU 机** | 样本清单 `repeat_subset_ids.json` 由本机生成（从 exp1 结果 JSON 提取）后传入 |
| R2 真实语音：下载/构建/增强/QA | **本机** | 纯 CPU；QA 中的 System A WER sanity 可用本机 3060 跑 Whisper-Turbo（QA 用途，非论文数字）；产物打包传入 GPU 机 |
| R2 真实语音：实验运行 | **GPU 机** | handoff §4-E2 |
| R3 LA 基线：样本清单（从 exp2 结果 JSON 提取并固定） | **本机** | ✅ 已生成（`make_sample_lists.py`）：干净成对子集 498 条（long 108 / very_long 150 / extra_long 240），排除规则显式化（运行错误 3 条 + 流式 TTFT>10s 挂起 4 条）；旧手工修复 `static-repair.csv` 不可复现已弃用，Table IV 数字按本清单重算更新 |
| R3 LA 基线：运行 | **GPU 机** | handoff §4-E3（代码由本机经 git 交付） |
| R4 插桩复跑 + 分词接缝离线分析 | **GPU 机 + 本机** | 插桩复跑在 GPU 机（含 Qwen2-7B，本机显存不够）；接缝分析脚本在本机对回传日志运行 |
| R5 复跑取完整回复 | **GPU 机** | 与 R4 合并为同一次运行 |
| R5 嵌入相似度 + LLM 裁判 | **本机** | bge-m3 用本机 3060 加速；裁判走 API |
| R6.1 端点等待测量 | **GPU 机** | handoff §4-E5 |
| R6.2 TTS 首包测量 | **GPU 机** | 依赖该机可达的 CosyVoice 服务 |
| R7 汇总、绘图、回信证据清单 | **本机** | 全部结果回传后进行 |

---

## 一、R0：前置准备（0.5 天，一切实验的前提）

### 1.1 原始实验数据 ✅ 在本机，完整

**原始 1,132 条合成样本完整保存在本机** `experiments/datasets/processed/`（2026-08-18 核验：multiwoz 630 JSON+630 WAV、crosswoz 503 JSON+503 WAV；crosswoz 比论文计数 502 多 1 条为原实验运行时失败样本，属正常，共 3.4 GB）。

**注意：GPU 机系统已重装为全新 Ubuntu 22.04，盘内无历史数据。** 因此：

- 原始数据包由本机打包（整个 `processed/` 目录）经文件渠道传给 GPU 机，GPU 机按 handoff §1.1 做完整性核验（数量 + 时长抽验）；
- GPU 机环境按 handoff §1.0 从零初始化（驱动/uv/clone/uv sync/模型预下载/CosyVoice 重新部署）；
- 本机离线分析工作不依赖原始音频，仅依赖已有结果 JSON（已核实字段齐全：exp1 2266 条逐样本结果、exp2 三模式结果、exp3 三配置结果均在 `experiments/results/` 下）。

### 1.2 环境核对（与论文 §V-A 声明一致）

已从历史结果 JSON 的 config 块确认论文实验的实际配置（**所有复跑必须与此锁定一致**，注意 `--suffix-segments 0` 不是脚本默认值，必须显式传参）：

| 参数 | 锁定值 |
|---|---|
| ASR 模型 | `turbo`（openai-whisper large-v3-turbo），`cuda:0` |
| LLM 模型 | `Qwen/Qwen2-7B-Instruct`，`cuda:1` |
| chunk_duration | 500 ms |
| prefix / suffix segments | **1 / 0** |
| recognition_threshold | 2.0 s |
| 解码参数 | temperature 0.1, top_p 0.9, repetition_penalty 1.1 |
| max_tokens | 50（R5 取完整回复的运行除外：128） |
| warmup | 3 轮真实音频 |

- [ ] GPU 机 `.env`：`ASR_MODEL_NAME=turbo`、`LLM_MODEL_NAME=Qwen/Qwen2-7B-Instruct`。
- [ ] 记录版本信息备回信使用：`pip list` 输出存档到 `experiments/results/revision/env_versions.txt`。
- [ ] 确认预热流程（3 轮真实音频预热）沿用现有脚本，不改动。

### 1.3 建立本次修订的目录与清单

```
experiments/
├── datasets/raw_data/            # 新增：公开数据集原始包
│   ├── librispeech/  aishell1/  musan/
├── results/revision/             # 新增：本次修订全部产物
│   ├── r1_stats/  r2_real_speech/  r3_baseline_la/
│   ├── r4_commit/  r5_semantic/  r6_ttfa/
└── REVISION_CHANGELOG.md         # 新增：每个实验完成后记录关键数字，供回信引用
```

---

## 二、R1：统计显著性补强（意见3，1 天，绝大部分纯离线）

### 2.1 离线重算分位数（无需 GPU）✅ 已完成（2026-08-18）

**脚本** `experiments/scripts/recompute_stats.py` 已实现并运行，产出见 `results/revision/r1_stats/`。关键发现：

- **Table III 均值与论文完全一致**（streaming Long 1126.63 / Very Long 1099.16 / Extra Long 1087.70 ms），仅成对排除 1 条运行错误样本（crosswoz_7310_turn10）；baseline Extra Long 均值 6753.43→6745.57。
- **Table V 的 "ASR time" 列口径已查明**（2026-08-18 二次核实，推翻此前"无法复现"的判断）：该列 = summary CSV 中 `asr_time_ms` 对**两种模式合并**（300 行）的均值，1327.48/1224.96/1086.16 与归档数据**精确一致**，论文 Table V 保持原值无需修改。注意该口径混合了流式尾时延（1123.07/932.96/668.61）与非流式全音频解码时间（≈1500ms），分模式明细与分位数已存入 `table5_context_percentiles.csv`（含 pooled 复核行）；修改稿如需更纯粹的"尾时延"叙述，可引用 streaming 单列（suffix1−default=+190ms，default−pre0suf0=264ms，方向与现文一致）。
- 平台稳定性表述改为可辩护口径：System B 流式 P99 有界（1979/2174/2605 ms），Long→Extra Long 仅增 1.32×（baseline 4.96×）；Extra Long 流式 P99 为 baseline 的 0.21 倍。不再使用"P99≤1.5×mean"判定（该口径不成立）。

**原规格（已按上述实现）**：

- 读取：
  - `experiments/results/exp1_latency/exp1_results_20251210_024430.json`
  - `experiments/results/exp2_ablation/exp2_results_20251214_002214.json`
  - `experiments/results/exp3_quality/{suffix0_result,suffix1_result,prefix0suffix0}/exp3_results_*.json`
- 对每个分组 × 每个模式（streaming / non-streaming；exp2 为 baseline / streaming_asr_only / full_streaming）计算：**mean、std、P50、P90、P95、P99、min、max、n**（TTFT，单位 ms）；exp3 另对 `asr_time_ms` 与各 WER/CER 列做同样统计。
- 输出：
  - `results/revision/r1_stats/table3_latency_percentiles.csv`（对应论文 Table III 行布局）
  - `table4_ablation_percentiles.csv`、`table5_context_percentiles.csv`
  - `plateau_stability.txt`：Long/Very Long/Extra Long 三组 System B 的 P95、P99，与 baseline P99 的倍数关系，及尾部有界性结论。

**论文需要的数字**：三张表每格 mean±std 与 P95/P99；平台稳定性结论一句话（如"三组 P99 均低于 X ms"）。

### 2.2 重复测量（需 GPU，半天）

- 从 Very Long 组固定抽取 50 个样本（样本 ID 列表存 `results/revision/r1_stats/repeat_subset_ids.json`，固定种子选取，后续 R4/R5 复用同一列表）。
- **代码改动点**：`run_exp_latency.py` 增加 `--sample-list` 参数（读 ID 列表过滤 `load_samples` 结果；约 10 行）；顺手把 `--asr-model-size` choices 补上 `"turbo"`。
- 用相同配置连跑 **3 轮**（每轮换 `--output-dir` 子目录）。
- 分析：逐样本计算 3 轮 TTFT 的变异系数 CV，汇总 mean/max CV 与轮间均值漂移，写入 `repeat_stability.txt`。
- **论文需要的数字**："50 样本 × 3 轮，TTFT 轮间 CV 均值 X%、最大 Y%"，支撑"repeat measurements"要求。

### 2.3 GPU 竞争（可选，P2）

若时间允许：在跑 2.2 子集时，于同两张卡上启动一个常驻负载进程（简单 PyTorch matmul 循环脚本 `experiments/scripts/gpu_coload.py`，占 ~8 GB/卡），对比有无竞争的 TTFT 分布差异。不做则在论文 Limitations 加一句"实验为独占 GPU 环境"。

---

## 三、R2：真实语音验证（意见1，2–3 天）

### 3.1 数据下载（已核实的地址与许可）

| 数据集 | 用途 | 下载地址 | 许可 | 大小 |
|---|---|---|---|---|
| LibriSpeech test-clean | 英文真实朗读 | `https://www.openslr.org/resources/12/test-clean.tar.gz` | CC BY 4.0 | ~346 MB |
| LibriSpeech test-other | 英文真实朗读（含口音/较差录音） | `https://www.openslr.org/resources/12/test-other.tar.gz` | CC BY 4.0 | ~328 MB |
| AISHELL-1 | 中文真实朗读（400 说话人各口音区） | `https://www.openslr.org/resources/33/data_aishell.tgz` | Apache 2.0 | ~15 GB（可只解 `wav/test`+`wav/dev` 与 `transcript/`） |
| MUSAN | 噪声注入素材 | `https://www.openslr.org/resources/17/musan.tar.gz` | CC BY 4.0 | ~11 GB（可只解 `noise/` 与 `speech/`） |

存放：`experiments/datasets/raw_data/{librispeech,aishell1,musan}/`。
LibriSpeech 为 16 kHz FLAC；AISHELL-1 为 16 kHz WAV；MUSAN 采样率不一，注入前统一重采样到 16 kHz。

### 3.2 清洗与长语音构建

**新脚本** `experiments/datasets/tools/build_real_speech_set.py`：

1. **LibriSpeech**：按 `(speaker_id, chapter_id)` 分组，句按 utterance id 排序；同章节内顺序拼接，**句间插入随机静音间隔 U(0.2, 1.0) s**（制造停顿/端点变化，压力测试 VAD）；参考文本 = 对应 `.trans.txt` 行顺序拼接。
   - 过滤：丢弃无转写或音频损坏的句；丢弃拼接后仍不足 15 s 的章节。
2. **AISHELL-1**：按 speaker 分组（优先官方 test/dev 说话人），同说话人顺序拼接，同样插随机静音；参考文本取自 `transcript/aishell_transcript_v0.8.txt`（**注意该转写为字间带空格格式，读取后去空格**）。
3. **目标规模**：英文 75 条（Long 30 / Very Long 30 / Extra Long 15），中文 75 条（同分布）。拼接目标时长落在对应组区间；总长上限 150 s（与原实验一致）。
4. **QA 校验（必做）**：
   - 每条样本：时长 = 各句时长+间隔之和（容差 50 ms）；参考文本非空；RMS 能量检查排除静音文件；
   - 抽 5 条人工试听；
   - ** sanity check**：对构建好的集合跑一遍 System A（离线 Whisper-Turbo），test-clean 部分 WER 应在低个位数；若明显偏高（>10%），说明拼接/转写对齐有 bug，先修再继续。
5. **输出格式与现有管线完全一致**（保证 `load_samples` 免改）：
   - JSON：`processed/json/{librispeech,aishell1}/<sample_id>.json`，字段同原 schema（`sample_id, dialog_id(=speaker/chapter), turn_index=0, text, text_length, audio_file, audio_duration, language, dataset`）；
   - 音频：`processed/audio/{librispeech,aishell1}/<sample_id>.wav`（16 kHz 单声道）。

### 3.3 增强变体（噪声 + 变速）

**新脚本** `experiments/datasets/tools/build_augmented_variants.py`，对 3.2 的每条真实样本生成：

- **加噪**：MUSAN `noise/free-sound`（环境噪声）与 `speech/`（ babble）两类，按 RMS 归一叠加到 **SNR = 20 / 15 / 10 dB** 三档；噪声长度不足则随机裁剪循环。
- **变速**：`librosa.effects.time_stretch`（或 torchaudio sox `tempo`），**0.9× 与 1.1×** 两档；变速后时长改变 → 更新 JSON 的 `audio_duration` 并重新判定分组。
- 每个变体写独立的 JSON+audio 目录（如 `librispeech_snr10`、`aishell1_speed09`），保持 schema 不变。

### 3.4 实验运行

- **代码改动点**：`run_exp_latency.py` 与 `run_exp_quality.py` 的 `--dataset` choices 与 `load_samples` 的数据集列表加入 `librispeech / aishell1`（约 5 行）。
- **运行矩阵**（System A vs System B，沿用原 .env 配置、3 轮预热）：
  - 干净集：全部 150 条 × 2 模式（复用 `run_exp_latency.py`，TTFT + 逐样本转写落盘）；
  - 增强集：每语言每条件抽 30 条（Long+Very Long 为主）× 2 模式：3 SNR 档 + 2 变速档 + babble 一档，共约 6 条件 × 60 条。
- **WER/CER 离线计算**（新小脚本 `experiments/scripts/score_wer_offline.py`）：读取运行结果 JSON 中两模式的 `transcribed_text`，与样本 JSON 的 `text` 比对，**复用 exp3 的 `normalize_text / wer / cer`**，按语言分别汇总（英文报 WER、中文报 CER，同时保留双指标与原文一致）。

### 3.5 交付数据（论文新 Table VI + 正文段落）

- `results/revision/r2_real_speech/ttft_real.csv`：真实语音上 System A/B 的 TTFT mean±std/P95（分组），验证 1.1 s 平台在真实语音上是否保持；
- `wer_real.csv`：干净 + 各增强条件下 System A/B 的 WER/CER（含 System A 在非零错误下的真实上界，回应"near-zero error"）；
- 一段机制性观察（如噪声下 VAD 段数变化、提交延迟变化），从日志提取 2–3 个具体数字。

---

## 四、R3：LocalAgreement 流式基线（意见4，2–3 天）

### 4.1 实现要点

**新模块** `src/asr/local_agreement_streamer.py`（参照 whisper_streaming 的 LocalAgreement-2 策略，自行实现以保证同引擎）：

- **模型加载：复用 `whisper.load_model(ASR_MODEL_NAME)`**（与 `StreamingASRProcessor` 完全相同的权重/精度/设备），只换上下文与提交策略。
- 流程：复用同一 `StreamAudioSegmenter` 分段 → 维护持续增长音频缓冲 → 每达到触发条件（沿用 `recognition_threshold=2s`）对缓冲整体转录（`word_timestamps=True`，参数与现有 `_transcribe_segments` 一致）→ **提交规则**：词结束时间落在"缓冲总时长 − 尾随保护余量（= 当前 suffix 余量，默认 1 个段时长）"以内、且与上一轮假设在重叠区一致的词 → 提交给下游。
- 下游：提交的文本走与 System B 相同的 `StreamLLMInference.cache_prompt` 增量预填路径（主对比臂），保证**唯一差异是 ASR 上下文/提交策略**。
- 复用 `run_exp_latency.py` 的队列/线程骨架：新增 `mode="la_streaming"` 分支即可，工程量集中在提交逻辑（约 150–250 行）。
- 引用：论文中引 whisper_streaming（UFAL）作为策略出处。

### 4.2 运行

- 数据：**与 exp2 消融完全相同的样本列表**，即 `exp2_ablation_sample_list.json`（已生成，498 条干净成对子集；排除规则：任一模式运行错误 3 条 + 任一流式模式 TTFT>10s 判定挂起 4 条）。该清单同时是 Table IV 重算口径——旧论文数字来自不可复现的手工修复（`static-repair.csv`），修改稿 Table IV 按本清单重算后更新（均值变化很小，结论不变）；论文中"50 utterances per group"的表述同步改为实际样本量。GPU 机不自行选定样本。
- 每样本跑 1 次 `la_streaming` 模式；System A/B 的对应数字直接取自已有 exp2 结果，不重跑。
- 检查点/断点续传机制照搬现有脚本。

### 4.3 交付数据（论文新 Table VII）

- `results/revision/r3_baseline_la/la_vs_b.csv`：三分组下 System A / LA-streaming / System B 的 TTFT mean±std 与 WER/CER；
- 一段归因讨论数字：System B 相对 LA 基线在 TTFT 与错误率上的差异，以及差异来源（prefix–suffix 上下文 vs 无上下文的局部一致）。

---

## 五、R4：提交分歧率 + 分词接缝不匹配率（意见5 机制部分，1–2 天）

### 5.1 提交分歧率（commit-divergence）—— 小插桩复跑

**背景**：设计上只有稳定区文本被提交，KV cache 为 append-only，无回滚代码路径。需要用实测数字背书"提交后不再变"。

**代码改动点**（`src/asr/faster_whisper_streamer.py`，约 30 行，不影响行为）：

1. `ASRAudioSegment` 增加 `committed: bool = False` 与 `committed_text: str | None = None`；
2. `_extract_output_text` 输出某段时：置 `committed=True`，记录 `committed_text`；
3. 下一轮 `_extract_segment_text` 覆写段文本时：若该段 `committed=True` 且新文本 ≠ `committed_text`，记录一次 **correction event**（`sample_id, segment_id, old, new, round_idx`）到全局列表；
4. 同时把**每轮实际提交的片段文本**（带提交时刻）记录到日志文件（R4 的分词分析需要逐片段序列）。

**运行**：用 R1 的固定 150 样本列表（Long/Very Long/Extra Long 各 50）跑 System B（合成集即可；真实集可选）。

**分析**（离线脚本）：correction event 总数、涉及段比例、每事件的编辑距离；另算"已提交前缀拼接串 vs System A 全量转写"的 WER 作为外部一致性参照。
**预期**：事件数为 0 或接近 0（保留前缀段重识别偶尔漂移）。无论结果如何都如实报告。

### 5.2 分词接缝不匹配率 —— 纯离线分析（无需 GPU）

**新脚本** `experiments/scripts/check_tokenizer_seams.py`：

- 输入：5.1 记录的逐片段文本序列 `[f1..fn]`（每样本）。
- 复现 LLM 侧实际分词路径（`stream_llm_inference.py`）：`ids_stream = tokenize(chat_template_prefix + f1)` 后逐片 `tokenize(fi, add_special_tokens=False)` 拼接，最后拼 `tokenize(generation_prompt)`；
- 对照：`ids_oneshot = tokenize(chat_template(user = f1..fn) + generation_prompt)` 一次性分词（Qwen2-7B-Instruct 原分词器）；
- 输出：每样本 `len_diff`、首个分歧位置、不一致 token 数；汇总"含 ≥1 处接缝不一致的样本比例"、"平均每条不一致 token 数"。
- 说明材料：现有实现按词/空白边界提交（英文天然空格分隔；中文按 Whisper 词级时间戳的词拼接），预期不匹配率极低；若个别样本出现 BPE 跨缝合并，报告其频率并说明对语义影响可忽略（配合 R5 的语义一致性证据）。

### 5.3 交付数据（回信 + 论文 §IV 新增小节）

- `results/revision/r4_commit/commit_divergence.json`：correction 事件统计；
- `tokenizer_seams.csv`：逐样本不匹配明细 + 汇总行；
- 三个写进回信的关键数字：**回滚次数 = 0（构造保证）**、**提交分歧率 = X%**、**接缝不一致样本比例 = Y%（平均 Z token/条）**。

---

## 六、R5：下游语义一致性评估（意见5 质量部分，1–2 天）

### 6.1 复跑取完整回复（需 GPU）

- **代码改动点**：exp 脚本结果结构增加 `full_response` 字段（不再截断 100 字符）；`max_tokens` 提到 128；System A、B 两模式都存。
- 运行：R1 固定 150 样本列表（或压缩至 100 条）合成集。若与 R4 复跑合并成一次运行（同批样本同时插桩 + 存全量回复），省一遍 GPU 时间——**推荐合并**。

### 6.2 双轨评估（离线）

- **轨道 A：嵌入相似度**。本地模型 `BAAI/bge-m3`（多语言，中英通吃，~2 GB，HF 下载）逐样本计算 `cos(embed(R_A), embed(R_B))`，汇总 mean/分布。
- **轨道 B：LLM-as-a-Judge**。API 调一个强模型（优先可用的：DashScope `qwen-max` / OpenAI `gpt-4o`；key 走 `.env`，不进仓库），prompt 给"用户输入 + 回复A + 回复B"，要求 1–5 分评语义等价性并判断是否保留用户意图；每样本一条 JSON 落盘。汇总 mean 分、≥4 分占比。
- 注意：评审打分时隐藏 A/B 来源（随机交换顺序），避免顺序偏置。

### 6.3 交付数据（论文 §V 新增小节 + 回信）

- `results/revision/r5_semantic/semantic_consistency.csv`：逐样本相似度与裁判分；
- 汇总：mean cosine = X、judge mean = Y/5、≥4 分占比 = Z%；
- 挑 2 个代表性 case（一个高度一致、一个差异最大）作论文定性分析。

---

## 七、R6：TTFA 端到端预算（意见2，1–2 天）

目标产出：**语音结束 → 首个可听音频帧** 的分解预算表（System A vs B）。

### 7.1 端点等待测量（需 GPU 小跑）

- **现状**：现有仿真在音频文件结束时立即 `flush()`，端点判定延迟≈0，**不能如实反映真实端点等待**（min_silence 300 ms + chunk 量化至 500 ms）。
- **代码改动点**：exp 脚本增加 `--append-silence-ms 2000` 选项：音频尾部拼 2 s 静音，并记录 (a) 真实语音结束时刻（原文件结尾）、(b) VAD 关闭最终段的提交时刻。端点等待 = b − a。
- 运行：50 条子集（Long+），仅 System B 需要（System A 的端点等待相同，取同一组数）。

### 7.2 TTS 首包延迟（无需 GPU，需 CosyVoice 服务）

- 复用 `experiments/datasets/tools/tts.py` 的流式客户端：**加计时**——从发起请求到收到第一块 PCM 数据的时间（time-to-first-chunk），以及整段合成的 RTF。
- 输入文本：直接用 R5 落盘的 50 条 System B 真实回复（覆盖中英）。
- 若 CosyVoice 服务不可用：在方案执行时确认；不可用则改用与论文数据生成一致的自建 CosyVoice 部署，或如实改用替代流式 TTS 并在论文中声明型号（避免与正文 CosyVoice 声明冲突，优先恢复原服务）。

### 7.3 汇总预算表

- 组成：`TTFA = T_endpoint(7.1) + TTFT(已有/R1) + T_decode_to_first_sentence(R5 日志中取首个句末标点 token 的时刻) + T_TTS_first_chunk(7.2)`。
- 网络延迟：单机部署、线程队列通信，T_Net≈0，正文与回信中说明界定理由。
- **交付**：`results/revision/r6_ttfa/ttfa_budget.csv`：System A/B 各组成项 mean±std 与合计 TTFA（预期 B 的优势在端到端口径下进一步扩大）。

---

## 八、R7：汇总交付物（进入论文修改前的最后一步，1 天）

### 8.1 论文图表/表格更新清单

| 论文位置 | 动作 | 数据来源 |
|---|---|---|
| Table III（TTFT 分组表） | 加 mean±std、P95、P99 列 | R1 `table3_latency_percentiles.csv` |
| Table IV（消融表） | 同上 | R1 `table4_ablation_percentiles.csv` |
| Table V（上下文配置表） | 同上 | R1 `table5_context_percentiles.csv` |
| Fig.6（TTFT 趋势图） | 重绘：加 P95 误差带 | R1 重算后的逐样本数据 |
| 新 Table VI（真实语音） | 新增 | R2 |
| 新 Table VII（LA 基线对比） | 新增 | R3 |
| 新 Table VIII（TTFA 预算） | 新增 | R6 |
| §IV 新增小节（append-only 不变式 + 测量） | 新增文字 + 两个数字 | R4 |
| §V 新增小节（语义一致性） | 新增文字 + 数字 + 1 个 case | R5 |

### 8.2 回复信证据清单

在 `REVISION_CHANGELOG.md` 中按"意见 i → 修改位置 → 支撑数据文件 → 关键数字"逐条登记，确保回信每个论断都有文件可指。

### 8.3 论文文字修改点（提前列好，改稿时照单打勾）

1. §I / Abstract：补 TTFA 口径一句话（配合 R6 数字），避免"voice latency"与 TTFT 口径错位；
2. §III-C：指标定义补 TTFA 定义与端点等待说明；
3. §IV-B/IV-C：显式陈述"committed text immutable → incremental prefill append-only → 无回滚"不变式，补 R4 两个测量段落；
4. §V-A：实验设置补真实语音集描述（来源、构建、规模、许可）、增强条件、重复测量协议；
5. §V 新增：R2/R3/R5/R6 各一小节；Table III–V 替换；
6. §VI Limitations：删去"TTS 合成音频"一条（已被 R2 覆盖），改为"独占 GPU 环境/单机部署"等仍然成立的限制；Future Work 中"语义一致性指标"一条相应更新（已被 R5 部分覆盖）。

---

## 九、执行顺序与工期（双机并行）

**本机轨道（CPU/网络 + 本机 3060 小模型冒烟，与 GPU 轨道并行）：**

```
Day 1     R1.1 离线重算分位数（当天出第一批数字）＋ 生成 repeat_subset_ids.json、
          exp2_ablation_sample_list.json，commit + push 供 GPU 机 pull
Day 1–3   DEV-1/2/3/4/5 程序开发，用 .env 默认小模型（whisper-tiny + Qwen2.5-0.5B）
          在本机冒烟通过后 commit + push（GPU 机随时 pull 即可开工 E1/E4/E5/E3）
Day 1–3   （并行）R2 数据下载（LibriSpeech/AISHELL-1/MUSAN）＋ 构建 ＋ QA
          （System A WER sanity 用本机 3060 跑 Whisper-Turbo）＋ 增强变体，
          打包经 git 之外的渠道传 GPU 机
Day 3–4   编写离线分析脚本（score_wer_offline / check_tokenizer_seams / 语义评估）
Day 5+    GPU 机结果陆续回传后：R4 接缝分析、R5 嵌入+裁判评估、R6 预算汇总、R7 绘图与证据清单
```

**GPU 机轨道（按 `GPU_EXPERIMENT_HANDOFF.md` 执行，全新 Ubuntu 22.04）：**

```
Day 0–1   handoff §1.0 环境初始化（系统包/NVIDIA 驱动/uv/clone/uv sync/模型预下载，
          约 25 GB 下载）＋ CosyVoice 服务部署——无需等待本机，可立即开始
Day 1     原始数据包到达 → §1.1 数据核验 → 版本存档
Day 1–2   pull 到本机推送的 DEV 代码后按 §3 复验冒烟；E1 重复测量（等 repeat 清单）
Day 2–4   E2 真实语音运行（等本机真实语音数据包到达后开始；干净集 → 增强集）
Day 4–6   E3 LA 基线运行 → E4 R4+R5 合并复跑
Day 6–7   E5 端点等待测量 ＋ E6 TTS 首包测量（CPU/网络，可与 GPU 运行穿插）
Day 7+    按 §5 清单回传全部结果
```

GPU 纯运行时间约 21–28 小时（仿真为实时节奏，见 handoff §4 时间估算表），另有环境初始化约 1–1.5 天；两机并行下**总工期约 8–10 个工作日**。

## 十、风险与回退

| 风险 | 影响 | 回退方案 |
|---|---|---|
| 原始数据在传输中损坏/缺项 | GPU 机无法开工 | 本机保留完整副本可重传；GPU 机按 handoff §1.1 核验（数量 + 抽验）把关 |
| GPU 机全新环境初始化踩坑（驱动版本、uv sync 大下载失败、模型下载受阻） | Day 0–1 延期 | handoff §1.0 已给出驱动版本要求与 hf-mirror 备选；逐项重试，网络问题换镜像 |
| CosyVoice 服务在新机器上无法恢复部署 | R6.2 无法测首包 | 仅影响 E6：改用其他流式 TTS 并在论文中如实写明型号；预算表仍成立。若镜像仍在本地备份，优先恢复 |
| Judge API（qwen-max/gpt-4o）无 key | R5 轨道 B 缺失 | 降级为双嵌入模型交叉验证（bge-m3 + 另一中英模型），回信说明 |
| R4 插桩发现 correction event 显著非零 | "无回滚"叙事需软化 | 如实报告频率与编辑距离，改叙事为"极低频漂移且不影响下游（引 R5 证据）" |
| LA 基线实现进度超期 | R3 延期 | 裁剪为只在 Long/Very Long 两组各 50 条上运行；Extra Long 可省 |
| GPU 时间被占用 | 整体延期 | R2 增强条件砍半（只保 SNR 15 一档 + 0.9× 一档），先保干净集 |
