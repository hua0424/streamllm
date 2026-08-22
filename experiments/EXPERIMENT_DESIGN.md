# 实验设计方案：级联式语音对话系统的延迟优化

本文档用于规划硕士学位论文《级联式语音对话系统的延迟优化》的实验部分。实验旨在验证流式处理（Streaming）与 KV 缓存预填充（KV Cache Prefill）技术在降低长语音交互延迟方面的有效性。

## 一、 实验目录结构

```
experiments/
├── datasets/                     # 数据集相关
│   ├── raw_data/                 # [只读] 原始数据集存放位置
│   │   ├── MultiWOZ/             # 英文多轮对话数据集 (文本)
│   │   └── CrossWOZ/             # 中文跨域对话数据集 (文本)
│   ├── processed/                # [生成] 实验用数据 (由 tools 生成)
│   │   ├── json/                 # 任务描述文件 (含文本、时长等元数据)
│   │   │   ├── crosswoz/         # CrossWOZ 处理后的 JSON
│   │   │   └── multiwoz/         # MultiWOZ 处理后的 JSON
│   │   └── audio/                # 生成的音频文件 (.wav)
│   │       ├── crosswoz/
│   │       └── multiwoz/
│   └── tools/                    # 数据处理工具代码
│       ├── data_processor.py     # 数据预处理模块
│       ├── run_pipeline.py       # 数据处理管线主程序
│       ├── tts.py                # TTS 客户端 (支持多并发)
│       ├── doc/                  # 工具使用文档
│       │   ├── PIPELINE_USAGE.md
│       │   └── TTS_USAGE.md
│       └── scripts/              # 运行脚本
│           └── run_data_pipeline.sh
├── scripts/                      # [待开发] 实验运行脚本
│   ├── run_exp_latency.py        # 实验一：延迟与长度关系
│   ├── run_exp_ablation.py       # 实验二：消融实验
│   └── run_exp_quality.py        # 实验三：准确率验证
└── results/                      # [自动生成] 实验结果数据与图表
    ├── latency_logs/
    ├── ablation_tables/
    └── quality_metrics/
```

## 二、 实验环境与基准

### 2.1 待测系统定义

为了控制变量，我们将对比以下两种系统配置：

| 系统标识 | 系统名称 | 配置描述 | 关键技术 |
| :--- | :--- | :--- | :--- |
| **System A** | **Baseline (非流式基线)** | 完整音频录制 -> 完整ASR转录 -> 完整Prompt送入LLM | 传统的级联架构，无流水线并行 |
| **System B** | **Ours (流式优化方案)** | VAD分段 -> 流式ASR (上下文感知) -> 流式LLM (KV缓存增量预填充) | **Streaming ASR + LLM KV Cache Prefill** |

### 2.2 核心评价指标

1.  **TTFT (Time to First Token)**: 首字延迟。定义为从**语音输入结束**到**LLM生成第一个Token**的时间差。这是衡量用户"等待感"的核心指标。
2.  **WER/CER (Word/Character Error Rate)**: 词/字错误率。用于衡量 ASR 转录的准确性。
3.  **Latency Improvement**: 延迟优化率。计算公式：$(TTFT_A - TTFT_B) / TTFT_A \times 100\%$。

---

## 三、 数据准备

### 3.1 数据筛选策略

为了获取足够长的语音输入数据用于实验，我们采用以下策略：

1.  **筛选长对话**：计算每个对话的总文本长度（所有轮次累积），按长度降序排序，选取前 100 条对话
2.  **累积对话生成**：对每个对话的每个用户轮次，生成累积的对话文本
3.  **过滤过长文本**：由于过长的实验数据意义不大，且中英文语速不同，因此设置文本长度上限，超过上限的样本将被跳过

### 3.1.1 文本长度限制

| 语言 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 中文 (CrossWOZ) | `--max-text-length-zh` | 720 | 约对应 150 秒音频 |
| 英文 (MultiWOZ) | `--max-text-length-en` | 2050 | 约对应 150 秒音频 |

**估算依据**：
- 中文语速约 4-5 字/秒（720 字 ÷ 4.8 字/秒 ≈ 150 秒）
- 英文语速约 13-14 字符/秒（2050 字符 ÷ 13.7 字符/秒 ≈ 150 秒）

### 3.2 累积对话逻辑

| 轮次 | 发言者 | 原始文本 | 累积输出 | 输出时机 |
|------|--------|----------|----------|----------|
| 1 | 用户 (a1) | "你好，推荐一个景点" | "你好，推荐一个景点" | ✓ 生成 turn1 |
| 2 | 系统 (b1) | "推荐颐和园" | - | 累积但不输出 |
| 3 | 用户 (a2) | "门票多少钱" | "你好，推荐一个景点 推荐颐和园 门票多少钱" | ✓ 生成 turn2 |
| 4 | 系统 (b2) | "30元" | - | 累积但不输出 |
| 5 | 用户 (a3) | "怎么去" | "你好... 门票多少钱 30元 怎么去" | ✓ 生成 turn3 |

通过这种方式，随着 turn 增加，输入文本长度递增，生成的音频时长也随之增加，便于分析不同长度输入对系统延迟的影响。

### 3.3 数据处理管线

数据处理分为三个阶段：

**阶段1：数据预处理 (Data Preprocessing)**
- 读取原始数据集 JSON 文件
- 按总文本长度筛选前 N 个对话
- 生成累积对话格式的任务 JSON 文件

**阶段2：TTS 音频生成 (TTS Batch Processing)**
- 调用 CosyVoice TTS 服务
- 多并发处理提高效率（默认 4 并发）
- 生成 WAV 格式音频文件（22050Hz）

**阶段3：更新音频时长 (Update Audio Duration)**
- 读取生成的音频文件
- 获取实际时长，写入 JSON 文件

### 3.4 输出数据格式

每个样本对应一个 JSON 文件：

```json
{
  "sample_id": "crosswoz_391_turn3",
  "dialog_id": "391",
  "turn_index": 3,
  "text": "累积的对话文本...",
  "text_length": 256,
  "audio_file": "crosswoz_391_turn3.wav",
  "audio_duration": 18.52,
  "language": "zh",
  "dataset": "crosswoz"
}
```

### 3.5 数据分组

根据 `audio_duration` 字段，将样本分为以下组别用于实验：

| 分组 | 时长范围 | 典型轮次 | 用途 |
|------|----------|----------|------|
| 短语音 | < 5s | turn1 | 基准测试 |
| 中等语音 | 5-15s | turn2-3 | 主要对比 |
| 长语音 | 15-30s | turn4-6 | 效果验证 |
| 超长语音 | > 30s | turn7+ | 极限测试 |

---

## 四、 详细实验设计

### 实验一：延迟与语音长度的关系验证 (Effect Validation)

**实验目的**：
验证本项目的核心假设：随着语音输入长度的增加，流式方案 (System B) 的 TTFT 保持相对稳定，而非流式方案 (System A) 的 TTFT 呈线性增长。同时寻找"交叉点" (Crossover Point)，即流式方案开始优于非流式方案的音频时长阈值。

**数据准备**：
1.  使用数据处理管线生成的音频样本
2.  根据 `audio_duration` 分组（短/中/长/超长）
3.  每组至少 50 个样本

**执行步骤**：
1.  对每个时长组的音频，分别运行 System A 和 System B
2.  记录每次运行的 `audio_duration` 和 `ttft`
3.  统计每个组的平均 TTFT 和标准差

**预期结果**：
- 绘制折线图：X轴为音频时长，Y轴为 TTFT
- System A 的曲线应随 X 轴线性上升
- System B 的曲线应趋于平缓（主要取决于最后一段音频的处理时间，而非总时长）

### 实验二：消融实验 (Ablation Study)

**实验目的**：
量化分解"流式 ASR"和"LLM 流式缓存"两个模块对整体延迟优化的贡献度。

**实验设置**：
选取实验一中效果最明显的"长语音组" (例如 15s-30s) 进行测试。对比以下三种配置：

1.  **Baseline**: 非流式 ASR + 非流式 LLM
2.  **Ablation 1 (Only Streaming ASR)**: 流式 ASR + 非流式 LLM (LLM 等待最终完整文本，不预填充 KV Cache)
3.  **Ours (Full)**: 流式 ASR + 流式 LLM (KV Cache 预填充)

**分析方法**：
- **流式 ASR 贡献**: $TTFT_{Baseline} - TTFT_{Ablation1}$
- **KV 缓存贡献**: $TTFT_{Ablation1} - TTFT_{Ours}$

### 实验三：准确率与质量边界实验 (Accuracy & Quality)

**实验目的**：
验证流式处理是否牺牲了识别准确率。流式 ASR 由于缺乏未来上下文 (Future Context)，理论上准确率略低于整句识别。本实验旨在量化这种差异，证明其在可接受范围内。

**执行步骤**：
1.  选取包含 200 条不同内容的音频测试集
2.  **ASR 准确性**：
    *   运行 System A (Whisper 完整识别)，记录转录文本 $T_{offline}$
    *   运行 System B (Whisper 流式识别)，记录转录文本 $T_{streaming}$
    *   以原始数据集的 Ground Truth 文本为基准，分别计算 System A 和 B 的 WER (英文) 和 CER (中文)
3.  **LLM 语义一致性 (可选扩展)**：
    *   比较 System A 和 System B 生成的最终回复 $R_A$ 和 $R_B$
    *   计算 BERTScore 或语义相似度，确保 ASR 的细微差异没有导致 LLM 回复产生幻觉或逻辑错误

**文本归一化**：
计算 WER/CER 前，会对参考文本和识别文本进行标点符号归一化处理：
- 移除中英文标点符号（如 `，。！？` 和 `,.!?` 等）
- 原因：流式 ASR 使用词级时间戳匹配输出，标点符号可能落在音频段边界而丢失；非流式 ASR 直接使用完整转录文本，包含标点符号
- 这是语音识别评估的业界标准做法，确保流式与非流式模式的公平比较
- 归一化后的指标仅反映**内容准确率**，不包含标点预测的评估

---

## 五、 实施路线图

### 5.1 数据准备 ✅ 已完成

- [x] 实现数据预处理模块 (`data_processor.py`)
- [x] 实现数据处理管线 (`run_pipeline.py`)
- [x] 实现 TTS 客户端，支持多并发 (`tts.py`)
- [x] 编写使用文档和运行脚本

**运行方式**：
```bash
conda activate streamllm

# 完整管线（预处理 + TTS + 更新时长）
uv run python -m experiments.datasets.tools.run_pipeline

# 使用 8 并发加速 TTS
uv run python -m experiments.datasets.tools.run_pipeline --tts-workers 8

# 仅预处理（跳过 TTS）
uv run python -m experiments.datasets.tools.run_pipeline --skip-tts

# 使用自定义文本长度限制（默认已启用：中文720，英文2050）
uv run python -m experiments.datasets.tools.run_pipeline \
    --max-text-length-zh 300 \
    --max-text-length-en 800 \
    --skip-tts
```

### 5.2 实验脚本开发 ✅ 已完成

- [x] 基于 `src/run_test_simple.py` 改造，编写批量测试脚本
- [x] 实现自动记录日志到 CSV/JSON 的功能
- [x] 实现按时长分组运行实验的功能
- [x] 添加 ASR 流式参数控制 (`--prefix-segments`, `--suffix-segments`, `--recognition-threshold`)
- [x] 实现 WER/CER 计算时的文本归一化（标点移除）

**运行方式**：
```bash
# 实验一：延迟与语音长度关系
./experiments/scripts/run_exp_latency.sh full

# 实验二：消融实验
./experiments/scripts/run_exp_ablation.sh full

# 实验三：准确率验证
./experiments/scripts/run_exp_quality.sh full
```

### 5.3 执行与分析（历史章节，已由归档结果覆盖）

> **状态（2026-08-22）**：原计划三项均已完成——完整实验在两平台执行并归档于
> `experiments/results/exp1_latency|exp2_ablation|exp3_quality/`（只读）；
> 审稿修订补充实验见 §六（R1–R7）；论文图表由 `results/revision/fig/` 与各 R 目录提供。
> 本小节保留为原始设计的历史记录，其"待完成"状态不再代表项目现状。

- [x] 在 GPU 环境下运行完整实验（已完成并归档）
- [x] 收集实验数据（已完成并归档）
- [x] 使用 Python 绘制论文所需的图表（Fig.6 重绘于 `results/revision/fig/Fig6.pdf`）

---

## 六、 CISR 审稿修订实验（2026-08 新增）

本章登记审稿意见驱动的补充实验。执行细则以 `experiments/CISR_REVISION_PLAN.md`（总方案）与
`experiments/GPU_EXPERIMENT_HANDOFF.md`（GPU 机任务书）为准；本节保证方法学定义的唯一权威来源。

### 6.1 R1：统计稳健性（意见3）

- **分位数重算**：`experiments/scripts/recompute_stats.py` 离线重算 exp1/2/3 的
  mean±std/P50/P90/P95/P99/min/max（numpy 线性插值）。过滤规则显式化：成对排除运行错误样本、
  流式模式 TTFT>10000ms 判定平台挂起；Table III 的排除明细写 `r1_stats/table3_filter_manifest.json`，
  Table IV 重算前强制三模式配对完整性校验（缺模式/重复/错误/时长不一致即报错退出）。
- **重复测量**：Very Long 组固定 50 样本（`r1_stats/repeat_subset_ids.json`，seed=42）连跑 3 轮，
  逐样本计算 TTFT 变异系数 CV。
- **Table V 历史口径**：论文 Table V 的 "ASR time" 列为 summary CSV 中 streaming 与 non-streaming
  合并（300 行）的 `asr_time_ms` 均值（1327.48/1224.96/1086.16，与归档精确一致）；重算 CSV 中
  以 `pooled` 行复核该口径，另提供 streaming-only 明细。

### 6.2 R2：真实语音验证（意见1）

- 数据源：LibriSpeech test-clean/test-other（CC BY 4.0）与 AISHELL-1（Apache 2.0），
  按原实验相同的拼接策略构造 Long/Very Long/Extra Long 样本（各集 75 条：30/30/15）。
- 构建脚本（2026-08-18 定稿，GPU 机执行，handoff §4-E2-0）：
  `experiments/scripts/build_real_speech_set.py` —— 同章节（LibriSpeech）/同说话人
  （AISHELL-1）顺序拼接，句间随机静音 U(0.2, 1.0)s，seed=42 全流程确定性可重建；
  分组区间 long 15-30s / very_long 30-60s / extra_long 60-150s（与 DURATION_GROUPS 口径一致）；
  AISHELL 转写去字间空格；逐条 QA（时长重读 ±50ms、文本非空、RMS 下限）+ 构建 manifest。
- QA 脚本：`experiments/scripts/qa_real_speech.py` —— 静态校验 + Whisper 转写 sanity
  （镜像 System A 解码参数 beam=5/temperature=0，复用 exp3 的 WER/CER 归一化），
  干净集错误率 ≤10% 为验收线；冒烟 `test_r2_build_smoke.py`（伪造迷你语料 16/16）。
- 增强变体：`experiments/scripts/build_augmented_variants.py` —— MUSAN noise 按 SNR 20/15/10 dB
  混合（babble 为 speech 子集可选项）、变速 0.9×/1.1×（librosa time_stretch 保音高，
  时长与分组同步重判）；每变体抽 30 条（优先 long+very_long），seed 确定性。
- 运行：`run_exp_latency.py --dataset librispeech|aishell1|<变体>`（数据集目录扫描已通用化），
  配置与合成集锁定值一致（prefix=1, suffix=0, threshold=2.0s）。

### 6.3 R3：LocalAgreement-2 基线（意见4）

- 策略出处 ufal/whisper_streaming，同引擎自实现（`src/asr/local_agreement_streamer.py`）：
  模型加载/转录参数/分段器与 System A/B 完全一致，唯一变量是 ASR 上下文与提交策略。
  提交规则：相邻两轮假设的最长公共前缀中，`word.end ≤ 当前音频时长 − trailing_margin(=0)` 的词。
- 运行：`experiments/scripts/run_exp_baseline_la.py`，样本为 exp2 干净成对子集 498 条
  （`r3_baseline_la/exp2_ablation_sample_list.json`；逐样本数据源 `exp2_ablation/exp2_gains_clean.csv`，
  排除明细 `exp2_gains_exclusions.csv`）。
- 交付：TTFT 与 WER/CER（复用 run_exp_quality 归一化逻辑；空转写标记 `asr_no_text`）；
  LA 假设在已提交位置前分歧的事件数（`divergence_count`）一并记录。

### 6.4 R4/R5：append-only 观测与语义一致性（意见5）

- append-only 不变式：已提交段文本快照 `committed_text`，后续轮次重识别漂移记入
  `correction_events`；每次提交写 `commit_log.jsonl`（type=commit/correction）。
- 分词接缝分析：离线对比增量拼接分词与一次性分词的 token 序列一致性。
- 语义一致性：bge-m3 嵌入相似度 + LLM-as-judge（本机离线）。

### 6.5 R6：TTFA 端到端预算（意见2）

- 时间定义（E5，`--append-silence-ms 2000`）：
  - `speech_end_time`：最后一块真实（非拼接静音）音频块按实时节奏推送完的时刻；
  - `final_speech_segment_commit_time`：最后一个含语音段（VAD 闭段；无尾静音时为含语音的 flush 段）
    进入 `audio_segment_queue` 的时刻。**论文端点等待采用**：
    `endpoint_detection_wait = final_speech_segment_commit_time − speech_end_time`；
  - `final_is_final_segment_enqueue_time`：flush 产生的 `is_final=True` 段入队时刻（审计用，
    `final_enqueue_wait = final_is_final_segment_enqueue_time − speech_end_time`）。
- TTS 首包（E6）：`measure_tts_first_chunk.py` 测 TTFC 与 RTF；输入为 E4 的 `full_response`
  （缺 `full_response` 默认报错，preview fallback 需显式开启并在 CSV 标记 `text_source=preview`）；
  请求参数（spk_id/speed/PCM 解释）经 env/CLI 可配并写 RUNINFO。
- TTFA 预算：`TTFA = endpoint_wait + TTFT + T_decode_to_first_sentence + T_TTS_first_chunk`（T_Net≈0 单机部署）。

### 6.6 R7：TTFA 统一时间轴实测（W1，2026-08-22 定稿）

替代 §6.5 跨运行装配的 TTFA 预算口径（E5/E6/E4+补测装配稿已按 PRE-PAPER-AUDIT P0-1 作废，
不得与 R7 数据混用）。方法学定义以本节为唯一权威。

- **脚本**：`experiments/scripts/run_ttfa_unified.py`（单进程统一 `perf_counter_ns` 时间轴；
  PSE 能量+Silero 双仲裁、因果块回放、System A/B 成对执行、append-only checkpoint、
  fail-closed 守卫：探活/Silero hash/平台文件 hash/样本清单 hash/计划 hash 全绑定后才开跑）。
- **正式 run（r7_main）**：50 样本（very_long，zh/en 各 25，seed=42 清单）×2 模式（AB/BA
  分层平衡 25/25）+ 10 子集（每语种 load 序前 5）×2 补轮 = **140 任务**；max_tokens=128、
  chunk 500ms、prefix 1、suffix 0、threshold 2.0s、TTS 晓伊→内置中文女映射（注记入 RUNINFO）。
- **TTFA 定义**：`first_playable_pcm_ns − physical_speech_end_ns`（首个 ≥1324B 可播 PCM 相对
  真实语音结束）；组件链（speech_end→feed_end→input_close→first_token→text_ready→tts_req→
  first_pcm→playable）逐条闭合校验（`validate_record` 因果偏序）。⚠️ 分项标签（2026-08-22
  复审修正）：第二分项论文标签 **t_feed_to_close_wait**（= pipeline_input_close − feed_end，
  喂入结束→管线输入关闭）；checkpoint/summary 字段 `t_flush_to_close` 为历史命名，该 ~133ms
  为完整等待（flush 段自身仅 ~0.33ms），**不得归因为 flush 计算开销**。
- **Table VIII 装配层决策（W8 阶段 2，`experiments/scripts/assemble_table_viii.py` 唯一实现）**：
  repeat0、n=50/模式；mean/std(ddof=1)/P50/P90/P95（np.percentile 线性插值）；单位 ms 1 位小数；
  `ttfa_received` 仅 QA 补充（|received−playable| 入装配 QA 不进主表）；tts_control 7076ms
  只作归因/回信证据；旧 ttfa_budget.csv 估计项完全排除；装配 QA 四项（逐记录闭合恒等、
  received 差、与运行侧 summary 双入口对拍、checkpoint 哈希固定）。
- **匹配文本 TTS 控制（r7_tts_control）**：`--tts-control-only`，输入主 checkpoint；repeat-0
  完整配对中 sorted zh 前 5+en 前 5，每样本 B 首句/A 首句/A 全文 ×1 + 中英校准 2 = 32 调用。
- **数据状态（2026-08-22 审查终裁）**：r7_main **通过**（140/140、QA 0、结果级复核 47/47）；
  r7_tts_control 数据级通过，**流程偏差豁免采信**（提前执行、不构成追认、不重跑），
  登记见 `experiments/review/20260821-PRE-PAPER-AUDIT/deviation-waiver-r7-tts-control-20260822.md`。
- **治理铁律（裁定 §4-6）**：今后任何后置实验必须取得**独立书面放行**，不得由
  "前一阶段已完成"推断授权延伸；`r7_main/`、`tts_control/` 为只读归档，
  论文表格只可引用已过审的 R7 数字，不得引用作废装配稿或未 QA 估计值。

### 6.7 确定性 CPU 重分析（2026-08-22）

- `experiments/scripts/cpu_revision_analysis.py` 只读锁定的 JSON/CSV/JSONL 与 processed metadata；
  不运行 ASR/LLM/TTS/CUDA，不读取音频，也不外推修正后触发策略的性能。
- 通过 `sample_id` 唯一连接 `(dataset, dialog_id)`；主估计保持 turn-weighted 均值差及
  ratio-of-means 改善率，95% CI 采用 dialogue-cluster percentile bootstrap（10,000 次，
  base seed=20260821，并按比较名派生稳定 seed）。检验先在每个 dialogue 内取配对差均值，
  再做双侧 Wilcoxon；预先命名的 comparison family 内使用 Holm 校正。
- 样本流固定为 505 candidate − 7 个经 run log 复核受并发外部程序污染的 execution =
  498 个 valid complete three-arm 样本；污染运行只保留在审计 ledger，不作为系统结果分析。
  R7 描述统计仅使用 repeat-0，且明确区分 System A capped full response 与 System B first
  sentence 的 TTS 文本策略。
- 输入哈希、1,133 metadata IDs、498/99 cluster 结构、TTFA 50×2 主记录及历史汇总均
  fail closed；输出固定排序且不写生成时间，归档于 `results/revision/minimal_cpu_reanalysis/`。
