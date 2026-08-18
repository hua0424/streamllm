# GPU 实验机执行任务书（CISR 修订补充实验）

> **读者**：在 GPU 实验机上执行本任务的人员或 agent。
> **目标**：完成 CISR 审稿意见所需的全部 GPU 侧补充实验，按规定格式产出结果并回传。
> **配套文档**：`experiments/CISR_REVISION_PLAN.md`（总方案）、`experiments/EXPERIMENT_DESIGN.md`（原实验设计）。
> **机器情况**：全新安装的 Ubuntu 22.04，2×RTX 3090，**盘内无任何历史数据**——按 §1.0 从零初始化环境，原始实验数据由需求方从本机打包传来（§2）。
> **前置共识**：所有实验必须复用论文原配置（见 §1.3 配置锁定表），任何参数变更都必须先与需求方确认。

---

## 〇、背景与交付目标

论文（CISR 会议投稿）收到 5 条审稿意见，需要补充以下证据。GPU 机负责产出其中所有需要模型运行的实验数据：

| 编号 | 目的 | 对应审稿意见 |
|---|---|---|
| E1 | 50 样本 × 3 轮重复测量（TTFT 稳定性） | 意见3 |
| E2 | 真实语音（LibriSpeech/AISHELL-1 + 噪声/变速增强）上的 A/B 对比 | 意见1 |
| E3 | LocalAgreement 流式基线对比 | 意见4 |
| E4 | 插桩复跑：提交分歧日志 + 完整回复（供分词接缝分析与语义一致性评估） | 意见5 |
| E5 | 端点等待测量（尾部拼静音） | 意见2 |
| E6 | TTS 首包延迟测量（CPU/网络，穿插进行） | 意见2 |

全部结果写入 `experiments/results/revision/` 下对应子目录（**不得写入或改动 `experiments/results/` 下已有的 exp1/exp2/exp3 目录**），完成后按 §5.4 清单回传。

---

## 一、Day-0 核验清单（运行任何实验前逐项完成）

### 1.0 全新 Ubuntu 22.04 环境初始化（最先做）

本机为刚重装的全新系统，按以下顺序从零初始化：

```bash
# 1) 系统包
sudo apt update && sudo apt install -y git curl ffmpeg libsndfile1 build-essential

# 2) NVIDIA 驱动（torch cu121 轮子要求驱动 ≥ 530.30；CUDA 运行库由 pip 的
#    nvidia-* 包自带，无需另装 CUDA Toolkit）
sudo ubuntu-drivers install        # 或 sudo apt install -y nvidia-driver-550
sudo reboot
nvidia-smi                          # 应看到 2×RTX 3090

# 3) uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env         # 或按安装输出的提示操作

# 4) 克隆仓库（默认分支 main；本机无旧数据，全新 clone 即正确做法）
git clone https://github.com/hua0424/streamllm.git
cd streamllm

# 5) 安装依赖（uv 自动创建 .venv，安装 torch cu121 等，下载约 8 GB）
uv sync

# 6) .env 核对（.env 被 git 跟踪，clone 后已存在）
#    确认：ASR_MODEL_NAME=turbo、LLM_MODEL_NAME=Qwen/Qwen2-7B-Instruct
#    HF_HOME 指向存在的目录；网络受限时设 HF_ENDPOINT=https://hf-mirror.com

# 7) 模型预下载（共约 17 GB；也可留待首次运行自动下载，但预先拉取便于及早发现网络问题）
uv run python -c "import whisper; whisper.load_model('turbo')"                          # ~1.6 GB
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-7B-Instruct')"  # ~15 GB

# 8) 【仅 E6 需要，可延后】Docker + CosyVoice TTS 服务
#    部署方式见 experiments/datasets/tools/doc/TTS_USAGE.md；
#    服务地址应与 tts.py 默认一致（host.docker.internal:20401）。
#    若原镜像/部署方式无法恢复：记录现象并回告需求方，不要自行更换 TTS。
```

说明：**本任务全程不需要在 GPU 机上 commit/push**——代码由需求方在本机开发并推送到 main，你侧只 `git pull`；实验结果按 §5.4 打包经文件渠道回传。

### 1.1 数据完整性核验（数据包到达后执行）

原始 1,132 条合成实验数据**不在 git 中**（`experiments/datasets/processed` 被 .gitignore 排除），由需求方从本机打包传来（约 3.4 GB，`processed/` 整目录，含 multiwoz 630 + crosswoz 503 的 JSON 与 WAV；crosswoz 比论文计数多 1 条为运行时失败样本，属正常）。解压放置到仓库的 `experiments/datasets/processed/`，然后：

```bash
# 期望：multiwoz 630 个 JSON + 630 个 WAV；crosswoz 503 个 JSON + 503 个 WAV
# （crosswoz 比论文计数 502 多 1 条：该样本原实验运行时失败，不参与任何清单，属正常）
ls experiments/datasets/processed/json/multiwoz | wc -l
ls experiments/datasets/processed/audio/multiwoz | wc -l
ls experiments/datasets/processed/json/crosswoz | wc -l
ls experiments/datasets/processed/audio/crosswoz | wc -l
```

再随机抽 15 个样本校验 JSON 的 `audio_duration` 与 WAV 实际时长一致（容差 50 ms）、文本非空。可用下述一行脚本：

```bash
uv run python - <<'EOF'
import json, wave, random
from pathlib import Path
root = Path('experiments/datasets/processed')
bad = total = 0
for ds in ['multiwoz', 'crosswoz']:
    files = sorted((root/'json'/ds).glob('*.json'))
    random.seed(42); random.shuffle(files)
    for jf in files[:15]:
        total += 1
        d = json.load(open(jf, encoding='utf-8'))
        af = root/'audio'/ds/d['audio_file']
        if not af.exists(): print('MISSING AUDIO', af); bad += 1; continue
        with wave.open(str(af), 'rb') as w:
            dur = w.getnframes()/w.getframerate()
        if abs(dur - d['audio_duration']) > 0.05 or not d['text'].strip():
            print('BAD', jf.name, d['audio_duration'], round(dur, 3)); bad += 1
print(f'checked {total}, problems {bad}')
EOF
```

**若数量或抽验不通过：停止，联系需求方，不要自行补齐或重新合成。**

### 1.2 环境核验

- [ ] `nvidia-smi` 确认双 RTX 3090 可用（§1.0 第 2 步已装驱动）。
- [ ] `.env` 中 `ASR_MODEL_NAME=turbo`、`LLM_MODEL_NAME=Qwen/Qwen2-7B-Instruct`。
- [ ] 模型预下载已完成（§1.0 第 7 步）；用一个 5 秒音频试跑一次确认可加载。
- [ ] CosyVoice TTS 服务探活（仅 E6 前必须完成）：先读 `experiments/datasets/tools/doc/TTS_USAGE.md`，按其说明检查 `http://host.docker.internal:20401` 可达；不可达则记录现象并通知需求方，不自行换 TTS。
- [ ] 版本存档（回信要用）：

```bash
mkdir -p experiments/results/revision
uv run pip list > experiments/results/revision/env_versions.txt
nvidia-smi >> experiments/results/revision/env_versions.txt
```

### 1.3 配置锁定表（所有运行必须与论文原实验一致）

从历史结果 JSON 的 config 块核实，论文实验配置为：

| 参数 | 锁定值 | 备注 |
|---|---|---|
| ASR 模型 | `turbo`（openai-whisper large-v3-turbo） | 设备 `cuda:0` |
| LLM 模型 | `Qwen/Qwen2-7B-Instruct` | 设备 `cuda:1` |
| chunk_duration | 500 ms | |
| prefix / suffix segments | **1 / 0** | ⚠️ 脚本默认 suffix=1，**必须显式传 `--suffix-segments 0`** |
| recognition_threshold | 2.0 s | |
| 解码参数 | temperature 0.1, top_p 0.9, repetition_penalty 1.1 | |
| max_tokens | 50（仅 E4 用 128） | |
| warmup | 3 轮真实音频 | 每次脚本启动自动执行，不要跳过 |

---

## 二、等待本机传入的外部产物

代码类产物（DEV-1~5）**不经文件传输，全部由本机开发、冒烟后 push 到 `origin/main`，你 `git pull` 获取**（见 §1.0）。非代码产物如下，**收到后再开始对应实验**；如约定时间未收到，先做不依赖它们的任务，不要自行生成替代品：

| 文件 | 用途 | 获取方式 | 阻塞的任务 |
|---|---|---|---|
| **原始合成数据集包**（约 3.4 GB，`processed/` 整目录） | 全部合成集实验 | 文件传输（scp/rsync/网盘/U盘），解压为仓库下 `experiments/datasets/processed/`，验收见 §1.1 | E1、E3、E4、E5 |
| DEV-1~5 全部源码 | E1–E6 运行 | `git pull origin main`（本机推送后通知你） | E1、E3、E4、E5、E6 |
| `repeat_subset_ids.json` | 固定 50 样本清单（Very Long 组） | 文件传输，放 `experiments/results/revision/r1_stats/` | E1、E4、E5 |
| `exp2_ablation_sample_list.json` | 消融干净成对子集 498 条（排除规则见文件内 metadata；Table IV 按此口径重算） | 文件传输，放 `experiments/results/revision/r3_baseline_la/` | E3 |
| 真实语音数据包 | `processed/json|audio/{librispeech,aishell1}` 及增强变体目录 | 文件传输，解压到 `experiments/datasets/processed/` 对应子目录 | E2 |

**立即可做（不依赖任何本机输入）**：§1.0 环境初始化（系统包、驱动、uv、clone、uv sync、模型预下载）与 CosyVoice 服务部署。原始数据包、清单文件、DEV 代码到位后按上表逐项解锁。

数据包验收：每个数据集目录应有"等数量"的 JSON 与 WAV；期望规模：librispeech 75 条、aishell1 75 条（Long 30 / Very Long 30 / Extra Long 15），增强变体每个目录 30–60 条不等（以交接说明为准）。随机抽 3 条试听或查看波形确认非静音。

---

## 三、程序开发任务（DEV-1 ~ DEV-5）

> **开发责任已转移到本机**：以下全部代码由需求方在本机编写、用小模型（whisper-tiny + Qwen2.5-0.5B）完成逻辑冒烟后 push 到 `origin/main`。你侧的任务是：`git pull` → 对照本节规格**核对实现要点**（特别是参数锁定与新增字段）→ 用各条的冒烟用例**复验**。发现缺漏或与规格不符时，停止并回告需求方，**不要自行修改实现逻辑**；只有在需求方明确授权后，才可按本节规格补齐并提交到本地分支存档。
>
> 本节规格同时作为验收标准，阅读时重点关注：新增参数/字段清单、插桩点、算法步骤、冒烟预期。

### DEV-1：`experiments/scripts/run_exp_latency.py` 扩展

在原脚本基础上加 5 个能力（全部向后兼容，不传新参数时行为与原版完全一致）：

1. **`--sample-list <path>`**：JSON 数组（sample_id 字符串列表）。在 `load_samples()` 返回后过滤：`samples = [s for s in samples if s.sample_id in allow]`。
2. **数据集扩展**：`load_samples()` 中硬编码的 `["crosswoz", "multiwoz"]` 扩展为同时支持 `"librispeech"`、`"aishell1"` 及增强变体目录名；`--dataset` 的 choices 相应扩充（`all` = 扫描 `processed/json/` 下全部子目录）。
3. **`--append-silence-ms N`**（默认 0）：音频加载后在尾部拼接 N ms 零值静音。新增两个计时点：
   - `speech_end_time`：最后一块**真实（非静音）**音频块推送完毕的时刻；
   - `final_segment_commit_time`：分段器把 `is_final=True` 的段放入 `audio_segment_queue` 的时刻。
   两者存入结果（字段名同上）。原 `audio_end_time` 语义不变（= 含静音的全音频推送完）。
   **离线分析将用以下定义**：`endpoint_wait = final_segment_commit_time − speech_end_time`；`post_endpoint_ttft = first_token_time − final_segment_commit_time`；`total = first_token_time − speech_end_time`。
4. **`--save-full-response`**：`ExperimentResult` 增加 `full_response: str` 字段，保存完整生成文本（不受 100 字符截断），原 `response_preview` 保留。
5. **`--save-fragments`**：`ExperimentResult` 增加 `committed_fragments: List[str]`，按提交顺序记录流式模式下每一个文本片段（在 `text_queue.put((output_text, False))` 处同步收集）。
6. 顺手修复：`--asr-model-size` 的 choices 加入 `"turbo"`。

`ExperimentResult` 新字段默认值均为空，保证旧 checkpoint 可加载。

**冒烟测试**：`--dataset crosswoz --max-samples 2 --append-silence-ms 2000 --save-full-response --save-fragments`，确认结果 JSON 含新字段且数值合理（endpoint_wait 约 0.2–1.5 s 量级，取决于 TTS 尾部静音；本机 tiny 模型冒烟实测 0.18 s）。注意 `endpoint_wait = final_segment_commit_time − speech_end_time` 中的 commit 指 **VAD 关闭最后一个含语音段**的时刻（非 flush 静音残余段），实现已按此口径。

**新鲜环境注意**（本机冒烟已踩过，GPU 机若换机/清缓存会遇到）：
1. silero-vad 经 `torch.hub.load` 首次下载会交互询问信任仓库，非交互 shell 下直接 EOF 报错；预缓存：`echo y | uv run python -c "from src.asr.streamaudio_segmenter import StreamAudioSegmenter; StreamAudioSegmenter()"`。
2. `.env` 中的 HF_TOKEN 已失效（whoami 验证 401）。已缓存模型不受影响；如需新下载模型，临时置空：`HF_TOKEN= uv run ...`。

### DEV-2：`src/asr/faster_whisper_streamer.py` 提交分歧插桩

目的：实测"已提交文本在后续轮次重识别中是否漂移"（支撑"append-only、无回滚"的声明）。**只增加观测，不改变任何行为。**

1. `ASRAudioSegment` 数据类增加两个字段：`committed: bool = False`、`committed_text: str | None = None`。
2. `_extract_output_text()` 中，对每个被输出的段：置 `committed=True`、`committed_text = 该段 text`。
3. `transcribe_audio_segment()` 的逐段文本提取循环（调用 `_extract_segment_text` 处）：提取后若 `segment.committed and segment.text != segment.committed_text`，向 `self.correction_events` 追加一条记录。
4. `StreamingASRProcessor.__init__` 增加：`self.correction_events: list = []`、`self.commit_log: list = []`、`self.current_sample_id: str = ""`。在 `_extract_output_text` 输出非空文本时，向 `commit_log` 追加 `{"t": time.time(), "text": output_text, "segment_ids": [...]}`。
5. 实验脚本侧（配合 DEV-1）：每个样本测试前 `processor.current_sample_id = sample.sample_id` 并清空两个列表；测试结束后把本样本的 `commit_log` 与 `correction_events` 以 JSONL 追加写入 `<output_dir>/commit_log.jsonl`，每行格式：

```json
{"sample_id": "multiwoz_xxx_turn5", "type": "commit", "round": 3, "text": "...", "segment_ids": ["seg_004","seg_005"], "t": 1724.56}
{"sample_id": "multiwoz_xxx_turn5", "type": "correction", "segment_id": "seg_004", "old": "已提交文本", "new": "重识别文本", "t": 1731.02}
```

**冒烟测试**：同 DEV-1 冒烟运行，确认 `commit_log.jsonl` 生成且每样本至少有 commit 记录；correction 为 0 条属正常。

### DEV-3：`src/asr/local_agreement_streamer.py`（新文件，约 200–250 行）

实现 Whisper-Streaming 的 **LocalAgreement-2** 策略（策略出处：ufal/whisper_streaming，论文修改时会引用；此处为同引擎自实现）。

硬性要求：

- **模型加载必须复用** `src/asr/faster_whisper_streamer.py` 中的 `_load_whisper_model_offline_first()`，与 System A/B 完全同权重、同精度、同设备；
- 转录参数与现有 `_transcribe_segments` 完全一致：`beam_size=5, word_timestamps=True, temperature=0.0, compression_ratio_threshold=2.4, logprob_threshold=-1.0, no_speech_threshold=0.6, condition_on_previous_text=False`；
- 分段沿用同一 `StreamAudioSegmenter`，保证唯一的实验变量是 ASR 上下文/提交策略。

接口与算法：

```python
class LocalAgreementStreamer:
    def __init__(self, model_size: str, device: str, sample_rate: int = 16000,
                 decode_trigger_s: float = 2.0):
        """decode_trigger_s 与 System B 的 recognition_threshold 对齐（2.0s）"""

    def feed_segment(self, segment: ASRAudioSegment) -> list[str]:
        """喂入一个 VAD 音频段，返回本轮新提交的文本片段列表（可为空）。"""

    def flush(self) -> str:
        """流结束：提交缓冲区全部剩余文本。"""
```

算法步骤（LA-2）：

1. 维护：音频缓冲 `buffer`、已提交词数 `n_committed`、上一轮假设词序列 `prev_words`（词对象含 `text/start/end`，时间轴相对 buffer 起点）、自上次解码以来新增音频时长 `new_audio`。
2. `feed_segment`：音频追加到 buffer，`new_audio += segment.duration`；若 `new_audio < decode_trigger_s` 且非 final，直接返回空。
3. 达到触发条件或 `is_final`：对**整个 buffer** 调 `model.transcribe` 得到当前假设 `cur_words`。
4. 计算 `prev_words` 与 `cur_words` 的**最长公共前缀**（按词文本比较；中文按 Whisper 输出的词/字粒度即可）`agreed`。
5. 从 `agreed[n_committed:]` 中提交满足 `word.end <= 当前音频总时长 − trailing_margin` 的词；`trailing_margin = 最新一个 VAD 段的时长`（对应 System B 的 suffix=1 保护；若当前运行配置 suffix=0，则 `trailing_margin=0`）。
6. 新提交的词拼成文本片段返回，并更新 `n_committed`；将 `prev_words = cur_words`、`new_audio = 0`。
7. 缓冲裁剪：丢弃 buffer 中"最后提交词 end 时间 − 0.1 s"之前的音频，并把所有词时间戳相应前移（保持相对轴一致）。
8. `flush()`：提交所有未提交词，返回文本。

### DEV-4：`experiments/scripts/run_exp_baseline_la.py`（新脚本）

以 `run_exp_latency.py` 为模板复制修改：

- 仅新增一种模式 `la_streaming`：segmentation worker、audio_gen worker、llm worker（增量 `cache_prompt` 路径）完全复用；`asr_worker` 的 transcriber 循环改为：从 collector 收到段 → `la_streamer.feed_segment(seg)` → 返回的文本片段 `text_queue.put((frag, False))`；收到 final 段且队列排空后 `text_queue.put((la_streamer.flush(), True))`。
- TTFT 定义与原脚本一致（`first_token_time − audio_end_time`）。
- 复用 `SharedModels` 的 LLM 部分与预热流程；ASR 侧实例化 DEV-3 的 `LocalAgreementStreamer`（同一 cuda:0）。
- 继承 `--sample-list`、断点续传、检查点机制（直接把 DEV-1 的对应代码搬过来）。

**冒烟测试**：`--max-samples 2` 跑通，确认 LA 模式有文本提交、LLM 正常生成、结果 JSON 完整。

### DEV-5：`experiments/scripts/measure_tts_first_chunk.py`（新脚本，CPU/网络）

先读 `experiments/datasets/tools/doc/TTS_USAGE.md` 弄清服务接口。复用 `experiments/datasets/tools/tts.py` 的 `TTSClient`（其已是流式 HTTP，`stream=True` 逐块读 PCM）：

- 输入：JSONL 文件，每行 `{"sample_id": ..., "text": ..., "language": ...}`（从 E4 产出的 `full_response` 提取，中英各 25 条共 50 条，提取代码随输入文件一并由本机提供或在脚本内实现 `--from-e4 <dir>` 自动抽取）。
- 对每条：记录 `t_request` → 收到**第一块 PCM 数据**的时间 `ttfc_ms`（time-to-first-chunk）、全部收完时间 `total_ms`、音频总字节数换算时长 `audio_sec`、`rtf = total_ms/1000/audio_sec`。
- 输出 CSV：`sample_id,language,n_chars,ttfc_ms,total_ms,audio_sec,rtf`。
- 每条之间间隔 ≥ 1 s；失败条目标记 error 并继续；服务不可达则停止并通知需求方。

---

## 四、实验运行矩阵（E0–E6）

> ⚠️ **仿真为实时节奏**：`audio_gen_worker` 按 chunk 时长 sleep 模拟真实到达，这是方法学要求（流水线负载行为依赖到达节奏），**严禁为加速而删掉 sleep**。时间估算已按此给出。

每条命令中的 `$REV` = `experiments/results/revision`。所有运行统一加 `--suffix-segments 0`（配置锁定）。

### E0 冒烟（每改完一个 DEV 立即做）

各新/改脚本 `--max-samples 2` 跑通，检查 §3 中各自验收点。

### E1 重复测量（R1.2，意见3）｜约 3.5–4 GPU 小时

前置：`repeat_subset_ids.json` 已到位。

```bash
for r in 1 2 3; do
  uv run python -m experiments.scripts.run_exp_latency \
    --dataset all --sample-list $REV/r1_stats/repeat_subset_ids.json \
    --suffix-segments 0 \
    --output-dir $REV/r1_stats/repeat_r$r --no-resume
done
```

产出：3 份结果 JSON/CSV（每份 50 样本 × 2 模式）。

### E2 真实语音（R2，意见1）｜约 3 + 6–9 GPU 小时

前置：真实语音数据包已解压验收（§2）。

```bash
# E2a 干净集（150 条 × 2 模式）
uv run python -m experiments.scripts.run_exp_latency \
  --dataset librispeech --suffix-segments 0 --output-dir $REV/r2_real_speech/librispeech_clean
uv run python -m experiments.scripts.run_exp_latency \
  --dataset aishell1 --suffix-segments 0 --output-dir $REV/r2_real_speech/aishell1_clean

# E2b 增强集（每个变体目录各跑一次；以数据包实际目录名为准，例如）
for v in librispeech_snr20 librispeech_snr15 librispeech_snr10 librispeech_speed09 librispeech_speed11 \
         aishell1_snr20 aishell1_snr15 aishell1_snr10 aishell1_speed09 aishell1_speed11; do
  uv run python -m experiments.scripts.run_exp_latency \
    --dataset $v --suffix-segments 0 --output-dir $REV/r2_real_speech/$v
done
```

若 GPU 时间紧张：E2b 只保留 `*_snr15` 与 `*_speed09` 共 4 个变体，其余跳过并在 changelog 注明。

### E3 LocalAgreement 基线（R3，意见4）｜约 2.5 GPU 小时

前置：DEV-3/4 完成并冒烟通过；`exp2_ablation_sample_list.json` 已到位。

```bash
uv run python -m experiments.scripts.run_exp_baseline_la \
  --dataset all --sample-list $REV/r3_baseline_la/exp2_ablation_sample_list.json \
  --output-dir $REV/r3_baseline_la
```

### E4 插桩 + 完整回复合并复跑（R4+R5，意见5）｜约 4–5 GPU 小时

前置：DEV-1（能力 3/4/5）、DEV-2 完成并冒烟通过。

```bash
uv run python -m experiments.scripts.run_exp_latency \
  --dataset all --sample-list $REV/r1_stats/repeat_subset_ids.json \
  --suffix-segments 0 --max-tokens 128 \
  --save-full-response --save-fragments \
  --output-dir $REV/r4_commit --no-resume
```

注：本次运行样本量为 50（repeat 清单）。若需求方后续提供更大清单（最多 150），用 `--output-dir $REV/r4_commit_ext` 再跑一次。
产出：结果 JSON（含 `full_response`、`committed_fragments`）+ `commit_log.jsonl`。

### E5 端点等待测量（R6.1，意见2）｜约 1 GPU 小时

```bash
uv run python -m experiments.scripts.run_exp_latency \
  --dataset all --sample-list $REV/r1_stats/repeat_subset_ids.json \
  --suffix-segments 0 --append-silence-ms 2000 \
  --output-dir $REV/r6_ttfa/endpoint --no-resume
```

产出：含 `speech_end_time`、`final_segment_commit_time` 的结果 JSON。

### E6 TTS 首包测量（R6.2，意见2）｜约 0.5 小时（CPU/网络，可穿插）

前置：CosyVoice 服务探活通过；E4 已完成（需要其 `full_response`）。

```bash
uv run python -m experiments.scripts.measure_tts_first_chunk \
  --from-e4 $REV/r4_commit --n-zh 25 --n-en 25 \
  --output $REV/r6_ttfa/tts_first_chunk.csv
```

### 时间估算汇总

| 任务 | 时间（约） | 依赖 |
|---|---|---|
| §1.0 环境初始化 | 0.5–1 天（主要是驱动与约 25 GB 下载） | 无 |
| 原始数据包传输 + §1.1 核验 | 0.5 天（视传输渠道） | 无 |
| E1 | 3.5–4 GPU 小时 | 数据包 + repeat 清单 + DEV-1 |
| E2a | 3 GPU 小时 | 真实数据包 |
| E2b | 6–9 GPU 小时（可砍至 3–4） | 真实数据包 |
| E3 | 2.5 GPU 小时 | DEV-3/4 + 消融清单 + 数据包 |
| E4 | 4–5 GPU 小时 | DEV-1/2 + 数据包 |
| E5 | 1 GPU 小时 | DEV-1 + 数据包 |
| E6 | 0.5 小时（非 GPU） | E4 + TTS 服务 |
| **合计** | **初始化约 1–1.5 天 + GPU 运行约 21–28 小时** | |

---

## 五、结果格式、QA 与回传

### 5.1 结果格式

- 沿用现有脚本的三件套：`exp*_results_<timestamp>.json`（含 config 块 + 逐样本结果）、`exp*_summary_*.csv`、`exp*_statistics_*.csv`，输出目录一律在 `experiments/results/revision/` 下。
- 新增字段（E4/E5）：`full_response`、`committed_fragments`、`speech_end_time`、`final_segment_commit_time`。
- E4 额外产物：`commit_log.jsonl`（格式见 DEV-2）。
- E6 产物：CSV（格式见 DEV-5）。
- 每次运行后在该目录写一行 `RUNINFO.md`：完整命令行、起止时间、样本数、error 数。

### 5.2 QA 验收线（每次运行后自查）

| 检查项 | 通过标准 |
|---|---|
| 失败样本占比 | < 2%，且 error 信息已记录 |
| 合成集 sanity（E1/E4/E5 的 System A） | Extra Long 组 mean TTFT ≈ 6–8 s；System B 各 Long+ 组 mean ≈ 0.9–1.3 s（与论文 Table III 同量级） |
| 真实语音 sanity（E2a 的 System A） | librispeech clean 转写文本与参考文本目测一致（WER 由本机离线计算；若转写明显错乱/为空占比 >10%，停止并报告） |
| E3 | LA 模式有稳定文本提交，无队列死锁（日志无长时间空转） |
| config 块 | 与 §1.3 锁定表一致（重点：suffix_segments=0、max_tokens、模型名） |

**任何一项不通过：停止后续运行，保留现场（日志+已产出文件），通知需求方。不要自行调参重试。**

### 5.3 Changelog

每完成一个 E 任务，在 `experiments/results/revision/REVISION_CHANGELOG.md` 追加：

```
## <日期> <任务编号> <一句话说明>
- 命令：<完整命令行>
- 产物：<目录/文件清单>
- 关键数字：<样本数、error 数、1-2 个 sanity 数字>
- 异常与处理：<无 / 描述>
```

### 5.4 回传清单（全部完成后打包）

```
experiments/results/revision/
├── env_versions.txt
├── REVISION_CHANGELOG.md
├── r1_stats/repeat_r{1,2,3}/
├── r2_real_speech/（全部子目录）
├── r3_baseline_la/
├── r4_commit/（含 commit_log.jsonl；若有 _ext 也一并）
├── r6_ttfa/endpoint/ 与 tts_first_chunk.csv
└── （新增/修改的源码 diff：git diff > code_changes.patch）
```

打包：`tar -czf revision_results_<date>.tar.gz experiments/results/revision code_changes.patch`，传回本机。

---

## 六、注意事项（务必遵守）

1. **不改旧数据**：`experiments/results/exp1_latency|exp2_ablation|exp3_quality/` 只读。
2. **参数锁定**：除各任务明确给出的参数外，一律按 §1.3 锁定表；特别是 `--suffix-segments 0`（脚本默认是 1，不传就错了）。
3. **不删实时仿真 sleep**（见 §4 警告）。
4. 每次脚本启动的 3 轮预热是流程一部分，不要跳过或计入结果。
5. 遇到显存不足：先 `clear_gpu_memory`（脚本内已有周期清理）；仍不足则记录并报需求方，不要自行降精度/换模型。
6. 长任务建议 `nohup`/`tmux` 运行并保留完整日志文件（`--log-level INFO`，日志随结果目录一并保存）。
7. 断点续传：中断后同一 `--output-dir` 重跑同一命令即可续传；换配置才用 `--no-resume`。
