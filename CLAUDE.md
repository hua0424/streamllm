# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

硕士论文项目，研究级联式语音对话系统（ASR → LLM → TTS）的**低延迟优化**。代码是论文的实验载体，不是产品——评判标准是延迟指标（尤其 TTFT）与实验可复现性，而非工程完备性。

**两期工作**：
- **一期（已完成，`main` 分支）**：流式 ASR + LLM KV cache 增量预填充，打破"TTFT 随语音长度线性增长"。这是当前 `src/` 的全部内容。
- **二期（进行中，`bargeincache` 分支）**：播放感知的 KV 缓存管理 + barge-in（打断）。核心原则「对话历史 = 用户实际听到的内容」。**二期尚无代码**，处于设计阶段。开始二期任务前必读 `docs/paper2_context.md`（主交接文档）、`docs/decisions.md`（决策日志 D-001~）、`docs/handoff.md`（当前断点与下一步）。

## 环境与命令

用 `uv` 管理（不要用裸 `python`/`pip`；`torch` 走 cu121 专用 index，见 `pyproject.toml`）：

```bash
uv venv --python 3.10 && uv sync        # 建/同步环境
uv run python -m src.run_test_simple --mode both --audio path/to.wav   # 全链路延迟对比
uv run python -m src.asr.run_stream_asr_test    # ASR 模块 smoke test
uv run python -m src.llm.run_llm_test           # LLM 模块 smoke test
```

**三个论文实验**（项目根目录运行，均支持增量保存/断点续传，结果入 `experiments/results/`）：
```bash
uv run python -m experiments.scripts.run_exp_latency    # 实验一：TTFT vs 语音长度
uv run python -m experiments.scripts.run_exp_ablation   # 实验二：流式ASR / KV预填充 各自贡献消融
uv run python -m experiments.scripts.run_exp_quality    # 实验三：流式 vs 非流式 ASR 的 WER/CER
```

**运行陷阱**：
- 必须从**项目根目录**以 `-m` 模块方式运行（代码用绝对导入 `from src.config import ...`）。`src/run_test_simple.sh` 会自动 `cd` 根目录并设 `PYTHONPATH=.`。
- GPU 上若报 cudnn 相关错误，需要 `export LD_LIBRARY_PATH=".venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"`（`run_test_simple.sh` 已内置）。
- **无 pytest**。验证改动 = 跑上面的脚本看打印的延迟指标是否合理。
- 模型加载是 **offline-first**：先试本地缓存（`local_files_only=True`），失败再联网。首次运行需下载 Whisper + Silero VAD + HF 模型。

## 架构大图（跨文件才能看清的部分）

### 全链路是四线程流水线，用 Queue 串接

`src/run_test_simple.py` 的 `StreamPipelineTest` 是理解整个系统的入口。四个 worker 线程通过三个 `queue.Queue` 级联，模拟真实时序（音频按 `chunk_duration` 逐块 `time.sleep` 产生）：

```
_audio_generation_worker → audio_chunk_queue
  → _segmentation_worker (Silero VAD 切分)      → audio_segment_queue
    → _asr_worker (collector + transcriber 两子线程)  → text_queue
      → _llm_worker (KV prefill 累积，is_end 时 generate)
```

延迟指标的核心是 `timings` 字典里的相对时间戳（`start_time` / `audio_end_time` / `last_text_time` / `first_token_time`），`_get_metrics()` 由它们算出 "audio-end → first-token" 等关键延迟。改流水线时保持这些时间戳的打点位置，否则实验数据失去可比性。

### ASR 与 LLM 的衔接是 "final segment" 粒度

`src/asr/faster_whisper_streamer.py` 的 `StreamingASRProcessor` 维护一个滑动窗口的段队列（`ASRCache`）。关键机制在 `_determine_output_segments()`：靠 `prefix_segments`（前缀上下文，不输出）+ `suffix_segments_atleast`（后缀，等未来上下文）夹出中间的**确定性段**才输出给 LLM。所以 LLM 收到的是 final 文本片段流，不是会被修正的 partial transcript。词级时间戳用于把合并转录结果切回各段（`_extract_segment_text`）。

### LLM 增量 KV 预填充是一期的核心，也是二期改造主战场

`src/llm/stream_llm_inference.py` 的 `StreamLLMInference`：

- **手工字符串拼 chat template**（不是分段 `apply_chat_template`）。初始化时抽出 `generation_prompt` 段（`<|im_end|>\n<|im_start|>assistant\n`）存起来，`cache_prompt(is_end=True)` 时拼到末尾关闭 user role。
- **`cache_prompt()`** 增量预填充：每个 final 片段调 `_add_stream_prompt()`，让 transformers 自动 append KV，但 **`attention_mask` 和 `position_ids` 都手动显式拼**（`position_ids` 用 `pre_attention_mask.shape[1]` 当 past_length）。
- **`generate()`** 是手写 token-by-token 循环（非 `model.generate()`），yield 每个 token，打断只需消费侧停止迭代。
- `past_key_values` 被当作**不透明对象**传递，从不调 cache 方法。

**给二期的两个已知关键点**（细节见 `docs/paper2_context.md` Q1-Q5）：
1. `generate()` 内部更新的 KV **没有写回 caller 的 KVCache** —— 一期不累积 assistant 端 KV，也不支持多轮。二期需新建"边生成边累积、可被 crop 的 assistant-side KVCache"。
2. 二期 KV 截断走 `DynamicCache.crop()`；crop 后必须**同步截短 `pre_attention_mask`** 并用新 past_length 重算 position_ids，否则位置编码错乱。

## 约定

- 配置集中在 `src/config.py`（读 `.env`）；模块内不要硬编码路径/设备。ASR 与 LLM 可分卡（`--asr-device` / `--llm-device`）。
- 核心代码用 `src/utils/logging_utils.py` 的结构化日志，不要用 `print()`（流水线里 `print` 仅用于流式吐字展示）。
- 提交信息用简短祈使句（多为中文，如"修复…/增加…/完成…"），一次提交一个逻辑变更。改实验方法学时同步更新 `experiments/EXPERIMENT_DESIGN.md`。
- 二期每次技术决策倒序追加到 `docs/decisions.md`；里程碑结束同步 `docs/paper2_context.md` §九时间线。
