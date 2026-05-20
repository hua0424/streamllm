# 智能语音对话系统低延迟优化研究 (StreamLLM-LowLatency)

## 1. 项目概述

本项目旨在研究和实现一套低延迟智能语音对话系统。核心目标是优化级联式语音对话系统（ASR -> LLM -> TTS）中从用户语音输入开始到大语言模型（LLM）产生首个响应Token的端到端延迟。

通过采用流式处理和KV缓存等技术，显著降低系统的响应时间，提升用户交互体验。

### 核心特性

- **流式 ASR (Streaming ASR)**: 基于 `openai-whisper` (PyTorch) 与 `Silero VAD` 的实时语音识别。支持动态音频分段、重叠窗口识别和上下文感知，以在保证准确率的同时降低延迟。
- **流式 LLM (Streaming LLM)**: 基于 `transformers` 的低延迟文本生成。利用 KV Cache (键值缓存) 技术，支持流式文本片段输入（Incremental Prompting），避免重复计算历史上下文。
- **端到端流水线**: 完整的 "音频流 -> 分段 -> ASR流式转录 -> LLM流式输入 -> LLM流式生成" 链路，并提供详细的延迟性能分析。

## 2. 系统架构与项目结构

### 2.1 项目文件结构

```
src/
├── __init__.py
├── config.py              # 全局配置管理，支持 .env 环境变量
├── run_test_simple.py     # 全链路测试主程序 (Python)
├── run_test_simple.sh     # 启动脚本 (Bash wrapper)
├── asr/                   # ASR (自动语音识别) 模块
│   ├── __init__.py
│   ├── faster_whisper_streamer.py   # 流式ASR处理器，实现上下文感知和缓存管理
│   ├── run_stream_asr_test.py       # ASR 模块独立测试脚本
│   └── streamaudio_segmenter.py     # 基于 Silero VAD 的流式音频分段器
├── llm/                   # LLM (大语言模型) 模块
│   ├── __init__.py
│   ├── run_llm_test.py              # LLM 模块独立测试脚本
│   └── stream_llm_inference.py      # 流式LLM推理引擎，实现 KV Cache 管理
└── utils/                 # 通用工具模块
    ├── __init__.py
    ├── audio_utils.py               # 音频处理工具
    ├── audio2stream.py              # 音频流模拟生成器
    └── logging_utils.py             # 日志配置
```

### 2.2 核心模块详解

#### 自动语音识别 (ASR)
- **引擎**: `openai-whisper` (官方 PyTorch 实现)。
- **分段**: 使用 `Silero VAD` 进行实时语音活动检测，智能切分音频流。
- **流式策略**: 
    - 维护一个音频缓冲区 (`ASRCache`)。
    - 使用滑动窗口机制，保留前缀 (`prefix_segments`) 以提供识别上下文。
    - 动态判断何时输出确定性的文本片段。

#### 大语言模型 (LLM)
- **引擎**: HuggingFace `transformers`。
- **KV Cache 优化**: 
    - `StreamLLMInference` 类维护 `KVCache` 状态。
    - 支持 `cache_prompt` 接口：当 ASR 产出新的文本片段时，仅对新片段计算 KV 并追加到缓存中，无需重新计算整个 Prompt。
    - 显著降低了长上下文下的首字生成延迟 (TTFT)。

## 3. 快速开始

### 3.1 环境配置

1. **Python环境**: 推荐 Python 3.10+
2. **依赖安装（uv 推荐）**:
   ```bash
   # 创建/同步虚拟环境
   uv venv --python 3.10
   uv sync
   ```
   如需使用 pip，可参考 `requirements.txt`（需确保安装匹配 CUDA 版本的 `torch`）。

3. **环境变量配置**:
   项目根目录下创建 `.env` 文件 (参考 `.env.example`，如果存在):
   
   ```bash
   # 基础设备配置 (auto/cuda/cpu)
   DEVICE="auto"

   # 模型配置
   LLM_MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"  # 推荐使用小模型进行测试
   ASR_MODEL_NAME="base"                         # whisper 模型大小 (tiny/base/small/medium/large)

   # Hugging Face 缓存
   HF_HOME="./cache"
   
   # ASR流式参数
   VAD_THRESHOLD="0.5"
   ASR_CHUNK_SECONDS="10"
   ```

### 3.2 运行全链路测试

项目提供了一个便捷的 Shell 脚本 `src/run_test_simple.sh` 来运行测试。该测试会使用一段音频文件，分别模拟流式输入和非流式输入，并对比两者的延迟性能。

**基本用法**:
```bash
# 运行对比测试 (Streaming vs Non-Streaming)
./src/run_test_simple.sh both
# 或使用 uv（无需手动激活虚拟环境）
uv run python -m src.run_test_simple --mode both --audio /path/to/your/audio.wav

# 指定音频文件运行
./src/run_test_simple.sh both --audio /path/to/your/audio.wav

# 仅运行流式测试，开启调试日志
./src/run_test_simple.sh streaming --log-level DEBUG

# 仅运行非流式测试
./src/run_test_simple.sh non-streaming
```

**参数说明**:
- `mode`: `streaming` (流式), `non-streaming` (非流式), `both` (对比测试)。
- `--audio`: 输入音频文件路径。
- `--chunk-duration`: 流式模式下模拟音频块的时长 (ms)，默认 500ms。
- `--asr-device` / `--llm-device`: 显式指定各模块运行设备。
- `--save-results`: 将测试结果保存为 JSON 文件到 `experiments/results`。

## 4. 性能评估指标

运行测试后，控制台会输出详细的性能统计信息：

1. **Audio End to First Token (流式核心指标)**: 
   - 定义：从用户说话结束（音频流结束）到 LLM 生成第一个字符的时间。
   - 优化目标：越短越好。本项目通过流式 ASR 和 LLM KV Cache 旨在最小化此指标。

2. **Audio Load to First Token (非流式基准)**:
   - 定义：从音频加载完成（模拟录音结束）到 LLM 生成第一个字符的时间。
   - 包含：完整的 ASR 转录时间 + 完整的 LLM Prompt 处理时间。

3. **Latency Improvement**:
   - 流式方案相对于非流式方案的延迟降低值。

## 5. 核心 API 接口示例

### 5.1 ASR 流式处理
```python
from src.asr.faster_whisper_streamer import StreamingASRProcessor, ASRCache

processor = StreamingASRProcessor(model_size="base")
cache = ASRCache()

# 模拟接收音频段
# audio_segment 是包含音频数据和元数据的对象
cache, new_text, is_final = processor.transcribe_audio_segment(cache)

if new_text:
    print(f"流式识别结果: {new_text}")
```

### 5.2 LLM 流式推理与缓存
```python
from src.llm.stream_llm_inference import StreamLLMInference

llm = StreamLLMInference(model_name="Qwen/Qwen2.5-0.5B-Instruct")
kv_cache = None

# 第一段输入
kv_cache = llm.cache_prompt("你好，", pre_cache=kv_cache)

# 第二段输入 (增量计算，复用之前的 kv_cache)
kv_cache = llm.cache_prompt("请介绍一下你自己。", pre_cache=kv_cache, is_end=True)

# 生成回复
for token in llm.generate(pre_cache=kv_cache):
    print(token, end="")
```

## 6. 常见问题 (FAQ)

**Q: 首次运行速度较慢？**
A: 首次运行需要下载 HuggingFace 模型和 Silero VAD 模型。此外，模型加载到 GPU 也需要时间。建议在测试脚本中观察 "Warmup" 阶段的日志。

**Q: VAD 切分不准确？**
A: 可以在 `.env` 或 `config.py` 中调整 `VAD_THRESHOLD` (默认 0.5) 和 `VAD_MIN_SPEECH_MS`。

**Q: ASR 识别结果在流式输出时会变化？**
A: 是的，这是预期的。为了提高准确率，流式 ASR 会保留一部分音频上下文（Context），当前的输出可能会在接收到更多音频后被修正（如果使用了更复杂的输出策略）。但在本项目的简单实现中，我们主要关注**确定的**输出片段如何尽快送入 LLM。

## 7. 许可证

本项目采用 MIT 许可证。
