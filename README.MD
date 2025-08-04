# 智能语音对话系统低延迟优化研究 (StreamLLM-LowLatency)

## 1. 项目概述

本项目旨在研究和实现一套低延迟智能语音对话系统。核心目标是优化级联式语音对话系统（ASR -> LLM -> TTS）中从用户语音输入开始到大语言模型（LLM）产生首个响应Token的端到端延迟。通过采用流式处理和KV缓存等技术，显著降低系统的响应时间，提升用户交互体验。

### 核心功能
- **流式ASR**: 基于 faster-whisper 的实时语音识别，支持动态音频分段和上下文感知
- **流式LLM**: 基于 KV 缓存的低延迟文本生成，支持文本片段的流式输入和处理
- **端到端流水线**: 完整的音频->文本->生成响应流程，支持实时处理

## 2. 系统架构

### 2.1 核心模块结构

```
src/
├── asr/                    # 自动语音识别模块
│   ├── faster_whisper_streamer.py   # 基于faster-whisper的流式ASR处理器
│   └── audio_segmenter.py           # VAD音频分段器 (基于Silero VAD)
├── llm/                    # 大语言模型模块
│   └── stream_llm_inference.py      # 流式LLM推理引擎 (支持KV缓存)
├── pipeline/               # 流水线模块
│   └── streaming_pipeline.py        # ASR-LLM级联处理流程
├── utils/                  # 工具模块
│   ├── audio2stream.py              # 音频流处理工具
│   └── logging_utils.py             # 日志工具
└── config.py               # 配置文件
```

### 2.2 自动语音识别 (ASR) 优化

**核心技术**:
- **ASR引擎**: 采用 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 作为ASR引擎
- **音频分段**: 基于 Silero VAD 的智能语音活动检测和音频分段
- **流式处理**: 支持实时音频块处理和文本输出

**关键特性**:
1. **动态分段策略**:
   - 使用 VAD (Voice Activity Detection) 检测语音活动
   - 支持基于静音的智能分段和固定时长分段两种模式
   - 前缀段机制确保识别上下文的连续性

2. **上下文感知识别**:
   - 采用重叠音频段策略提高识别准确率
   - 利用词级时间戳进行精确文本对齐和替换
   - 支持历史上下文传递，提升连续对话识别质量

3. **实时性优化**:
   - 流式音频输入处理
   - 动态识别阈值调整
   - 支持并发音频处理和文本输出

### 2.3 大语言模型 (LLM) 流式响应优化

**核心技术**:
- **KV缓存机制**: 利用 Transformer 模型的键值缓存减少计算开销
- **流式文本处理**: 支持文本片段的增量输入和处理
- **首Token延迟优化**: 针对首个响应Token的生成时间进行专门优化

**关键特性**:
1. **流式输入处理**:
   - 支持文本片段的流式添加和KV缓存更新
   - 动态生成触发机制（标点符号检测）
   - 支持评估模式和完整生成模式

2. **KV缓存优化**:
   - 预计算KV缓存减少首Token延迟
   - 增量式缓存更新避免重复计算
   - 支持缓存状态管理和重置

3. **生成控制**:
   - 灵活的生成参数配置（温度、top_p等）
   - 支持停止序列和EOS检测
   - 详细的性能计时统计

## 3. 快速开始

### 3.1 环境配置

1. **Python环境**: 推荐使用 Python 3.8+
2. **依赖安装**:
   ```bash
   pip install -r requirements.txt
   ```

3. **环境变量配置**: 复制 `.env.example` 到 `.env` 并配置相应参数:
   ```bash
   # 模型配置
   LLM_MODEL_NAME="your_llm_model_name"
   ASR_MODEL_NAME="base"  # faster-whisper模型大小
   
   # Hugging Face配置
   HF_HOME="./cache"
   HF_TOKEN="your_hf_token"
   
   # 数据路径
   DATA_DIR="data"
   ```

### 3.2 测试运行

**完整流水线测试**:
```bash
python test_streaming.py --wav_path "path/to/your/audio.wav" --chunk_duration 0.5
```

**ASR模块单独测试**:
```bash
python -m src.asr.faster_whisper_streamer --wav_path "path/to/audio.wav" --model_size base
```

**LLM模块单独测试**:
```bash
python -m src.llm.stream_llm_inference
```

## 4. 核心API接口

### 4.1 StreamingASRProcessor

```python
from src.asr.faster_whisper_streamer import StreamingASRProcessor

# 创建ASR处理器
asr_processor = StreamingASRProcessor(
    model_size="base",
    recognition_threshold=3.0,  # 识别阈值（秒）
    prefix_segments=1           # 前缀段数
)

# 处理音频块
for audio_chunk, is_last in audio_stream:
    new_texts = asr_processor.add_audio_chunk(audio_chunk, is_last)
    for text in new_texts:
        print(f"识别结果: {text}")
```

### 4.2 StreamLLMInference

```python
from src.llm.stream_llm_inference import StreamLLMInference

# 创建LLM推理引擎
llm_processor = StreamLLMInference()

# 流式文本处理
def text_generator():
    texts = ["你好", "，请问", "你能做什么？"]
    for i, text in enumerate(texts):
        yield text, (i == len(texts) - 1)  # (文本, 是否结束)

# 生成响应
for token, timestamp in llm_processor.stream_add_and_generate(text_generator()):
    print(f"生成Token: {token}")
```

### 4.3 端到端流水线

```python
from src.pipeline.streaming_pipeline import StreamingPipeline

# 创建流水线
pipeline = StreamingPipeline(
    asr_model_size="base",
    llm_model_name="your_model"
)

# 处理音频流
pipeline.process_audio_stream(audio_frames_producer)
```

## 5. 实验设计与评估

### 5.1 评估指标

- **首个Token延迟 (First Token Latency)**: 从语音输入完成到LLM生成首个Token的时间
- **ASR准确率**: 使用词错误率(WER)评估识别准确性  
- **端到端延迟**: 完整的音频->文本->响应生成链路延迟
- **实时因子**: 处理时间与音频时长的比值

### 5.2 测试数据

项目支持多种音频输入格式：
- **音频格式**: WAV, MP3等（自动转换为16kHz单声道）
- **测试数据**: 位于 `data/processed_audio/` 目录
- **时长分类**: 按语音长度分类测试不同场景的性能

### 5.3 性能优化参数

**ASR优化参数**:
- `recognition_threshold`: 识别阈值，控制何时触发转录
- `prefix_segments`: 前缀段数，影响上下文质量
- `min_speech_duration`: 最小语音段长度

**LLM优化参数**:
- `eval_mode`: 评估模式，只生成首Token用于延迟测试
- `temperature`: 生成随机性控制
- KV缓存策略选择

## 6. 技术实现细节

### 6.1 ASR流式处理

1. **音频分段策略**:
   - 基于VAD的智能分段
   - 固定时长分段作为备选
   - 支持重叠段处理确保连续性

2. **文本对齐算法**:
   - 基于词级时间戳的精确对齐
   - 历史文本替换和更新机制
   - 上下文感知的识别优化

### 6.2 LLM KV缓存机制

1. **缓存更新策略**:
   - 增量式KV缓存计算
   - 动态缓存状态管理
   - 内存使用优化

2. **生成触发机制**:
   - 基于标点符号的智能触发
   - 支持手动和自动触发模式
   - 可配置的触发条件

### 6.3 并发处理架构

- 使用队列和线程实现模块间异步通信
- 支持生产者-消费者模式的数据流处理
- 内置错误处理和恢复机制

## 7. 扩展开发

### 7.1 添加新的ASR引擎

继承 `StreamingASRProcessor` 基类并实现相应接口:

```python
class CustomASRProcessor(StreamingASRProcessor):
    def _transcribe_segments(self, segments):
        # 实现自定义ASR逻辑
        pass
```

### 7.2 集成其他LLM模型

修改 `StreamLLMInference` 类以支持不同的模型架构:

```python
# 在config.py中配置新模型
LLM_MODEL_NAME = "path/to/your/model"
```

### 7.3 扩展流水线功能

在 `StreamingPipeline` 中添加新的处理阶段，如TTS集成。

## 8. 项目文件结构

```
streamllm/
├── CLAUDE.md                  # Claude Code 配置文件
├── README.md                  # 项目说明文档
├── test_streaming.py          # 主测试程序
├── config.json               # 配置文件
├── logs/                     # 日志目录
│   └── experiment.log
├── data/                     # 数据目录
│   └── processed_audio/
│       └── length30+/        # 按时长分类的音频文件
├── src/                      # 源代码目录
│   ├── config.py            # 配置管理
│   ├── asr/                 # ASR模块
│   │   ├── faster_whisper_streamer.py
│   │   └── audio_segmenter.py
│   ├── llm/                 # LLM模块
│   │   └── stream_llm_inference.py  
│   ├── pipeline/            # 流水线模块
│   │   └── streaming_pipeline.py
│   └── utils/               # 工具模块
│       ├── audio2stream.py
│       └── logging_utils.py
└── requirements.txt         # 依赖列表
```

## 9. 常见问题

### 9.1 模型加载问题

**Q**: faster-whisper模型加载失败？
**A**: 检查网络连接，或手动下载模型到本地目录，然后在config.py中指定本地路径。

**Q**: LLM模型显存不足？
**A**: 在config.py中设置`DEVICE="cpu"`使用CPU推理，或使用更小的模型。

### 9.2 性能优化

**Q**: 首Token延迟过高？
**A**: 调整以下参数：
- 降低`recognition_threshold`值
- 启用`eval_mode=True`只生成首Token
- 使用更小的模型

**Q**: ASR识别准确率低？
**A**: 尝试：
- 增加`prefix_segments`数量
- 调整VAD参数的`threshold`值
- 使用更大的whisper模型

### 9.3 开发调试

**Q**: 如何查看详细日志？
**A**: 在测试命令中添加`--log_level DEBUG`参数。

**Q**: 如何单独测试某个模块？
**A**: 使用各模块的main函数或测试脚本进行单独测试。

## 10. 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 11. 许可证

本项目采用 MIT 许可证。详见 `LICENSE` 文件。

## 12. 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues: 在GitHub仓库中提交Issue
- 邮箱: [项目维护者邮箱]

---

**注意**: 本项目仍在积极开发中，API可能会发生变化。请关注版本更新和变更日志。