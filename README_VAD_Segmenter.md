# VAD 音频分段器使用说明

## 概述

VAD音频分段器(`src/asr/audio_segmenter.py`)是一个基于Silero VAD的音频分割工具，支持批量和流式两种处理模式，确保分割后的音频段能完整重构原始文件。

## 功能特性

- ✅ **VAD语音检测**: 使用Silero VAD模型进行高精度语音活动检测
- ✅ **批量分段**: 对完整音频文件进行一次性分段处理
- ✅ **流式分段**: 支持实时音频流的分段处理
- ✅ **完整性保证**: 分割后的音频段可以完整重构原始文件
- ✅ **自动保存**: 将分段结果保存为WAV文件
- ✅ **完整性验证**: 自动验证分段结果的完整性和连续性

## 安装依赖

```bash
# 安装必要的依赖包
pip install -r requirements.txt
```

## 使用方法

### 1. 命令行使用

```bash
# 测试模式 - 使用合成音频进行测试
python src/asr/audio_segmenter.py --mode test

# 批量分段模式
python src/asr/audio_segmenter.py --mode batch --audio your_audio.wav

# 流式分段模式
python src/asr/audio_segmenter.py --mode streaming --audio your_audio.wav

# 同时测试批量和流式模式
python src/asr/audio_segmenter.py --mode both --audio your_audio.wav
```

### 2. 参数说明

- `--mode`: 测试模式 (`test`, `batch`, `streaming`, `both`)
- `--audio`: 输入音频文件路径
- `--output-dir`: 输出目录 (默认: `/home/project/streamllm/results/wav_segments`)
- `--chunk-duration`: 流式模式下的音频块大小，单位秒 (默认: 0.5)
- `--threshold`: VAD检测阈值，0-1之间 (默认: 0.5)
- `--min-speech`: 最小语音段长度，单位秒 (默认: 0.25)
- `--min-silence`: 最小静音段长度，单位秒 (默认: 0.2)

### 3. 编程接口使用

```python
from src.asr.audio_segmenter import VADAudioSegmenter

# 创建分段器
segmenter = VADAudioSegmenter(
    sample_rate=16000,
    threshold=0.5,
    min_speech_duration=0.25,
    min_silence_duration=0.2
)

# 批量分段
segments = segmenter.segment_audio_file(
    audio_file_path="your_audio.wav",
    output_dir="/path/to/output"
)

# 流式分段
for audio_chunk in audio_stream:
    new_segments = segmenter.process_streaming_audio(audio_chunk)
    # 处理新的语音段
    
# 完成流式处理
final_segments = segmenter.finish_streaming()
```

## 输出格式

每个分段包含以下信息：

```python
{
    'segment_id': int,          # 段编号
    'start_time': float,        # 开始时间（秒）
    'end_time': float,          # 结束时间（秒）
    'duration': float,          # 持续时间（秒）
    'audio_data': np.ndarray,   # 音频数据
    'file_path': str            # 保存的文件路径（如果保存了）
}
```

## 输出文件命名

分段音频文件按以下格式命名：
```
{原文件名}_seg{编号:03d}_{开始时间:.2f}s-{结束时间:.2f}s.wav
```

例如：`test_audio_seg001_0.00s-2.30s.wav`

## 完整性验证

分段器会自动验证分割结果：

- **覆盖率**: 分段总时长与原始音频时长的比例
- **间隙检测**: 检查分段之间是否有遗漏
- **重叠检测**: 检查分段之间是否有重叠
- **完整性判断**: 综合评估分割结果是否完整

## 测试示例

运行测试模式会：

1. 创建一个10秒的合成测试音频，包含3个语音段和2个静音段
2. 分别使用批量和流式模式进行分段
3. 验证分段结果的完整性
4. 将分段结果保存到指定目录
5. 输出详细的分析报告

```bash
python src/asr/audio_segmenter.py --mode test
```

## 注意事项

1. **采样率**: 目前支持8000Hz和16000Hz采样率
2. **音频格式**: 支持librosa能够读取的所有音频格式
3. **内存使用**: 流式模式使用固定大小的缓冲区，适合处理长音频
4. **精度**: VAD阈值可以根据具体场景调整，建议在0.3-0.7之间

## 故障排除

如果遇到VAD模型加载失败，请检查：

1. 是否已安装 `silero-vad` 包
2. 网络连接是否正常（首次使用时需要下载模型）
3. 如果pip安装失败，会自动回退到torch.hub方式加载 