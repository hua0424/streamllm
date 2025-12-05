# TTS客户端使用说明

## 概述

TTS客户端工具用于连接远程TTS服务，进行文本到语音的转换。支持单个文本合成和批量JSON文件处理。

## 核心功能

### 1. 单个文本合成
调用远程TTS服务将单个文本转换为WAV音频文件。

### 2. 批量处理
批量处理JSON文件，为每个文件中的文本生成对应的音频文件。

### 3. 自动语言识别
批量处理时根据JSON文件中的language字段自动选择合适的说话人：
- 中文(zh)：自动设置为"中文女"
- 英文(en)：自动设置为"英文女"

## 系统要求

### Python依赖
- requests
- json
- wave  
- pathlib
- argparse
- time
- struct

### 服务依赖
- TTS后端服务运行在指定地址（默认：`http://host.docker.internal:20401`）
- 后端需要支持 `/inference_sft` 接口

## 使用方法

### 命令行参数

```bash
python tts.py [参数选项]
```

#### 基础参数
- `--url`: TTS服务URL（默认：http://host.docker.internal:20401）
- `--test`: 测试连接到TTS服务

#### 单个文本合成参数
- `--text`: 要合成的文本内容（必需）
- `--spk-id`: 说话人ID（默认：中文女）
- `--speed`: 语速调节系数（默认：1.0，>1.0加速，<1.0减速）
- `--output`: 输出音频文件路径

#### 批量处理参数
- `--batch`: 启用批量处理模式
- `--input-dir`: 包含JSON文件的输入目录（必需）
- `--output-dir`: 音频文件输出目录（必需）
- `--pattern`: JSON文件匹配模式（默认：*.json）

### 使用示例

#### 1. 测试连接
```bash
python tts.py --test
```

#### 2. 单个文本合成（中文）
```bash
python tts.py --text "你好，欢迎使用TTS服务" --output result.wav
```

#### 3. 单个文本合成（英文）
```bash
python tts.py --text "Hello, welcome to TTS service" --spk-id "英文女" --output result_en.wav
```

#### 4. 单个文本合成（调节语速）
```bash
python tts.py --text "这是一个语速测试" --speed 1.2 --output fast_result.wav
```

#### 5. 批量处理
```bash
python tts.py --batch --input-dir ./json_files --output-dir ./audio_files
```

#### 6. 批量处理（指定语速）
```bash
python tts.py --batch --input-dir ./json_files --output-dir ./audio_files --speed 1.1
```

### JSON文件格式

批量处理时，输入的JSON文件需要包含以下字段：

```json
{
  "text": "要合成的文本内容",
  "language": "zh",
  "audio_file": "output_filename.wav"
}
```

#### 字段说明
- `text`: 必需，要合成语音的文本内容
- `language`: 可选，语言代码（zh/en），默认为zh，用于自动选择说话人
- `audio_file`: 可选，输出音频文件名，默认使用JSON文件名+.wav

## 支持的说话人

| 说话人ID | 语言 | 描述 |
|---------|------|------|
| 晓伊 | 中/英文 | 女性声音（默认） |
| 云皓 | 中/英文 | 男性声音 |

## 技术参数

### 音频输出格式
- **格式**: WAV
- **采样率**: 22050Hz
- **声道**: 单声道
- **位深**: 16bit

### 性能参数  
- **连接超时**: 60秒（可配置）
- **批量处理间隔**: 0.5秒（避免过载服务器）
- **流式下载**: 使用16000字节块大小

## 错误处理

### 常见错误及解决方案

#### 1. 连接失败
```
Cannot connect to TTS service: http://host.docker.internal:20401
```
**解决方案**: 检查TTS服务是否运行，确认URL地址正确

#### 2. 请求超时
```
Request timeout (>60s)
```
**解决方案**: 检查网络连接，考虑增加timeout参数

#### 3. 文本为空
```
Skipping empty text file: sample.json
```
**解决方案**: 确保JSON文件包含有效的text字段

#### 4. 音频验证失败
```
Warning: Audio file may be corrupted
```
**解决方案**: 检查生成的音频文件，可能需要重新生成

## 目录结构

```
tools/
├── tts.py                    # TTS客户端主程序
├── TTS_USAGE.md             # 本使用说明
├── ttsresult/               # 默认输出目录
│   └── result.wav          # 默认输出文件
└── test_data/              # 测试数据示例
    ├── test_001.json
    ├── test_002.json
    └── test_003.json
```

## 高级功能

### 1. 自定义服务地址
```bash
python tts.py --url http://custom-tts-server:8080 --text "测试文本"
```

### 2. 保持目录结构
批量处理时会保持输入目录的子目录结构：
```
input/
├── group1/
│   └── text1.json
└── group2/
    └── text2.json

output/
├── group1/
│   └── text1.wav  
└── group2/
    └── text2.wav
```

### 3. 跳过已存在文件
批量处理会自动跳过已存在的音频文件，支持断点续传。

## 注意事项

1. **语速调节**: 语速参数由后端处理，客户端不做额外处理
2. **文件覆盖**: 单个合成模式会覆盖已存在的输出文件
3. **批量处理**: 会跳过已存在的文件，避免重复处理
4. **内存占用**: 流式处理音频数据，内存占用较低
5. **错误恢复**: 单个文件失败不影响批量处理继续进行

## 版本信息

- **当前版本**: 基于inference_sft接口
- **支持的后端**: CosyVoice TTS服务
- **更新日期**: 2025年

## 技术支持

如遇问题，请检查：
1. TTS服务是否正常运行
2. 网络连接是否稳定  
3. JSON文件格式是否正确
4. 输出目录是否有写入权限