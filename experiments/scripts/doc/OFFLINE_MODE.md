# 离线运行指南

本文档说明如何在完全离线的环境下运行实验。

## 概述

系统使用以下模型，都会自动缓存到本地：

1. **Silero VAD**: 缓存在 `~/.cache/torch/hub/snakers4_silero-vad_master/`
2. **OpenAI Whisper**: 缓存在 `~/.cache/whisper/`
3. **Hugging Face LLM**: 缓存在环境变量 `HF_HOME` 指定的目录

## 前置要求：首次联网运行

在离线运行前，需要先在联网环境下运行一次，让所有模型自动下载到本地缓存：

```bash
# 运行一次实验，自动下载并缓存所有模型
./experiments/scripts/run_exp_ablation.sh full --max-samples 1
```

## 离线运行

### 方法1：依赖自动缓存（推荐）

如果已经运行过一次并缓存了所有模型，直接运行即可：

```bash
./experiments/scripts/run_exp_ablation.sh full --max-samples 100
```

系统会自动使用本地缓存，不会访问网络。

### 方法2：强制禁用网络访问

如果要完全禁用网络访问（确保不会因为网络问题而失败），可以设置环境变量：

```bash
# 方法A：临时禁用网络（仅对当前命令有效）
HF_HUB_OFFLINE=1 ./experiments/scripts/run_exp_ablation.sh full --max-samples 100

# 方法B：在脚本中设置（对所有后续命令有效）
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
./experiments/scripts/run_exp_ablation.sh full --max-samples 100
```

## 验证缓存是否完整

运行以下命令检查模型缓存：

```bash
# 检查 Whisper 缓存
ls -lh ~/.cache/whisper/

# 检查 Silero VAD 缓存
ls -lh ~/.cache/torch/hub/snakers4_silero-vad_master/

# 检查 Hugging Face 缓存
ls -lh $HF_HOME/
```

## 常见问题

### Q1: 提示 "HTTP Error 503: Service Unavailable"

**原因**: `torch.hub.load` 或 Hugging Face 尝试访问网络，但网络不可用。

**解决方案**:
1. 确认模型已缓存（见上面的验证命令）
2. 设置 `HF_HUB_OFFLINE=1` 环境变量

### Q2: 每次都重新下载 Silero VAD

**原因**: 之前的代码配置错误（已修复）。

**解决方案**: 
- 更新到最新代码
- Silero VAD 会自动使用 `~/.cache/torch/hub/` 缓存

### Q3: Whisper 模型找不到

**原因**: 
- 模型未缓存
- 模型名称不正确

**解决方案**:
- 先在联网环境下运行一次
- 确认模型名称正确（tiny, base, small, medium, large, turbo, large-v3 等）

## 模型缓存大小参考

| 模型 | 大小 | 缓存位置 |
|------|------|----------|
| Silero VAD | ~2 MB | `~/.cache/torch/hub/` |
| Whisper tiny | ~75 MB | `~/.cache/whisper/` |
| Whisper base | ~142 MB | `~/.cache/whisper/` |
| Whisper small | ~466 MB | `~/.cache/whisper/` |
| Whisper medium | ~1.5 GB | `~/.cache/whisper/` |
| Whisper large-v3 | ~3 GB | `~/.cache/whisper/` |
| Whisper turbo | ~1.5 GB | `~/.cache/whisper/` |
| Qwen2-7B | ~15 GB | `$HF_HOME/` |

## 完全离线环境配置

如果需要在完全没有网络的机器上运行：

1. **在联网机器上准备缓存**:
   ```bash
   # 运行一次，下载所有模型
   ./experiments/scripts/run_exp_ablation.sh full --max-samples 1
   ```

2. **打包缓存目录**:
   ```bash
   tar -czf model_cache.tar.gz \
       ~/.cache/whisper/ \
       ~/.cache/torch/hub/ \
       $HF_HOME/
   ```

3. **在离线机器上解压**:
   ```bash
   cd ~
   tar -xzf model_cache.tar.gz
   ```

4. **运行实验**:
   ```bash
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ./experiments/scripts/run_exp_ablation.sh full
   ```

