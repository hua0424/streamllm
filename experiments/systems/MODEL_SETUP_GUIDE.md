# 模型配置和运行指南

## 📋 模型配置更新

根据实验的严谨性要求，已统一配置以下模型：

### 文本LLM模型
- **模型名称**: `Qwen/Qwen2-7B-Instruct`
- **用于系统**: 系统A、系统B、系统A'
- **模型大小**: 约14GB
- **显存需求**: 
  - FP16（半精度）: 约14GB
  - FP32（全精度）: 约28GB
  - INT8（量化）: 约7GB

### 端到端语音模型
- **模型名称**: `Qwen/Qwen2-Audio-7B-Instruct`
- **用于系统**: 系统C（理想化对比系统）
- **模型大小**: 约14GB
- **特点**: 直接处理音频输入，无需ASR

### ASR模型
- **模型**: `faster-whisper base`
- **大小**: 约150MB
- **用于**: 所有级联系统（A、B、A'）

## 🚀 运行方式

### 1. 环境准备

#### 检查CUDA环境
```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"
```

#### 安装依赖
```bash
# 如果使用pip
pip install transformers torch accelerate

# 如果使用uv
uv add transformers torch accelerate
```

### 2. 模型下载（可选）

提前下载模型可以避免运行时等待：

```bash
# 下载所有模型
python experiments/systems/download_models.py

# 如果在中国大陆，可以使用镜像
export HF_ENDPOINT=https://hf-mirror.com
python experiments/systems/download_models.py
```

### 3. 测试模型加载

验证模型是否可以正确加载：

```bash
# 测试模型加载和CUDA
python experiments/systems/test_model_loading.py
```

预期输出：
```
✅ CUDA可用
✅ Tokenizer加载成功
✅ 模型加载成功（使用FP16半精度）
✅ 推理测试成功
```

### 4. 运行系统A测试

```bash
# 测试系统A（基线串行系统）
python experiments/systems/system_a_baseline.py
```

如果遇到模型下载问题，可以：
1. 设置镜像源: `export HF_ENDPOINT=https://hf-mirror.com`
2. 使用代理: `export https_proxy=http://your-proxy:port`
3. 手动下载到cache目录

### 5. 完整实验运行

```bash
# 运行核心对比实验
python experiments/implementation/run_experiments.py --experiment core_comparison

# 运行所有实验
python experiments/implementation/run_experiments.py --all
```

## ⚠️ 常见问题

### 1. 显存不足

**问题**: `CUDA out of memory`

**解决方案**:
- 使用量化版本：
  ```python
  from transformers import BitsAndBytesConfig
  
  quantization_config = BitsAndBytesConfig(
      load_in_8bit=True,  # 或 load_in_4bit=True
  )
  ```
- 使用CPU模式（较慢）：
  ```python
  system = SystemA_BaselineSequential(device="cpu")
  ```
- 使用更小的模型进行测试

### 2. 模型下载失败

**问题**: 无法连接到Hugging Face

**解决方案**:
```bash
# 方案1: 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方案2: 手动下载
git lfs install
git clone https://huggingface.co/Qwen/Qwen2-7B-Instruct ./cache/models--Qwen--Qwen2-7B-Instruct

# 方案3: 使用modelscope（阿里云）
pip install modelscope
# 然后修改代码使用modelscope加载
```

### 3. 认证错误

**问题**: `401 Unauthorized`

**解决方案**:
```bash
# 登录Hugging Face（如果模型需要认证）
huggingface-cli login

# 或设置token
export HF_TOKEN=your_token_here
```

## 📊 性能优化建议

### 设备分配策略
```python
# ASR模型使用CPU（稳定性好，显存占用少）
asr_device = "cpu"

# LLM模型使用CUDA加速（如果可用）
llm_device = "cuda" if torch.cuda.is_available() else "cpu"

# 创建系统实例
system = SystemA_BaselineSequential(
    llm_device=llm_device  # 只指定LLM设备，ASR自动使用CPU
)
```

### 批处理优化
```python
# 对多个样本使用批处理
batch_size = 4  # 根据显存调整
```

### 缓存优化
- 模型会自动缓存在 `./cache` 目录
- 首次加载后，后续加载会更快

## 📝 实验配置确认

已更新的文件：
- ✅ `experiments/EXPERIMENT_DESIGN.md` - 更新模型说明
- ✅ `experiments/EXPERIMENT_IMPLEMENTATION_PLAN.md` - 更新实施计划
- ✅ `experiments/systems/system_a_baseline.py` - 支持CUDA和新模型

主要改动：
1. 默认LLM模型从 `Qwen/Qwen1.5-0.5B-Chat` 改为 `Qwen/Qwen2-7B-Instruct`
2. ASR模块使用CPU（稳定性好），LLM使用CUDA加速
3. 系统C使用 `Qwen/Qwen2-Audio-7B-Instruct`

## 🎯 下一步

1. **验证环境**: 运行 `test_model_loading.py`
2. **下载模型**: 运行 `download_models.py`（可选）
3. **测试系统A**: 运行 `system_a_baseline.py`
4. **运行实验**: 准备数据后运行完整实验

---

**注意事项**:
- 7B模型需要较大显存，建议使用有至少16GB显存的GPU
- 如果资源受限，可以考虑使用量化或更小的模型进行初步测试
- 确保网络连接稳定，模型下载可能需要时间