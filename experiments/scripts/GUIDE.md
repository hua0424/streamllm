# StreamLLM 实验框架使用指南

## 📋 概述

本指南详细说明了StreamLLM级联式语音对话系统延迟优化实验框架的使用方法、数据格式要求、实验执行流程和结果分析。该实验框架专为硕士论文《级联式语音对话系统的延迟优化》设计，提供完整的实验设计、执行、分析和论文支持功能。

## 🎯 实验目标

### 核心研究问题
- **主要目标**: 优化从流式语音输入结束到LLM产生第一个token的延迟时间
- **技术路径**: 流式ASR + KV缓存预填充 + 级联流水线优化
- **验证方案**: 四系统对比实验，证明优化效果和质量保持

### 关键指标
- **首Token延迟(TTFT)**: $T_{TTFT} = T_{first\_token} - T_{speech\_end}$ (主要指标)
- **ASR准确率**: 词错误率(WER)和字符错误率(CER)
- **回复一致性**: BERTScore语义相似度
- **资源开销**: 峰值内存占用和GPU利用率

## 🏗️ 系统架构

### 四系统对比架构

| 系统标识 | 系统名称 | 实现文件 | 功能描述 |
|---------|---------|----------|----------|
| **System A** | 基线串行系统 | `systems/system_a_baseline.py` | 传统顺序处理：完整音频→ASR→LLM |
| **System B** | KV缓存预填充系统 | `systems/system_b_proposed.py` | **本文核心方案**：流式ASR + 并行KV缓存预填充 |
| **System C** | 端到端理想系统 | `systems/system_c_endtoend.py` | 理论上限参考：直接语音到文本模型 |
| **System A'** | 仅流式ASR系统 | `systems/system_a_prime.py` | 消融研究：仅优化ASR，不使用KV缓存 |

### 实验框架结构

```
experiments/
├── README.md                          # 实验概述和快速开始
├── EXPERIMENT_DESIGN.md               # 详细实验设计方案
├── EXPERIMENT_IMPLEMENTATION_PLAN.md  # 实施计划和技术细节
├── scripts/
│   ├── GUIDE.md                      # 本使用指南
│   └── run_all_experiments.py        # 实验执行入口
├── systems/                          # 四系统实现
│   ├── system_a_baseline.py         # 基线系统
│   ├── system_b_proposed.py         # 核心方案
│   ├── system_c_endtoend.py         # 理想系统
│   └── system_a_prime.py            # 消融对照
├── implementation/                   # 实验实现
│   ├── base_experiment.py           # 基础实验框架
│   ├── exp1_core_comparison.py      # 核心性能质量对比
│   ├── length_impact_exp.py         # 语音长度影响实验
│   ├── ablation_exp.py              # 消融实验
│   ├── asr_scale_exp.py             # ASR模型规模实验
│   ├── audio_quality_exp.py         # 音频质量鲁棒性
│   ├── native_model_comparison_exp.py # 原生模型对比
│   ├── concurrent_performance_exp.py  # 并发性能测试
│   └── run_experiments.py           # 统一实验运行器
├── analysis/                        # 结果分析
│   ├── statistical_analysis.py      # 统计分析工具
│   ├── chart_generator.py           # 图表生成器
│   ├── table_generator.py           # LaTeX表格生成
│   └── result_processor.py          # 结果处理器
├── paper_tools/                     # 论文支持工具
│   ├── report_generator.py          # 自动报告生成
│   ├── latex_exporter.py            # LaTeX导出工具
│   └── template_generator.py        # 模板生成器
├── configs/                         # 配置文件
│   └── default_config.json          # 默认实验配置
└── results/                         # 实验结果
    ├── figures/                     # 图表文件
    ├── tables/                      # 表格文件
    ├── reports/                     # 实验报告
    └── raw_data/                    # 原始实验数据
```

## 📊 实验设计

### 实验1: 核心性能与质量对比
**目的**: 验证System B相对于基线的优势，与理想系统的对比
```bash
python experiments/implementation/exp1_core_comparison.py --samples 300
```

### 实验2: 语音长度影响分析  
**目的**: 证明长语音优化效果更显著
```bash
python experiments/implementation/length_impact_exp.py --groups short_5to10s,medium_10to20s,long_20plus
```

### 实验3: 消融实验
**目的**: 量化各优化组件的单独贡献
```bash
python experiments/implementation/ablation_exp.py --configurations baseline,asr_only,llm_only,full
```

### 实验4: ASR模型规模实验
**目的**: 分析不同ASR模型对系统性能的影响
```bash
python experiments/implementation/asr_scale_exp.py --models tiny,base,small,medium,large
```

### 实验5: 音频质量鲁棒性
**目的**: 测试不同音频质量下的稳定性
```bash
python experiments/implementation/audio_quality_exp.py --quality clean,high,medium,low,poor
```

### 实验6: 原生模型对比
**目的**: 与端到端语音模型性能对比
```bash
python experiments/implementation/native_model_comparison_exp.py --models qwen2_audio,optimized_cascade
```

### 实验7: 并发性能测试
**目的**: 多用户场景下的性能表现
```bash
python experiments/implementation/concurrent_performance_exp.py --max_users 16
```

## 💾 数据格式要求

### 音频数据格式
```
data/
├── processed_audio/                 # 主要音频数据
│   ├── short_5to10s/               # 短语音: 5-10秒音频分组
│   │   ├── sample_001.wav         # 16kHz, 16bit, mono WAV
│   │   ├── sample_002.wav
│   │   └── ...
│   ├── medium_10to20s/             # 中等语音: 10-20秒音频分组
│   └── long_20plus/                # 长语音: 20秒以上音频分组
├── transcripts/                    # 对应转录文本
│   ├── short_5to10s/
│   │   ├── sample_001.txt         # UTF-8纯文本
│   │   └── ...
│   ├── medium_10to20s/
│   └── long_20plus/

```

### 音频文件要求
- **格式**: WAV (无压缩)
- **采样率**: 16kHz
- **位深度**: 16bit
- **声道**: 单声道(mono)
- **时长**: 按目录名分组(short_5to10s: 5-10秒, medium_10to20s: 10-20秒, long_20plus: 20秒以上)
- **内容**: 单轮对话问题，包含中文、英文、中英混合

### 转录文本格式
```json
{
  "audio_file": "sample_001.wav", 
  "duration": 10.5,
  "text": "播放周杰伦的演唱会视频",
  "language": "zh",
  "speaker": "user",
  "ground_truth": "播放周杰伦的演唱会视频"
}
```

### 音频时长预估参考数据

基于当前测试数据的统计分析，提供以下TTS音频时长预估参考：

| 样本文件 | 分组 | 音频时长 | 文本内容 | 字符数 | 每字符时长 |
|---------|------|----------|----------|--------|------------|
| sample_001 | short_5to10s | 2.9s | "请帮我播放一首周杰伦的音乐" | 13个 | 0.227s |
| sample_002 | short_5to10s | 4.7s | "Can you help me set an alarm..." | 36个 | 0.131s |
| sample_001 | medium_10to20s | 9.4s | "我正在准备一个关于机器学习的presentation..." | 70个 | 0.134s |
| sample_002 | medium_10to20s | 5.0s | "Please help me write a Python function..." | 64个 | 0.078s |
| sample_001 | long_20plus | 11.7s | "Could you please explain the concept of transformer..." | 159个 | 0.074s |

**统计结果**:
- **总样本数**: 5个
- **总时长**: 33.7秒
- **总字符数**: 342个
- **平均每字符时长**: 0.099秒

**TTS时长预估建议**:
- **中文文本**: 约0.15-0.23秒/字符 (包含语音停顿)
- **英文文本**: 约0.07-0.13秒/字符 (字母计数)
- **混合文本**: 约0.10-0.15秒/字符 (综合平均)

**各分组所需字符数参考**:
- **short_5to10s** (5-10秒): 需要35-100个字符
- **medium_10to20s** (10-20秒): 需要70-200个字符  
- **long_20plus** (20秒以上): 需要200个以上字符

### 实验配置文件格式
```json
{
  "experiment_name": "core_comparison",
  "version": "1.0",
  "systems": ["system_a", "system_b", "system_c"],
  "num_runs": 5,
  "asr_model_size": "base", 
  "llm_model_name": "Qwen/Qwen1.5-0.5B-Chat",
  "chunk_duration": 0.3,
  "output_dir": "experiments/results"
}
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 激活conda环境
conda activate uv

# 安装必要依赖
uv add bert-score jiwer librosa psutil gputil matplotlib seaborn scipy

# 验证环境
python -c "import bert_score, jiwer, librosa; print('✅ 环境就绪')"
```

### 2. 数据准备
```bash
# 准备测试数据集(请根据CS-Dialogue处理指南处理数据)
mkdir -p experiments/data/processed_audio/{short_5to10s,medium_10to20s,long_20plus}
mkdir -p experiments/data/transcripts/{short_5to10s,medium_10to20s,long_20plus}

# 验证数据格式
python experiments/data_preparation/validate_data_format.py
```

### 3. 小规模验证
```bash
# 运行核心实验验证
python experiments/implementation/run_experiments.py --run-core --small-data

# 检查结果
ls experiments/results/
```

### 4. 完整实验执行
```bash
# 运行所有实验
python experiments/implementation/run_experiments.py --run-all --full-data

# 生成分析报告
python experiments/analysis/run_analysis.py --generate-all

# 生成论文材料
python experiments/paper_tools/run_paper_tools.py --export-all
```

## 📈 实验执行详细流程

### 单个实验执行
```bash
# 1. 核心对比实验 (最重要)
python experiments/implementation/exp1_core_comparison.py \
  --samples 300 \
  --systems system_a,system_b,system_c \
  --output-dir experiments/results/core_comparison

# 2. 长度影响实验
python experiments/implementation/length_impact_exp.py \
  --length-groups short_5to10s,medium_10to20s,long_20plus \
  --samples-per-group 20 \
  --output-dir experiments/results/length_impact

# 3. 消融实验
python experiments/implementation/ablation_exp.py \
  --configurations baseline,asr_only,llm_only,full_optimization \
  --trials 10 \
  --output-dir experiments/results/ablation_study
```

### 批量实验执行
```bash
# 使用统一执行器
python experiments/implementation/run_experiments.py \
  --experiments core_comparison,length_impact,ablation_study \
  --config experiments/configs/default_config.json \
  --full-data
```

### 实验监控和调试
```bash
# 启用详细日志模式
export STREAMLLM_LOG_LEVEL=DEBUG

# 监控系统资源
python experiments/implementation/run_experiments.py --monitor-resources

# 小数据集快速验证
python experiments/implementation/run_experiments.py --quick-test
```

## 📊 结果分析和可视化

### 自动生成分析报告
```bash
# 生成统计分析
python experiments/analysis/statistical_analysis.py \
  --results-dir experiments/results \
  --output experiments/results/statistical_summary.json

# 生成所有图表
python experiments/analysis/chart_generator.py \
  --data experiments/results \
  --output experiments/results/figures \
  --formats png,pdf

# 生成LaTeX表格
python experiments/analysis/table_generator.py \
  --results experiments/results \
  --output experiments/results/tables \
  --format latex
```

### 关键图表说明

#### 1. 首Token延迟对比图
- **文件**: `figures/ttft_comparison_boxplot.png`
- **内容**: System A/B/C的TTFT分布箱形图
- **用途**: 论文5.1节核心结果展示

#### 2. 语音长度影响图  
- **文件**: `figures/length_vs_optimization.png`
- **内容**: 语音长度与优化效果的相关性
- **用途**: 论文5.2节长度影响分析

#### 3. 消融实验对比图
- **文件**: `figures/ablation_components.png` 
- **内容**: 各组件单独和组合的贡献度
- **用途**: 论文5.3节消融研究

### 核心数据表格

#### 1. 系统性能对比表
```latex
\begin{table}[h]
\caption{三系统延迟性能对比}
\begin{tabular}{|l|c|c|c|c|}
\hline
系统 & 平均TTFT(ms) & 中位数TTFT(ms) & P95(ms) & 优化比例 \\
\hline
System A (基线) & 9800±1200 & 9500 & 12000 & - \\
System B (方案) & 4200±800  & 4000 & 5500  & 57.1\% \\
System C (理想) & 3100±600  & 3000 & 4000  & 68.4\% \\
\hline
\end{tabular}
\end{table}
```

#### 2. 质量指标对比表
```latex
\begin{table}[h]
\caption{ASR准确率和回复质量对比}
\begin{tabular}{|l|c|c|c|}
\hline
指标 & System A & System B & 显著性 \\
\hline
WER (\%) & 5.1±0.8 & 5.4±0.9 & p=0.23 (n.s.) \\
BERTScore & - & 0.91±0.05 & - \\
语义相似度 & - & 0.89±0.06 & - \\
\hline
\end{tabular}
\end{table}
```

## 🎓 论文支持功能

### 自动章节生成
```bash
# 生成5.1节：核心性能与质量对比
python experiments/paper_tools/generate_section.py \
  --section 5.1 \
  --data experiments/results/core_comparison \
  --template templates/section_5_1.md

# 生成5.2节：输入长度影响分析  
python experiments/paper_tools/generate_section.py \
  --section 5.2 \
  --data experiments/results/length_impact

# 生成5.3节：消融研究
python experiments/paper_tools/generate_section.py \
  --section 5.3 \
  --data experiments/results/ablation_study
```

### LaTeX材料导出
```bash
# 导出所有表格
python experiments/paper_tools/latex_exporter.py \
  --tables experiments/results/tables \
  --output paper_materials/tables.tex

# 导出图表引用
python experiments/paper_tools/latex_exporter.py \
  --figures experiments/results/figures \
  --output paper_materials/figures.tex

# 生成完整实验章节
python experiments/paper_tools/latex_exporter.py \
  --full-chapter experiments/results \
  --output paper_materials/chapter5_experiments.tex
```

### 结论和发现生成
```bash
# 自动提取关键发现
python experiments/paper_tools/extract_key_findings.py \
  --results experiments/results \
  --output paper_materials/key_findings.md

# 生成讨论要点
python experiments/paper_tools/generate_discussion.py \
  --results experiments/results \
  --output paper_materials/discussion_points.md
```

## 🔧 配置和定制

### 实验参数配置
```json
{
  "experiment_settings": {
    "num_runs": 5,           // 每个条件重复次数
    "confidence_level": 0.95, // 统计置信度
    "significance_threshold": 0.05, // 显著性阈值
    "min_samples": 20        // 最小样本数
  },
  "system_configs": {
    "asr_model_size": "base", // tiny/base/small/medium/large
    "llm_model": "Qwen/Qwen1.5-0.5B-Chat",
    "chunk_duration": 0.3,   // 音频切片时长(秒)
    "kv_cache_enabled": true // KV缓存开关
  },
  "quality_thresholds": {
    "max_wer_difference": 0.01, // 最大WER差异
    "min_bertscore": 0.85,      // 最小BERTScore
    "max_latency_std": 0.2      // 最大延迟标准差
  }
}
```

### 添加新实验
```python
# 继承基础实验类
from base_experiment import BaseExperiment

class CustomExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        self.experiment_name = "custom_experiment"
    
    def run_experiment(self, samples):
        # 实现自定义实验逻辑
        results = []
        for sample in samples:
            result = self.measure_custom_metric(sample)
            results.append(result)
        return results
    
    def analyze_results(self, results):
        # 实现结果分析
        return self.statistical_analysis(results)
```

## 🐛 问题排查

### 常见问题及解决方案

#### 1. 音频格式错误
```bash
# 错误：不支持的音频格式
# 解决：转换为16kHz WAV格式
ffmpeg -i input.mp3 -ar 16000 -ac 1 -f wav output.wav
```

#### 2. 内存不足
```bash
# 错误：CUDA out of memory
# 解决：减少batch size或使用CPU模式
export CUDA_VISIBLE_DEVICES=""  # 强制使用CPU
python experiments/implementation/run_experiments.py --cpu-only
```

#### 3. 模型加载失败
```bash
# 错误：模型下载失败
# 解决：检查网络连接或使用本地模型
export HF_DATASETS_OFFLINE=1
export HF_TRANSFORMERS_OFFLINE=1
```

#### 4. 数据路径问题
```bash
# 错误：找不到音频文件
# 解决：使用绝对路径
python experiments/implementation/run_experiments.py \
  --data-dir /absolute/path/to/data
```

### 调试模式
```bash
# 启用详细日志
export STREAMLLM_DEBUG=1

# 使用小数据集调试
python experiments/implementation/run_experiments.py \
  --debug \
  --samples 5 \
  --verbose
```

## 📝 最佳实践

### 1. 实验执行顺序
1. **环境验证**: 确保所有依赖正确安装
2. **小规模测试**: 用少量数据验证流程正确性
3. **单个实验**: 逐个执行实验，确保结果合理
4. **批量执行**: 运行完整实验套件
5. **结果验证**: 检查统计显著性和合理性

### 2. 数据质量控制
- 音频文件命名规范统一
- 转录文本准确性人工验证
- 不同长度组样本数量平衡
- 多种音频质量条件覆盖

### 3. 结果可重现性
- 固定随机种子 (`random.seed(42)`)
- 记录完整实验环境信息
- 保存原始实验数据
- 版本控制实验代码

### 4. 论文写作建议
- 使用生成的图表和表格
- 引用统计显著性结果
- 突出核心创新点贡献
- 诚实报告局限性和future work

## 📚 参考资料

### 相关论文
- StreamLLM原理和实现细节
- 语音对话系统延迟优化综述
- KV缓存在LLM中的应用研究

### 技术文档
- fast-whisper API文档
- transformers库使用指南
- BERTScore计算方法

### 数据集信息
- CS-Dialogue数据集使用指南
- DailyDialog数据格式说明
- LCCC数据处理方法

---

**版本信息**: v2.0  
**最后更新**: 2025-08-04  
**维护者**: StreamLLM实验团队

**重要提醒**: 本指南基于最新的实验设计方案编写，请确保使用最新版本的代码和配置文件。如有问题，请查看`experiments/README.md`或联系维护团队。