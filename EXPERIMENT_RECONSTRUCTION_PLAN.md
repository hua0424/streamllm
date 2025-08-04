# 实验重构执行计划 - 级联式语音对话系统延迟优化

## 📋 重构概览

**日期**: 2025-08-04  
**目标**: 根据新的实验设计方案重构现有实验框架  
**原因**: 为硕士论文《级联式语音对话系统的延迟优化》提供更系统、更学术化的实验支持

## 🔍 新旧实验设计对比分析

### 旧实验设计要点
- **主要实验**: 
  - 实验1：语音长度影响实验
  - 实验3：消融实验
  - 实验4：ASR模型规模实验
- **系统对比**: 主要是基线vs优化版本的二元对比
- **指标**: 首token延迟、优化比例、ASR准确率等
- **实验数量**: 相对简单，主要关注长度和消融

### 新实验设计要点
- **三系统对比架构**:
  - 系统A：基线串行系统 (Baseline Sequential System)
  - 系统B：KV缓存预填充系统 (Proposed System) 
  - 系统C：理想化端到端系统 (End-to-End Oracle System)
- **四核心实验**:
  - 实验一：核心性能与质量对比
  - 实验二：输入长度对优化效果的影响分析
  - 实验三：消融研究（引入系统A'仅流式ASR）
  - 实验四：案例分析（时序对比图）
- **更系统的指标体系**:
  - 首Token延迟 (TTFT) - 主要效率指标
  - ASR准确率 (WER) - 质量指标  
  - 回复一致性 (BERTScore) - 质量指标
  - 计算资源开销 - 资源指标

### 主要差异总结
1. **系统架构**: 从二元对比升级为三/四系统对比
2. **实验深度**: 新增端到端系统对比和精确消融研究
3. **学术严谨性**: 更加符合学术论文的实验设计标准
4. **指标完整性**: 涵盖效率、质量、资源三个维度

## 📝 具体重构任务清单

### 阶段1: 架构调整和文档更新
- [x] ✅ 分析新旧实验设计差异
- [ ] 🔄 更新 `EXPERIMENT_IMPLEMENTATION_PLAN.md` 
- [ ] 📝 重新设计实验代码架构
- [ ] 📋 更新实验配置文件结构

### 阶段2: 系统实现代码调整
- [ ] 🏗️ 实现系统A（基线串行系统）
- [ ] 🏗️ 完善系统B（KV缓存预填充系统）
- [ ] 🏗️ 实现系统C（端到端系统接口）
- [ ] 🏗️ 实现系统A'（仅流式ASR系统）用于消融研究

### 阶段3: 实验代码重构
- [ ] 🧪 重构核心性能与质量对比实验
- [ ] 🧪 调整输入长度影响分析实验
- [ ] 🧪 重新设计消融研究实验
- [ ] 🧪 新增案例分析实验功能

### 阶段4: 指标与分析工具
- [ ] 📊 实现TTFT精确计时
- [ ] 📊 实现WER计算工具
- [ ] 📊 实现BERTScore计算工具
- [ ] 📊 实现资源监控工具
- [ ] 📈 更新图表生成器以匹配新实验

### 阶段5: 验证与优化
- [ ] ✅ 运行小规模验证测试
- [ ] 🔧 修复发现的问题
- [ ] 📚 更新文档和使用说明
- [ ] 🧹 清理冗余代码

## 🏗️ 新实验代码架构设计

```
experiments/
├── systems/                     # 新增：系统实现
│   ├── __init__.py
│   ├── system_a_baseline.py    # 系统A：基线串行系统
│   ├── system_b_proposed.py    # 系统B：KV缓存预填充系统
│   ├── system_c_endtoend.py    # 系统C：端到端系统
│   └── system_a_prime.py       # 系统A'：仅流式ASR系统
├── experiments/                 # 调整：实验实现
│   ├── __init__.py
│   ├── base_experiment.py      # 基础实验类
│   ├── exp1_core_comparison.py # 实验1：核心性能与质量对比
│   ├── exp2_length_analysis.py # 实验2：输入长度影响分析
│   ├── exp3_ablation_study.py  # 实验3：消融研究
│   ├── exp4_case_analysis.py   # 实验4：案例分析
│   └── experiment_config.py    # 实验配置
├── metrics/                     # 新增：指标计算模块
│   ├── __init__.py
│   ├── ttft_calculator.py      # TTFT计算器
│   ├── wer_calculator.py       # WER计算器
│   ├── bertscore_calculator.py # BERTScore计算器
│   └── resource_monitor.py     # 资源监控器
├── analysis/                    # 保留：结果分析代码
│   ├── __init__.py
│   ├── statistical_analyzer.py
│   ├── chart_generator.py      # 需更新图表类型
│   ├── table_generator.py
│   └── result_processor.py
├── paper_tools/                 # 保留：论文辅助工具
│   ├── __init__.py
│   ├── report_generator.py
│   ├── latex_exporter.py
│   └── template_generator.py
├── validation/                  # 保留：验证和测试
│   ├── __init__.py
│   ├── small_dataset.py
│   ├── test_experiments.py
│   └── verification_tools.py
└── results/                     # 保留：实验结果存储
    ├── raw_data/
    ├── processed/
    ├── figures/
    ├── tables/
    └── reports/
```

## 📊 新指标体系实现要点

### 1. 首Token延迟 (TTFT)
```python
# 精确计时实现
import time
T_speech_end = time.perf_counter()  # 语音结束时间
T_first_token = time.perf_counter() # 首Token生成时间
TTFT = T_first_token - T_speech_end
```

### 2. ASR准确率 (WER)
```python
# 需要实现标准WER计算
WER = (S + D + I) / N
# S: 替换错误数, D: 删除错误数, I: 插入错误数, N: 参考词数
```

### 3. 回复一致性 (BERTScore)
```python
# 需要集成BERTScore库
from bert_score import score
P, R, F1 = score(cands, refs, lang="zh", verbose=True)
```

## 🔄 现有代码复用策略

### 可直接复用的模块
- `src/asr/faster_whisper_streamer.py` - ASR处理器
- `src/llm/stream_llm_inference.py` - LLM推理引擎  
- `src/utils/audio2stream.py` - 音频流处理
- `src/utils/logging_utils.py` - 日志工具
- `test_streaming.py` - 作为系统B的基础

### 需要调整的模块
- `experiments/implementation/` - 重构为新的四个实验
- `experiments/analysis/` - 更新图表类型和统计方法
- `experiments/paper_tools/` - 适配新的论文结构

### 需要新增的模块
- `experiments/systems/` - 四个系统实现
- `experiments/metrics/` - 专门的指标计算模块

## ⚠️ 关键技术实现注意点

### 1. 系统C（端到端系统）实现
- 需要集成 Qwen2-Audio 或类似模型
- 如果模型资源受限，可考虑使用API或模拟实现

### 2. KV缓存处理策略
- ASR结果修正时的缓存更新机制
- 需要明确文档说明所采用的策略

### 3. 精确计时
- 使用 `time.perf_counter()` 确保高精度
- 记录详细的时间戳用于分析

### 4. 实验环境记录
- 详细记录软件版本和硬件配置
- 确保实验可重现性

## 📋 进度跟踪

- [ ] **Phase 1**: 架构调整和文档更新 (预计耗时: 2小时)
- [ ] **Phase 2**: 系统实现代码调整 (预计耗时: 4小时)  
- [ ] **Phase 3**: 实验代码重构 (预计耗时: 4小时)
- [ ] **Phase 4**: 指标与分析工具 (预计耗时: 3小时)
- [ ] **Phase 5**: 验证与优化 (预计耗时: 2小时)

**总预计耗时**: 15小时

## 📝 修改记录

### 2025-08-04

#### 已完成工作
- ✅ **阅读新实验设计文档**: 深入理解了新的四个实验和三系统对比架构
  - 系统A：基线串行系统
  - 系统B：KV缓存预填充系统 (本文方案)
  - 系统C：理想化端到端系统
  - 系统A'：仅流式ASR系统 (消融研究用)

- ✅ **创建重构执行计划**: 建立了完整的15小时重构计划，分5个阶段实施

- ✅ **新旧实验设计对比分析**: 识别了关键差异
  - 从二元对比升级为三/四系统对比
  - 从3个简单实验扩展为4个系统性实验
  - 新增质量评估指标 (WER, BERTScore)
  - 更加符合学术论文标准

- ✅ **更新实验实施计画文档**: 完全重写了 `EXPERIMENT_IMPLEMENTATION_PLAN.md`
  - 更新为6个阶段的详细计划
  - 重新设计实验代码架构 (新增systems/和metrics/目录)
  - 定义新的指标体系 (效率、质量、资源、统计指标)
  - 规划12+个图表和8+个LaTeX表格
  - 制定严格的验收标准

#### 重要发现
1. **现有代码可复用性高**: `BaseExperiment`、`LengthImpactExperiment`、`AblationExperiment` 基础良好
2. **主要新增需求**: 四系统实现、指标计算模块、案例分析实验
3. **质量评估是重点**: 需要新增WER和BERTScore计算能力
4. **论文支持要求更高**: 需要生成更多类型的学术图表和表格

- ✅ **完成现有代码分析**: 详细检查了关键文件的实现和功能
  - `base_experiment.py`: 基础实验类功能完善，可直接复用
  - `length_impact_exp.py`: 基本符合实验二需求，需微调
  - `ablation_exp.py`: 需要重构以支持四系统对比
  - `run_experiments.py`: 实验管理器功能良好，需更新实验列表
  - `chart_generator.py`: 图表生成功能强大，需新增部分图表类型

#### 具体修改计划
**🆕 需要新增的文件**:
1. `experiments/systems/system_a_baseline.py` - 基线串行系统
2. `experiments/systems/system_b_proposed.py` - KV缓存预填充系统
3. `experiments/systems/system_c_endtoend.py` - 端到端理想系统
4. `experiments/systems/system_a_prime.py` - 仅流式ASR系统
5. `experiments/metrics/ttft_calculator.py` - TTFT精确计时器
6. `experiments/metrics/wer_calculator.py` - WER计算器
7. `experiments/metrics/bertscore_calculator.py` - BERTScore计算器
8. `experiments/metrics/resource_monitor.py` - 资源监控器
9. `experiments/implementation/exp1_core_comparison.py` - 实验一：核心性能质量对比
10. `experiments/implementation/exp4_case_analysis.py` - 实验四：案例分析

**🔧 需要修改的文件**:
1. `experiments/implementation/ablation_exp.py` - 重构为四系统消融研究
2. `experiments/implementation/length_impact_exp.py` - 微调以匹配新指标
3. `experiments/implementation/run_experiments.py` - 更新为四个核心实验
4. `experiments/analysis/chart_generator.py` - 新增TTFT箱形图、时序图等

**✅ 可直接复用的文件**:
1. `experiments/implementation/base_experiment.py` - 基础实验框架
2. `experiments/analysis/statistical_analyzer.py` - 统计分析工具
3. `experiments/analysis/table_generator.py` - 表格生成器
4. `experiments/paper_tools/` - 论文辅助工具

#### 下一步计划
- 🔄 **当前进行**: 开始实施系统和实验代码的修改
- 📋 **接下来**: 从四个系统实现开始，然后是指标计算模块

---

*此文档将持续更新，记录整个重构过程的详细内容和结果*