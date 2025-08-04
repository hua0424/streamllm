# 实验执行检查清单

## 📋 实验前准备

### 环境配置
- [ ] 确认Python环境已激活 (`conda activate uv`)
- [ ] 安装所有依赖包 (`uv run pip install -r requirements.txt`)
- [ ] 检查GPU/CPU资源可用性
- [ ] 确认硬盘空间充足（至少10GB）

### 数据准备
- [ ] 音频文件已按长度分类存放
  - [ ] `data/processed_audio/length_3s/` (目标: 20个文件)
  - [ ] `data/processed_audio/length_5s/` (目标: 20个文件) 
  - [ ] `data/processed_audio/length_10s/` (目标: 20个文件)
  - [ ] `data/processed_audio/length_15s/` (目标: 20个文件)
  - [ ] `data/processed_audio/length_20s/` (目标: 20个文件)
  - [ ] `data/processed_audio/length_30s/` (目标: 20个文件)
- [ ] Ground Truth转录文本准备完成
- [ ] 创建实验结果目录结构

### 模型配置
- [ ] ASR模型可用性确认 (faster-whisper)
- [ ] LLM模型可用性确认 (Qwen/Qwen1.5-0.5B-Chat)
- [ ] 对比模型下载 (如Qwen2-Audio-7B，可选)

### 配置文件
- [ ] 检查 `experiments/configs/default_config.json`
- [ ] 确认所有实验开关设置正确
- [ ] 验证文件路径配置

## 🧪 实验执行时间线

### 第1天：基础实验 (预计6-8小时)

#### 上午 (4小时)
- [ ] **9:00-10:00** 数据预处理和验证
  ```bash
  python experiments/data_preparation/prepare_test_sets.py
  python experiments/data_preparation/create_ground_truth.py
  ```
- [ ] **10:00-12:00** 实验1：语音长度影响分析
  ```bash
  python experiments/scripts/run_all_experiments.py --experiments 1
  ```

#### 下午 (4小时)  
- [ ] **13:00-15:00** 实验4：ASR模型规模测试
  ```bash
  python experiments/scripts/run_all_experiments.py --experiments 4
  ```
- [ ] **15:00-17:00** 数据初步分析和问题排查

#### 当日检查点
- [ ] 实验1结果文件生成: `experiments/results/exp1_length_impact/results.json`
- [ ] 实验4结果文件生成: `experiments/results/exp4_asr_model_scale/results.json`
- [ ] 无严重错误或异常

### 第2天：核心优化验证 (预计6-8小时)

#### 上午 (4小时)
- [ ] **9:00-11:00** 实验3：消融实验
  ```bash
  python experiments/scripts/run_all_experiments.py --experiments 3
  ```
- [ ] **11:00-12:00** 实验结果初步分析

#### 下午 (4小时)
- [ ] **13:00-16:00** 实验6：延迟-准确率权衡分析
  ```bash
  python experiments/scripts/run_all_experiments.py --experiments 6
  ```
- [ ] **16:00-17:00** 问题调试和参数调优

#### 当日检查点
- [ ] 实验3结果文件生成: `experiments/results/exp3_ablation_study/results.json`
- [ ] 实验6结果文件生成: `experiments/results/exp6_tradeoff_analysis/results.json`
- [ ] 关键指标数据收集完整

### 第3天：对比和鲁棒性测试 (预计6-8小时)

#### 上午 (4小时)
- [ ] **9:00-11:00** 实验2：与原生语音模型对比
  ```bash
  python experiments/scripts/run_all_experiments.py --experiments 2
  ```
- [ ] **11:00-12:00** 实验5：音频质量鲁棒性测试
  ```bash
  python experiments/scripts/run_all_experiments.py --experiments 5
  ```

#### 下午 (4小时)
- [ ] **13:00-15:00** 实验7：并发性能测试
  ```bash
  python experiments/scripts/run_all_experiments.py --experiments 7
  ```
- [ ] **15:00-17:00** 实验数据整理

#### 当日检查点
- [ ] 实验2结果文件生成: `experiments/results/exp2_model_comparison/results.json`
- [ ] 实验5结果文件生成: `experiments/results/exp5_audio_quality/results.json`
- [ ] 实验7结果文件生成: `experiments/results/exp7_concurrent_performance/results.json`

### 第4天：数据分析和可视化 (预计4-6小时)

#### 上午 (3小时)
- [ ] **9:00-10:00** 统计显著性分析
  ```bash
  python experiments/analysis/statistical_analysis.py
  ```
- [ ] **10:00-12:00** 生成图表和可视化
  ```bash
  python experiments/analysis/visualization.py
  ```

#### 下午 (3小时)
- [ ] **13:00-15:00** 生成实验报告
  ```bash
  python experiments/analysis/report_generator.py
  ```
- [ ] **15:00-16:00** 结果验证和质量检查

#### 当日检查点
- [ ] 统计分析报告: `experiments/results/statistical_analysis/statistical_report.md`
- [ ] 图表文件: `experiments/results/final_report/figures/`
- [ ] 最终报告: `experiments/results/final_report/summary.json`

### 第5天：报告整理和验证 (预计4小时)

#### 上午 (2小时)
- [ ] **9:00-11:00** 生成最终实验报告
- [ ] 检查数据完整性和一致性

#### 下午 (2小时)  
- [ ] **13:00-14:00** 验证实验可重复性
- [ ] **14:00-15:00** 准备论文实验章节内容

#### 当日检查点
- [ ] 完整实验报告生成
- [ ] 论文素材准备完成
- [ ] 代码和结果归档

## ✅ 每日质量检查

### 数据质量检查
- [ ] 检查结果文件完整性
- [ ] 验证关键指标合理性
- [ ] 确认无异常值或错误数据
- [ ] 检查日志文件无严重错误

### 实验质量检查
- [ ] 确认实验参数设置正确
- [ ] 验证对照组设置合理
- [ ] 检查样本量充足
- [ ] 确认统计方法适当

### 文档质量检查
- [ ] 实验记录完整
- [ ] 参数配置有文档化
- [ ] 结果文件有清晰命名
- [ ] 中间结果有备份

## 📊 关键指标监控

### 实验1：语音长度影响
- [ ] 各长度组延迟数据收集完整
- [ ] 优化比例计算正确
- [ ] 统计显著性检验通过
- [ ] 相关性分析完成

### 实验2：模型对比  
- [ ] 各模型性能指标收集
- [ ] 资源使用情况记录
- [ ] 准确率对比数据
- [ ] 成本效益分析

### 实验3：消融实验
- [ ] 4种配置结果完整
- [ ] 组件贡献度量化
- [ ] 方差分析通过
- [ ] 效应量计算正确

### 实验4：ASR模型规模
- [ ] 各模型大小测试完成
- [ ] 延迟-准确率权衡数据
- [ ] 帕累托前沿分析
- [ ] 最优配置推荐

### 实验5：音频质量鲁棒性
- [ ] 不同质量条件测试
- [ ] 性能退化量化
- [ ] 鲁棒性评分计算
- [ ] 适用场景分析

### 实验6：参数权衡分析
- [ ] 参数网格搜索完成
- [ ] 最优参数组合识别
- [ ] 权衡曲线绘制
- [ ] 配置建议生成

### 实验7：并发性能
- [ ] 不同并发数测试
- [ ] 系统资源监控
- [ ] 扩展性评估
- [ ] 瓶颈分析

## 🚨 常见问题排查

### 内存相关
- [ ] **问题**: GPU内存不足
  - **解决**: 减小批处理大小或使用CPU模式
- [ ] **问题**: 系统内存不足  
  - **解决**: 关闭其他程序，增加虚拟内存

### 模型加载
- [ ] **问题**: 模型下载失败
  - **解决**: 检查网络连接，使用本地模型路径
- [ ] **问题**: 模型版本不兼容
  - **解决**: 更新模型版本或调整代码

### 数据问题
- [ ] **问题**: 音频格式不支持
  - **解决**: 使用ffmpeg转换为16kHz WAV
- [ ] **问题**: 文件路径错误
  - **解决**: 使用绝对路径，检查文件存在性

### 性能问题
- [ ] **问题**: 实验运行太慢
  - **解决**: 减少样本数量，使用更小模型
- [ ] **问题**: 延迟测量不准确
  - **解决**: 多次运行求平均，检查计时代码

## 📝 实验记录模板

### 每日实验记录
```
日期: ________
执行人: ________
实验编号: ________

执行的实验:
- [ ] 实验1: 语音长度影响
- [ ] 实验2: 模型对比
- [ ] 实验3: 消融实验
- [ ] 实验4: ASR模型规模
- [ ] 实验5: 音频质量
- [ ] 实验6: 参数权衡
- [ ] 实验7: 并发性能

遇到的问题:
1. 
2.
3.

解决方案:
1.
2. 
3.

关键发现:
1.
2.
3.

明天计划:
1.
2.
3.
```

### 实验结果检查
```
实验名称: ________
结果文件: ________
数据完整性: [ ] 完整 [ ] 不完整
关键指标:
- 平均延迟: ____ms
- 标准差: ____ms  
- 样本数量: ____
- 显著性: [ ] 显著 [ ] 不显著

质量评估: [ ] 优秀 [ ] 良好 [ ] 需改进
备注: ________________
```

## 🎯 成功标准

### 数据完整性 (必须达成)
- [ ] 所有计划实验完成率 ≥ 90%
- [ ] 关键指标数据收集完整率 ≥ 95%
- [ ] 统计检验通过率 ≥ 80%

### 性能提升 (预期目标)
- [ ] 首Token延迟优化 ≥ 30%
- [ ] ASR准确率保持 ≥ 90%
- [ ] 系统稳定性 ≥ 95%

### 论文支撑 (质量要求)
- [ ] 至少3个显著性结果
- [ ] 完整的消融实验数据
- [ ] 与基线方法的对比数据
- [ ] 鲁棒性和扩展性证据

## 📞 应急联系

### 技术支持
- 项目维护者: [联系方式]
- 技术问题: GitHub Issues
- 紧急情况: [紧急联系方式]

### 备用方案
- [ ] 如果GPU不可用，切换到CPU模式
- [ ] 如果网络模型无法下载，使用本地模型
- [ ] 如果时间不够，优先完成核心实验1、3、4

---

**最后更新**: 2025-08-03  
**版本**: v1.0  
**审核人**: [审核人姓名]