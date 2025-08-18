# 正确的TTFT测量流程

  # 正确的实验流程示例
  system = SystemA_BaselineSequential()

  # 1. 预热阶段（耗时不计入实验）
  system.warmup()

  # 2. 正式实验（此时TTFT不含模型加载时间）
  for sample in test_samples:
      result = system.process_sample(sample, skip_warmup=True)  # 
  跳过预热
      ttft = result['ttft_ms']  # 这是纯推理延迟
      system.reset()  # 重置状态但不重新加载模型

  🚀 实验建议

  1. 批量实验时：
    - 在开始前调用一次 warmup()
    - 所有后续测试使用 skip_warmup=True
    - 第一个样本的结果可以考虑丢弃
  2. 对比实验时：
    - 系统A和系统B都要进行相同的预热
    - 确保测试环境完全一致
    - 记录预热时间用于分析
  3. 论文报告时：
    - 明确说明排除了模型加载时间
    - 报告预热阶段的耗时作为系统初始化成本
    - 强调TTFT测量的是纯推理性能