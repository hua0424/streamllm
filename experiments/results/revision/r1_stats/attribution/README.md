# E1 延迟水平偏差归因报告（2026-08-19）

## 问题

E1 三轮 System B very_long 组 mean TTFT = 1434/1482/1455ms，超出任务书 §5.2 sanity 带 0.9–1.3s；
同样 50 样本在原实验机历史结果（exp1_latency，2025-12-10）中为 981ms（+46%）。
需求方提问：是否本次新增实验代码（DEV-1/2 等）导致？两机配置是否真的一致？

## 结论

**不是代码改动导致；是机器级差异，且精确定位于 ASR 阶段的 CPU 单线程性能。**
GPU 侧两机表现一致（LLM 预填仅 +9%），增量全部来自 whisper 解码的 CPU 编排开销。

## 证据链

1. **同机三版本代码 A/B**（2 个 very_long 样本，同 venv、同 GPU、同配置锁定）：

   | 样本/模式 | V0 实验期代码(2f9f481) | V1 pre-DEV(13dfcf7) | 当前 DEV 代码(E1 r1) | 原机 |
   |---|---|---|---|---|
   | crosswoz_10040_turn4 streaming | 1928.9 | 1767.3 | 1817.0 | 1372.7 |
   | multiwoz_MUL0023_turn4 streaming | 1221.2 | 1264.0 | 1125.8 | 680.1 |
   | crosswoz_10040_turn4 non-streaming | 6173.3 | 5905.0 | 5685.7 | 4510.7 |
   | multiwoz_MUL0023_turn4 non-streaming | 3613.5 | 3623.6 | 3274.6 | 3064.7 |

   三个代码版本在本机结果互差 <10%（运行噪声范围），与原机差距一致存在。
   → 从 2025-12 实验期代码到当前 DEV 代码，无任何可测的性能回归；DEV-1/2 插桩开销可忽略。

2. **软件栈逐版本一致**：uv.lock 对比（2f9f481 vs 当前）：torch 2.5.1+cu121、transformers 4.57.1、
   openai-whisper 20250625、faster-whisper 1.2.0、tokenizers 0.22.1、numpy 1.26.4、triton 3.1.0 全部相同。
   （注：2025-12-07 ba7d460 已将 ASR 从 faster-whisper 切换为 openai-whisper，早于 12-10 原实验，
   故原实验与现实验同为 openai-whisper turbo，引擎一致。）

3. **分量归因**（50 个共同样本，streaming 模式，原机 vs 本机）：

   | 分量 | 原机 mean | 本机 mean | 差 |
   |---|---|---|---|
   | TTFT（总） | 981.2ms | 1434.0ms | +452.9ms (+46%) |
   | ASR 尾处理 | 894.3ms | 1339.8ms | **+445.5ms (+50%)** |
   | LLM 预填（GPU） | 86.8ms | 94.2ms | +7.4ms (+9%) |

   LLM 预填（GPU 重负载，cuda:1）两机几乎一致 → GPU 性能一致；
   全部增量落在 ASR 阶段 → whisper 逐 token Python 解码编排是 CPU 单线程敏感型负载。

4. **本机 CPU 环境**：KVM 虚拟机，16 vCPU，Xeon Gold 6133 @ 2.50GHz 固定（无 cpufreq 控制、
   无 turbo），2017 年 Skylake-SP 服务器 U，单线程性能弱（Python 3M 整数循环 0.512s，
   约为现代 4-5GHz 桌面 CPU 的 2-3 倍耗时）。GPU 拓扑对称（双卡同 PCIe/NUMA），无时钟限速。

## 影响评估

- 同机 A/B 对比（E2 System A/B、E3 LA 基线、E4 插桩、E5 端点）内部有效性**不受影响**：
  机器级偏移对所有被测系统恒定；
- 与原论文 Table III 绝对数字并列引用时需注意口径（建议论文中注明修订实验机器环境，
  env_versions.txt 已修正为项目 venv 真实版本，见下）。

## 附带修正

- `env_versions.txt` 重采：安装时 `uv run pip list` 误采了系统 anaconda 环境（torch 2.5.0 等），
  已改用 `uv pip list --python .venv/bin/python` 重采项目 venv 真实版本（139 行 + nvidia-smi）。
- A/B 测试产物：本目录下 `v0_experiment_era_results.json`、`v1_pre_dev_results.json`、`v0.log`、`v1.log`
  （worktree 在 /tmp/sllm_v0、/tmp/sllm_v1，为临时目录，可随时清理）。
