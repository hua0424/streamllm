# SCI 3–4 区补实验目录

本目录保存第二篇论文的新增补实验代码、协议和 GPU 执行说明。它与原始 `experiments/scripts/` 及 `experiments/results/*.json` 隔离，**不会覆盖旧实验结果**。

## 实验范围

| 编号 | 实验 | 目的 | 正式证据边界 |
|---|---|---|---|
| S-E3 | 固定生成轨迹 E3 | playback/generation 共享同一首轮输出，消除原 E3 的生成轨迹混杂 | 受控文本、Mock TTS 时长映射、greedy probe |
| S-A1 | joint crop + role microbenchmark | 直接测联合恢复路径，保存 raw repeats 和 median/IQR | 模型侧 GPU 微基准，不是完整 barge-in |
| S-P1 | headless async control path | 测软件停播确认、timeline lookup、GPU crop/role 联合路径 | 无声卡墙钟播放，不是声学停播或在线 TTS 取消 |

不包含：人工盲评、A2 重跑、完整真实音频闭环、B-syn、E4、中文或多主模型实验。

## 本机验证（不加载模型）

从项目根目录运行：

```bash
uv run python -m py_compile experiments/sci34_supplement/*.py
uv run python -m experiments.sci34_supplement.smoke
uv run python -m src.dialogue.run_timeline_test
```

`smoke` 使用 fake chat/KV runtime 和加速墙钟播放器，不访问 Hugging Face、不下载权重、不要求 CUDA。

## 结果目录

每个正式 run 使用独立 run ID：

```text
experiments/sci34_supplement/results/<experiment>/<run_id>/
├── manifest.json
├── records.jsonl
├── summary.json / analysis.json
└── progress.json
```

Resume 时会比较 config hash 与输入 SHA-256。不一致时拒绝续跑，避免再次混合 fixture、模型或旧 schema。

## 快速入口

- 完整协议：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
- 允许的论文主张：[CLAIMS_MATRIX.md](CLAIMS_MATRIX.md)
- GPU 主机步骤：[GPU_RUNBOOK.md](GPU_RUNBOOK.md)
- 一键编排：[run_all_gpu.sh](run_all_gpu.sh)
