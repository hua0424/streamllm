# SCI 3–4 区补实验目录

本目录保存第二篇论文的新增补实验代码、协议和 GPU 执行说明。它与原始 `experiments/scripts/` 及 `experiments/results/*.json` 隔离，**不会覆盖旧实验结果**。

## 实验范围

| 编号 | 实验 | 目的 | 正式证据边界 |
|---|---|---|---|
| S-E3 | 固定生成轨迹 E3 | playback/generation 共享同一首轮输出，消除原 E3 的生成轨迹混杂 | 受控文本、Mock TTS 时长映射、greedy probe |
| S-A1 | joint crop + role microbenchmark | 直接测联合恢复路径，保存 raw repeats 和 median/IQR | 模型侧 GPU 微基准，不是完整 barge-in |
| S-P1 | headless async control path | 已接受：180 条 prepared-state v2 事件，测软件停播确认、timeline lookup、GPU crop/role 联合路径 | 无声卡墙钟播放，不是声学停播或在线 TTS 取消 |
| C-E1/E2 | E1/E2 确认性 campaign | 新 holdout、0.92 预冻结、greedy、TEN cache、5 个独立进程；实际主指标为最后段到达→首 token 准备，`TTFT_eff` 仅为同步 oracle 时延的乐观下界（推测收益的上界），浪费率为 wasted/(wasted+final) | 受控同步文本段；endpoint accept 不等于最后段到达；不是实际音频或生产端到端 |

不包含：人工盲评、A2 重跑、完整真实音频闭环、B-syn、E4、中文或多主模型实验。C-E1/E2 虽确认 E1/E2 的受控模型侧结果，但明确不补齐上述真实音频闭环。

## 本机验证（不加载模型）

从项目根目录运行：

```bash
uv run python -m py_compile experiments/sci34_supplement/*.py
uv run python -m experiments.sci34_supplement.smoke
uv run python -m src.dialogue.run_timeline_test
```

`smoke` 使用 fake chat/KV runtime 和加速墙钟播放器，不访问 Hugging Face、不下载权重、不要求 CUDA。

## 结果目录

每个正式 run 使用独立 run ID。prepared-state P1 v2 的新 run、日志和打包件默认被 `results/.gitignore` 忽略，避免正式 run 或 resume 污染 clean-tree 检查；设计方验收后如需入库，应显式 `git add -f`：

```text
experiments/sci34_supplement/results/<experiment>/<run_id>/
├── manifest.json
├── records.jsonl
├── summary.json / analysis.json
└── progress.json
```

Resume 时会比较 config hash 与输入 SHA-256。不一致时拒绝续跑，避免再次混合 fixture、模型或旧 schema。

已接受的 P1 v2 结果位于 `results/async_bargein/sci34_dc52978_20260901_async_prepared_v2/`。GPU 实验员按约定用 `git add -f` 将默认忽略的正式结果、日志和环境快照纳入 commit `ee1dcc7`；该操作只增加版本化审计副本，不改变实验数据或协议。

C-E1/E2 formal `e1e2c_b8c758b_20260901T173306Z` 已完成并由 D-017 接受；正式输出位于 `results/e1e2_confirmatory/<campaign_id>/`，旧 GPU handoff 只作复现记录。二审后的当前 GPU 待办是独立 C2 correctness campaign：修复 EOS/EOT 角色状态，并比较 crop/recovery 与 canonical token-ID clean re-prefill；其输出使用 `results/c2_equivalence/<run_id>/`，不覆盖任何既有工件。

## 快速入口

- 当前 C2 正确性 campaign：[c2_equivalence/README.md](c2_equivalence/README.md)（GPU 唯一入口为其 `GPU_HANDOFF.md`）
- 已接受 E1/E2 确认性 campaign：[e1e2_confirmatory/README.md](e1e2_confirmatory/README.md)（历史复现入口）
- 既有 S-E3/S-A1/S-P1 GPU 交接：[HANDOFF_FOR_GPU.md](HANDOFF_FOR_GPU.md)
- 既有补实验完整协议：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
- 允许的论文主张：[CLAIMS_MATRIX.md](CLAIMS_MATRIX.md)
- GPU 主机步骤：[GPU_RUNBOOK.md](GPU_RUNBOOK.md)
- P1 prepared-state 定向重跑：[P1_PREPARED_RERUN.md](P1_PREPARED_RERUN.md)
- 一键编排：[run_all_gpu.sh](run_all_gpu.sh)
