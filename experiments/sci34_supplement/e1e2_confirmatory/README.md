# E1/E2 确认性 campaign

本目录实现并定义 E1/E2 的确认性受控文本段实验。它独立于旧 `experiments/scripts/run_exp1_latency.py`、`run_exp2_tradeoff.py` 与旧结果 JSON；正式输出写入：

```text
experiments/sci34_supplement/results/e1e2_confirmatory/<campaign_id>/
```

旧 `experiments/results/exp1_latency.json`、`exp2_tradeoff.json`、`paper2_reanalysis.json` 永久只读，不覆盖、不续写。

## 当前状态

- 协议、GPU handoff、holdout/TEN builder、campaign manifest、runner、analyzer、validator 和无模型 smoke：**代码已实现**；
- 本地 confirmatory smoke：已可直接运行；
- GPU holdout、真实 TEN cache、pilot、5 个 formal session：**待 GPU 执行**；
- 正式数字：尚无；
- 论文更新：冻结，直到实际主指标、analysis、validation 与 acceptance 全部通过。

## 设计口径

- 新的 100 条 MultiWOZ 2.1 派生 holdout，排除旧 E1/E2 与 accepted 固定轨迹 E3 manifest：
  `experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/manifest.json`；
- 输入为同步预切分文本段，不是实际音频，不含真实 ASR、在线 TTS、播放器或声卡；
- 主模型 Qwen2-7B-Instruct，greedy，batch size 1，`max_new_tokens=32`，`spec_chunk=12`；
- `0.92` 在新 holdout 结果可见前预冻结，不能事后改选；
- TEN confidence 先对所有累积前缀计算一次并存为只读 cache；formal session replay 同一 cache；
- `campaign` 在 TEN cache 后生成不可变 `campaign_manifest.json`，冻结输入/cache/模型/protocol identity；
- formal `run_session` 强制传同一个 `--campaign-manifest`；
- formal 为 5 个独立 Python 进程，`session-index` 固定 `0..4`，每个 100 条 × 10 条件；
- 主分析采用 session→dialogue 两层 bootstrap；
- 浪费率主定义为 `sum(wasted_tokens) / sum(wasted_tokens + final_tokens)`。

必须区分：

- `last_segment_arrival`：最后一段实际到达 harness；
- `first_token_ready`：首个最终可用 token 实际准备完成；
- `endpoint_accept`：同步 oracle 接受决策。

**实际受控墙钟主指标**是 `last_segment_arrival → first_token_ready`。`TTFT_eff` 在候选已准备、oracle 接受时将其记为 0，只是同步 oracle 接受策略的时延的乐观下界（推测收益的上界）；它不是实际墙钟主指标。`endpoint_accept` 不是最后一段到达瞬间，二者不得混淆。

Raw records 已保存 `last_segment_arrival_ns`、`first_token_ready_ns`、`arrival_to_first_token_ready_ns`、`endpoint_accept_ns` 与 `oracle_preaccept_processing_ns`；validator 会复算时间关系，analyzer 将 arrival-to-ready 作为 C-E1/C-E2 主指标，并将 `TTFT_eff` 明确标为 oracle latency lower bound / speculation-benefit upper bound。

## 实际 CLI

从项目根目录运行。以下命令均已与 `--help` 核对：

```bash
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.holdout_builder --help
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.trigger_cache --help
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.campaign --help
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.run_session --help
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.analyze --help
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.validate --help
```

`smoke` 没有 argparse 参数，直接执行：

```bash
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.smoke
```

`campaign` 生成 formal session 强制使用的 `campaign_manifest.json`。五个 formal session 都必须传同一个 `--campaign-manifest`；validator 会检查 session/record 中的共同 manifest SHA-256。

Pilot 使用独立 campaign/session、真实 `transformers` runtime、`--limit 3`、**不传 `--formal`**，也不传 formal campaign manifest。完整可复制命令只维护在 [GPU_HANDOFF.md](GPU_HANDOFF.md)。

## 文档入口

- GPU 唯一执行入口：[GPU_HANDOFF.md](GPU_HANDOFF.md)
- 完整协议与指标定义：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
- 允许与禁止的论文主张：[CLAIMS_MATRIX.md](CLAIMS_MATRIX.md)
- GPU 回传验收模板：[ACCEPTANCE_TEMPLATE.md](ACCEPTANCE_TEMPLATE.md)
