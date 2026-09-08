# E1/E2 确认性 campaign

本目录实现并定义 E1/E2 的确认性受控文本段实验。它独立于旧 `experiments/scripts/run_exp1_latency.py`、`run_exp2_tradeoff.py` 与旧结果 JSON；正式输出写入：

```text
experiments/sci34_supplement/results/e1e2_confirmatory/<campaign_id>/
```

旧 `experiments/results/exp1_latency.json`、`exp2_tradeoff.json`、`paper2_reanalysis.json` 永久只读，不覆盖、不续写。

## 当前状态

> 以下为 2026-09-03 的 post-run 状态。最初的 pre-run 状态可由 Git 历史恢复；冻结设计与预注册口径仍保留在 `EXPERIMENT_PLAN.md` §1–§13，不作追溯改写。

- formal campaign `e1e2c_b8c758b_20260901T173306Z` 已于代码 commit `b8c758bd8e97e519f041ac047d4f6c5f85697bc7` 完成，并在结果 commit `62508dc79a8843e5dbe58677750f2c22010a1e44` 入库；
- 5 个独立 formal session 共 5000 条 records，`validation.json` 为 `ok=true`，`analysis_v1.json`、`ACCEPTANCE.md` 与 72 文件历史 `checksums.sha256` 已归档，设计侧由 D-017 接受；
- 无条件 GPU 重跑已关闭；`GPU_HANDOFF.md` 仅供独立复现，禁止覆盖 accepted 目录；
- accepted 主结果与边界见 [`paper2/e1e2_confirmatory_acceptance_2026-09-02.md`](../../../paper2/e1e2_confirmatory_acceptance_2026-09-02.md)，全仓 campaign 矩阵见 [`REPRODUCIBILITY.md`](../../../REPRODUCIBILITY.md)。

### v2 交叉/乘积 bootstrap 复分析状态（2026-09-03，post-run）

Accepted campaign 已新增只读派生工件 `analysis_v2.json` 与 `analysis_v2.sha256`；未修改 raw、`validation.json`、`analysis_v1.json`、`ACCEPTANCE.md` 或历史 `checksums.sha256`。当前 analyzer/result 尚未提交，因此其 Git code/result commit 为 `not recorded`，待本批改动由作者提交后补足。

- analyzer：`analyze_v2.py`，`schema_version=2`，`analysis_version=crossed-product-bootstrap-v2`；
- 正式设计：完整 5 session × 100 global dialogue × 10 condition 网格；固定 `random.Random(20260901)`、10,000 repeats、percentile 95%；每次独立有放回抽 5 个 session 与 100 个全局 dialogue，以笛卡尔积权重 `m_s*n_d` 保留条件配对；
- candidate selection/compute readiness（raw alias `arrival_to_first_token_ready_ns`）：E1 A−B@0.92 = **−34.687728 ms**，95% CI **[−35.442098, −33.953509]**；E2 never−B@0.92 = **−0.033492 ms**，95% CI **[−0.638608, 0.614945]**；该事件不是 generator/production deliverability；
- `TTFT_eff` synchronous-oracle lower bound：E1 A−B@0.92 = **17.436697 ms**，95% CI **[14.407946, 20.323448]**；E2 never−B@0.92 = **20.803658 ms**，95% CI **[17.849195, 23.645048]**；
- B@0.92 pooled waste = **0.028527**，95% CI **[0.011239, 0.047345]**；survival = **0.670**，95% CI **[0.580, 0.760]**；
- output identity：A/B@0.92 full output **280/500**、first token **465/500**、44/100 unique dialogues 有任一 mismatch；B@0.92/never full output 与 first token 均 **500/500**。这些诊断不筛选任何主时延记录；
- v2 point estimates 与 v1 全部精确兼容（最大绝对差 0）；`analysis_v2.json` SHA-256 = `9bce6db5d93c1faccb4069b295df32ce5ee0778899b31ac6be17526bfb644456`；
- Windows checkout 的 CRLF 本地 hash 与 formal LF identity 同时记录，正式 provenance 绑定 normalized-LF SHA-256。

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
uv run python -m experiments.sci34_supplement.e1e2_confirmatory.analyze_v2 --help
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
