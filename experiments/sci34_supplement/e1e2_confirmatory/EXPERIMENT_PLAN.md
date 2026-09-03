# E1/E2 确认性 campaign 实验计划

## 1. 目的与证据边界

本 campaign 在新的未见 MultiWOZ 派生 holdout 上确认两个受控模型侧问题：

- **C-E1**：System A（完整输入后一次性 prefill）与预冻结 System B@`0.92` 的首 token 准备延迟差异；
- **C-E2**：冻结阈值网格下，首 token 准备延迟与推测浪费的离散工作点，并以 B@`0.92` 对 `never_speculate` 作预先指定比较。

输入是 harness 同步交付的预切分文本段，不是实际音频。实验不运行真实 ASR、实际 endpoint detector、在线 TTS、播放器或声卡，不测声学停播、mouth-to-ear 或生产端到端延迟。

旧 `experiments/results/exp1_latency.json`、`exp2_tradeoff.json` 和 `paper2_reanalysis.json` 永久只读。本 campaign 使用独立输入与结果目录。

## 2. 关键设计修正：墙钟主指标、oracle 与 `TTFT_eff`

必须保留三个不同事件：

1. **`last_segment_arrival`**：最后一个受控文本段实际进入模型侧 harness 的单调墙钟时刻；
2. **`first_token_ready`**：第一个属于最终被接受响应、可供后续链路使用的 token 实际准备完成的单调墙钟时刻；
3. **`endpoint_accept`**：同步 oracle 根据最终输入决定接受当前候选/路径的时刻。

实际受控墙钟主指标定义为：

```text
arrival_to_first_token_ready_ns = first_token_ready_ns - last_segment_arrival_ns
```

它回答“最后一段到达后，到最终首 token 实际准备好用了多久”。`endpoint_accept` 不是最后一段到达的别名：最后一段还可能需要增量 prefill、候选完成或同步，oracle 才能作接受决策。文档、字段命名和结果解释都不得把二者混为一谈。

`TTFT_eff` 保留为次要策略量：若候选在 oracle 接受前已经准备完成并存活，则 oracle 接受后可立即交付，`TTFT_eff=0`；否则为接受后到首 token 可交付的时间。因此 `TTFT_eff` 是**候选已准备后同步 oracle 接受的时延的乐观下界（推测收益的上界）**，不等于实际 `last_segment_arrival→first_token_ready` 墙钟延迟，也不代表零计算、零调度或零用户感知延迟。

当前 raw record schema 已正式保存 `last_segment_arrival_ns`、`first_token_ready_ns`、`arrival_to_first_token_ready_ns`、`endpoint_accept_ns` 与 `oracle_preaccept_processing_ns`。Runner/validator 强制复算时间差，analyzer 将 `arrival_to_first_token_ready_ns` 作为 C-E1/C-E2 主指标，并把 `ttft_eff_ns` 作为 oracle latency lower-bound / speculation-benefit upper-bound 次要策略量。

## 3. 研究问题与预指定比较

### 3.1 C-E1

配对单位为 `(session_id, dialogue_id)`：

- System A：最后输入完整后进行一次性 full prefill 与 greedy decode；
- System B@`0.92`：允许早期 prefix 推测，最终输入到达后由同步 oracle 接受存活候选或走现场生成；
- 首选主比较应使用 `arrival_to_first_token_ready_ns`；
- `TTFT_eff` 仅作接受时延的乐观下界（推测收益的上界）诊断；
- C-E1 的 B@`0.92` 必须直接复用 C-E2 同一 raw record，禁止另跑。

### 3.2 C-E2

报告八个阈值和 `never_speculate` 的全部离散工作点。主比较为 B@`0.92` 对 `never_speculate`，不得在新 holdout 上重选主阈值或把离散点宣传为连续、普适或部署最优前沿。

## 4. 冻结条件与配置

每个 formal session 对 100 条输入执行 10 个条件：

1. `system_a_full_prefill`
2. `b_threshold_0.0052`
3. `b_threshold_0.1979`
4. `b_threshold_0.3906`
5. `b_threshold_0.5833`
6. `b_threshold_0.7760`
7. `b_threshold_0.8500`
8. `b_threshold_0.9200`
9. `b_threshold_0.9688`
10. `b_never_speculate`

统一配置：

- 本地 Qwen2-7B-Instruct；
- greedy，`do_sample=false`；
- batch size 1；
- `max_new_tokens=32`；
- `spec_chunk=12`；
- 固定 system prompt、tokenizer、dtype、attention backend；
- 条件网格和 confirmatory threshold 由 `protocol.py` 冻结，不通过 CLI 动态指定；
- 每个 session 的 `session-index` 必须为 `0..4`。

## 5. 新的 disjoint holdout

从本地 MultiWOZ 2.1 原始文件确定性派生 100 条话语：

- 固定 seed `20260901`；
- 每条保存 `id`、`full_text`、`segments`；
- ID 唯一，每条至少两个非空 segment，且 `"".join(segments) == full_text`；
- 显式排除旧 E1、旧 E2 与 accepted 固定轨迹 E3；
- accepted E3 排除源固定为：
  `experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/manifest.json`；
- formal 默认拒绝 fixture-like 路径、`fx*` ID、数量不足、缺失排除文件或任何联网回退；
- 保存 source、exclusions、holdout 与 provenance SHA-256。

实际 CLI 为 `holdout_builder --input --output --provenance [--exclude ...] [--count] [--seed] [--non-formal]`。formal 是默认值，没有 `--formal` 参数。

## 6. TEN trigger cache

在主推理前对每个累积 segment prefix 运行一次真实 TEN，输出单个只读 JSON cache。保存：

- ID、prefix index、累积文本 SHA-256；
- 未舍入 confidence；
- template/hash、正负类别、token IDs 与聚合规则；
- TEN model identity；
- holdout SHA-256、entry count、cache identity/hash。

五个 formal session replay 同一个 cache；TEN runtime 不进入本 campaign 的首 token 时间窗，不支持“在线 trigger 零开销”主张。

实际 CLI 为 `trigger_cache --input --output --model [--device]`。没有 `--dtype`、`--formal` 或输出目录参数。

## 7. 不可变 campaign manifest

TEN cache 完成后，使用 `campaign` CLI 生成 `$CAMPAIGN_DIR/campaign_manifest.json`。它在 formal 默认模式下要求 clean tree 与严格离线，冻结 campaign ID、100 条 input、TEN cache/strong identity、主模型 strong identity、runtime/device 和 `ProtocolConfig`，manifest 内保存 content hash，CLI 输出并由流程另行记录 artifact SHA-256。输出存在时拒绝覆盖。

实际 CLI 为：

```text
campaign --campaign-id --input --trigger-cache --main-model [--device] --output [--non-formal]
```

Formal `run_session` 强制要求 `--campaign-manifest`，并核对 manifest payload 与当前 input/cache/model/runtime/device identity 完全相同。五个 session、session manifests 和所有 records 必须共享同一个 campaign manifest SHA-256。

## 8. Pilot 与五个独立 formal 进程

### 8.1 Pilot

Pilot 使用独立 campaign/session、`--runtime transformers`、`--limit 3` 并省略 `--formal`。Non-formal runner 禁止传 formal campaign manifest，因此 pilot 无需也不得传 formal manifest；它不进入 formal campaign grid 或汇总。

### 8.2 Formal

- 5 个独立 Python 进程：例如 `s01..s05`；
- `session-index` 分别为 `0,1,2,3,4`；
- 每个进程重新加载主模型；
- 每 session 100 × 10 = 1000 records，总计 5000；
- `warmup_repeats=3`，五条 path 共 15 条 warmup JSONL，formal records 不含 warmup；
- condition order 按 session/dialogue 的冻结循环算法平衡；
- record 保存 PID、`process_start_id`、session/dialogue/condition/order 与单调时间；
- 进程重启后不得拼接同一 session。当前 `--resume` 仅允许仍保持原 `process_start_id` 的同一进程；重启会被拒绝。

`run_session` 实际参数以 `--help` 为准，包括 `--campaign-id`、`--session-id`、`--session-index`、`--input`、`--trigger-cache`、`--campaign-manifest`、`--results-root`、`--runtime`、`--model`、`--device`、seed/protocol 参数、pilot-only `--limit`、`--formal`、`--allow-dirty`、`--resume`。Formal 模式强制 `--campaign-manifest`，禁止 `--limit` 与 `--allow-dirty`。

## 9. 原始字段与浪费率

每条 record 至少保存当前实现字段：

- `last_segment_arrival_ns`、`first_token_ready_ns`、`arrival_to_first_token_ready_ns`；
- `endpoint_accept_ns`、`oracle_preaccept_processing_ns`、`first_deliverable_token_ns`；
- `consumer_delivery_ns`、`consumer_delivery_from_arrival_ns`、`generation_done_ns`；
- `ttft_eff_ns`；
- candidate start/first/lead（适用时）；
- `survived`、`ready_tokens`、on-demand TTFT；
- speculation/invalidation/wasted/final/speculative token counts；
- EOS、max-token、输出 token IDs/text、trigger prefix/hash。

Runner/validator 要求 `arrival_to_first_token_ready_ns == first_token_ready_ns - last_segment_arrival_ns`，并检查 `oracle_preaccept_processing_ns == endpoint_accept_ns - last_segment_arrival_ns`；这些 raw 字段是正式主指标与 oracle 诊断的直接证据。

浪费率主定义为 pooled：

```text
sum(wasted_tokens) / sum(wasted_tokens + final_tokens)
```

utterance-level waste 为 `wasted / (wasted + final)` 后再描述其分布。`speculative_tokens` 只作诊断，不作为主浪费率分母。

## 10. 结果布局

当前实际布局：

```text
experiments/sci34_supplement/results/e1e2_confirmatory/<campaign_id>/
├── inputs/
│   ├── holdout.json
│   └── holdout.provenance.json
├── trigger_cache/
│   └── trigger_cache.json
├── campaign_manifest.json
├── sessions/s01..s05/
│   ├── manifest.json
│   ├── records.jsonl
│   ├── warmups.jsonl
│   ├── progress.json
│   └── summary.json
├── run_logs/
├── snapshots/before|after/
├── validation.json
├── analysis_v1.json
├── checksums.sha256
└── ACCEPTANCE.md
```

`campaign_manifest.json`、`validation.json` 和 `analysis_v1.json` 都不可覆盖；identity 或分析口径变化必须使用新的 campaign/versioned 路径。

## 11. 分析与不确定性

正式分析：

- C-E1：A vs B@0.92 同 session/dialogue 配对；
- C-E2：B@0.92 vs never-speculate 主比较，并报告九个 B 工作点；
- analyzer 以 `arrival_to_first_token_ready_ns` 为实际主指标，并将 `TTFT_eff` 汇总明确标为 oracle latency lower bound / speculation-benefit upper bound；
- `TTFT_eff`、consumer delivery、survival、ready tokens、invalidations、candidate lead、on-demand TTFT 与 EOS/max-token 均作为诊断；
- pooled waste 使用 `wasted/(wasted+final)`；
- 不做 outlier trimming。

两层 bootstrap：

1. 有放回重采样 5 个 session；
2. 在每个抽中 session 内有放回重采样 dialogue；
3. 保留 dialogue 的全部条件；
4. 固定 seed `20260901`、10,000 repeats、percentile 95% CI。

实际 analyzer CLI：`analyze --campaign-dir [--out] [--bootstrap-repeats] [--bootstrap-seed] [--expected-sessions] [--expected-dialogues] [--non-formal]`。

## 12. 验收门禁

必须全部满足才能标记 accepted：

- clean commit、严格离线、HF token 为空；
- holdout 恰好 100 条，与旧 E1/E2 和固定 accepted E3 disjoint；
- TEN cache 覆盖所有 prefix，五 session 使用同一 hash；
- formal `campaign_manifest.json` 的 content hash、artifact SHA 和 input/cache/model/protocol identity 完整；
- 五个 session manifests/records 共享同一 campaign manifest SHA-256；
- 5 个独立 process identity，每 session 1000 records，总计 5000；
- 十条件完整、condition-order 平衡；
- B@0.92 在 E1/E2 中复用同一 raw records；
- greedy、32 token、12-token spec chunk、batch 1 和 prompt/model identity 一致；
- 时间顺序与 token accounting 合法；
- `TTFT_eff=0` 只按同步 oracle 接受语义解释；
- raw records 中 `last_segment_arrival_ns`、`first_token_ready_ns` 和 `arrival_to_first_token_ready_ns` 完整且可直接复算；
- pooled waste 可按 `wasted/(wasted+final)` 复算；
- validation、analysis、bootstrap provenance、snapshots、checksums、tarball 齐全；
- 旧结果跑前/跑后 hash 一致；
- 未修改论文稿和旧结果。

实际 validator CLI：`validate --campaign-dir [--out] [--expected-sessions] [--expected-dialogues] [--non-formal]`。

## 13. 代码与 GPU 状态

- `holdout_builder`、`trigger_cache`、`campaign`、`run_session`、`analyze`、`validate`、`smoke`：代码已实现；
- 所有 argparse CLI 已通过 `--help` 核对；
- `smoke` 无 argparse，直接运行；
- GPU formal campaign：待执行；
- 论文数字：仍冻结。

完整、可复制的正式命令只维护在 [GPU_HANDOFF.md](GPU_HANDOFF.md)。

## 14. v2 post-run：交叉/乘积 bootstrap 复分析（2026-09-03）

§1–§13 原样保留为 pre-run protocol 与 v1 历史。accepted source campaign `e1e2c_b8c758b_20260901T173306Z` 现已完成 versioned v2 离线复分析；原始 records、`validation.json`、`analysis_v1.json`、`ACCEPTANCE.md` 与 `checksums.sha256` 均未修改。

### 14.1 冻结方法

`analyze_v2.py` 使用 `schema_version=2`、`analysis_version=crossed-product-bootstrap-v2`。正式参数固定为 5 sessions、100 global dialogues、10,000 repeats、seed `20260901`。每个 replicate 在排序后的 ID 上用同一 `random.Random(20260901)` 流：先有放回抽 5 个 session，再有放回抽 100 个全局 dialogue；原始 `(session, dialogue)` cell 的权重为两个边际 multiplicity 的乘积 `m_s*n_d`，其 10 个条件共享该权重，因此保持 paired estimand。区间为 percentile 95%，沿用 `analyze.py` 的线性分位数插值。点估计始终使用未加权完整 5×100 网格。

该方法 supersede v1 的“抽 session 后在各抽中 session 内嵌套抽 dialogue”区间估计，但不更改 v1 点估计。B 条件 waste 每次 replicate 计算 weighted numerator / weighted denominator 的 ratio-of-sums，不平均 utterance ratios；survival 使用同一 product weights。

### 14.2 正式结果与诊断

- candidate selection/compute readiness（raw alias `arrival_to_first_token_ready_ns`，非 generator/production deliverability）：E1 System A−B@0.92 **−34.687728 ms**，95% CI **[−35.442098, −33.953509]**；E2 never−B@0.92 **−0.033492 ms**，95% CI **[−0.638608, 0.614945]**；
- `TTFT_eff` synchronous-oracle latency lower bound：E1 A−B@0.92 **17.436697 ms**，95% CI **[14.407946, 20.323448]**；E2 never−B@0.92 **20.803658 ms**，95% CI **[17.849195, 23.645048]**；
- B@0.92 pooled waste **0.028527**，95% CI **[0.011239, 0.047345]**；survival **0.670**，95% CI **[0.580, 0.760]**；
- A/B@0.92：full output exact **280/500**，first token exact **465/500**，44/100 unique dialogues 有任一 mismatch；各 session 均为 44/100，比较签名及两侧输出均 100/100 dialogues 跨 session 不变；
- B@0.92/never：full output、first token、长度、EOS、max-token、文本全部 **500/500 exact/agreement**；
- output identity 仅作 implementation-path 诊断，不过滤主时延；consumer/yield marker 仅称 harness diagnostic，不称 production deliverability；
- 四个配对效应均提供 per-session 与 leave-one-session-out sensitivity；所有条件均补充 arrival→candidate readiness、arrival→endpoint、arrival/endpoint→first-deliverable event、arrival/endpoint→consumer marker 的派生事件汇总；
- v2 point estimates 与 v1 对应值最大绝对差为 0；正式 source provenance 绑定 repo-relative path + normalized-LF SHA-256，并另存 Windows CRLF local hash；
- 输出：campaign root `analysis_v2.json`、`analysis_v2.sha256`；analysis SHA-256 为 `9bce6db5d93c1faccb4069b295df32ce5ee0778899b31ac6be17526bfb644456`。

验证已完成：`py_compile`、`analyze_v2 --self-test`、formal generation 与不依赖 analyzer 的独立 10,000-repeat sanity extraction 全部通过。self-test 覆盖 crossed≠nested、配对、product weight sum、ratio-of-sums、空输出 identity、reproducibility 以及 missing/duplicate/malformed/timing fail-closed。
