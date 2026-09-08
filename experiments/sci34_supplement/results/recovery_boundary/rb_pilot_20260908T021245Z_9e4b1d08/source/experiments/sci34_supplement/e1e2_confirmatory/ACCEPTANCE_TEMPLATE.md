# E1/E2 确认性 campaign 验收模板

> GPU 回传后填写。所有 `[待填]` 必须由 campaign/session manifest、raw records、analysis、validation、snapshot 或 checksum 直接支持。任一硬性项失败时只能为 `rejected` 或 `pending rerun`；不得引用 `TTFT_eff` 冒充实际主指标。

## 0. 验收结论

- Campaign ID：`[待填]`
- 协议/config identity：`[待填]`
- 代码 commit：`[待填]`
- 结果 commit（未入库填 pending）：`[待填]`
- Analysis：`analysis_v1.json / [其他版本]`
- 状态：`pending / accepted / rejected / pending rerun`
- 日期与验收人：`[待填]`
- 一句话限定性结论：`[待填；不得超出 CLAIMS_MATRIX.md]`

## 1. 代码、工作区与旧结果（硬性）

- [ ] formal 从 clean commit 开始，未使用 `--allow-dirty`。
- [ ] `pyproject.toml`、`uv.lock` 运行前 hash 已记录。
- [ ] 以下旧文件跑前/跑后 SHA-256 一致：
  - `experiments/results/exp1_latency.json`
  - `experiments/results/exp2_tradeoff.json`
  - `experiments/results/paper2_reanalysis.json`
- [ ] 未修改论文稿、chapter、abstract、thesis draft、IEEE 或旧结果。

| 项目 | 跑前 SHA-256 | 跑后 SHA-256 | 一致 |
|---|---|---|---|
| `exp1_latency.json` | `[待填]` | `[待填]` | `[是/否]` |
| `exp2_tradeoff.json` | `[待填]` | `[待填]` | `[是/否]` |
| `paper2_reanalysis.json` | `[待填]` | `[待填]` | `[是/否]` |
| `pyproject.toml` | `[待填]` | — | — |
| `uv.lock` | `[待填]` | — | — |

## 2. 离线环境、模型与数据（硬性）

- [ ] `HF_TOKEN` 为空，日志未泄漏凭据。
- [ ] `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- [ ] 使用 `uv` 冻结环境，未用裸 `python`/`pip` 修改依赖。
- [ ] Qwen2-7B-Instruct、TEN、MultiWOZ 均来自本地资产。
- [ ] accepted E3 排除源固定为：
  `experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/manifest.json`。
- [ ] before/after snapshot 包含 CPU、RAM、OS/kernel、driver、GPU、Python/uv/torch/CUDA/transformers 与 GPU process 清单。

| 身份 | 值 / SHA-256 |
|---|---|
| 主模型路径/config/weights | `[待填]` |
| TEN 路径/config/weights | `[待填]` |
| MultiWOZ path/hash | `[待填]` |
| accepted E3 manifest hash | `[待填]` |
| Python / uv | `[待填]` |
| torch / CUDA / transformers | `[待填]` |
| CPU / RAM / kernel | `[待填]` |
| GPU / driver | `[待填]` |

## 3. CLI、smoke 与 pilot（硬性）

- [ ] `holdout_builder`、`trigger_cache`、`campaign`、`run_session`、`analyze`、`validate` 的 `--help` 均通过。
- [ ] 所有命令均与当前 CLI help 一致。
- [ ] `smoke` 直接执行，未把 `--help` 当测试。
- [ ] py_compile、SCI smoke、timeline smoke、confirmatory smoke、`git diff --check` 通过。
- [ ] pilot 若执行，使用独立非 formal campaign、`--limit 3`，且不传 formal campaign manifest。
- [ ] pilot 未进入 formal records/analysis，也未改变冻结协议。

| 检查 | 日志路径 | 结果 |
|---|---|---|
| CLI help | `[待填]` | `[PASS/FAIL]` |
| py_compile | `[待填]` | `[PASS/FAIL]` |
| SCI smoke | `[待填]` | `[PASS/FAIL]` |
| timeline smoke | `[待填]` | `[PASS/FAIL]` |
| confirmatory smoke | `[待填]` | `[PASS/FAIL]` |
| pilot / skipped | `[待填]` | `[PASS/SKIPPED/FAIL]` |

## 4. Holdout（硬性）

- [ ] 恰好 100 条、ID 唯一、无 fixture-like ID。
- [ ] 每条至少两个非空 segment，且拼接等于 `full_text`。
- [ ] 与旧 E1/E2 交集为 0。
- [ ] 与固定 accepted E3 manifest 的样本交集为 0。
- [ ] seed、筛选、切分、排除源和全部 SHA-256 写入 provenance。

| 项目 | 值 |
|---|---|
| holdout path / SHA-256 | `[待填]` |
| provenance path / SHA-256 | `[待填]` |
| count | `[待填；预期 100]` |
| 与旧 E1/E2 交集 | `[待填；预期 0]` |
| 与 accepted E3 交集 | `[待填；预期 0]` |
| seed | `[待填；预期 20260901]` |

## 5. TEN cache（硬性）

- [ ] 每个累积 prefix 恰好有一条 cache entry。
- [ ] 保存未舍入 confidence、累积文本 hash、template/hash、类别 token/聚合规则。
- [ ] 保存 input SHA、TEN model identity、entry count、identity hash 和文件 SHA。
- [ ] 五个 formal session 使用同一只读 cache。
- [ ] 不把 replay TEN 称为在线零成本。

| 项目 | 值 |
|---|---|
| cache path / SHA-256 | `[待填]` |
| entry count | `[待填]` |
| input SHA-256 | `[待填]` |
| template SHA-256 | `[待填]` |
| TEN identity / cache identity | `[待填]` |

## 6. 不可变 campaign manifest（硬性）

- [ ] `campaign` 在 TEN cache 后、formal session 前生成 manifest。
- [ ] manifest 记录 campaign ID、formal 维度、input、TEN cache/strong identity、主模型 strong identity、protocol、runtime/device 和 content hash；artifact SHA-256 由 CLI 输出并另行归档。
- [ ] manifest 在 clean tree、strict offline 下生成，且未覆盖已有文件。
- [ ] 五个 session manifests 与 5000 条 records 均记录同一个非空 campaign manifest SHA-256。
- [ ] validator 的 campaign manifest hash、campaign identity、runtime metadata 检查全部通过。

| 项目 | 值 |
|---|---|
| manifest path | `[待填]` |
| manifest SHA-256 | `[待填]` |
| manifest content hash | `[待填]` |
| campaign identity hash | `[待填]` |
| main model strong identity | `[待填]` |
| trigger strong identity | `[待填]` |
| input/cache hashes | `[待填]` |
| resolved dtype / attention backend | `[待填]` |

## 7. 五个独立 formal session（硬性）

- [ ] 五个 formal 命令均传同一个 `--campaign-manifest`。
- [ ] 五个 session index 恰好为 `0,1,2,3,4`。
- [ ] 五个不同 `process_start_id`，每个重新加载模型。
- [ ] 每 session 100 × 10 = 1000 records，总计 5000。
- [ ] 无 missing、duplicate 或 truncated JSONL。
- [ ] 每 session 五条 warmup path × 3 repeats；warmup 未混入 records。
- [ ] 条件顺序平衡。
- [ ] E1 直接复用 E2 的 B@0.92 raw records。
- [ ] 未拼接进程重启前后记录。

| Session | Index | PID / process identity | Campaign manifest SHA | Records | Grid | Warmups | 状态 |
|---|---:|---|---|---:|---|---|---|
| `s01` | `0` | `[待填]` | `[待填]` | `[1000]` | `[是/否]` | `[15]` | `[待填]` |
| `s02` | `1` | `[待填]` | `[待填]` | `[1000]` | `[是/否]` | `[15]` | `[待填]` |
| `s03` | `2` | `[待填]` | `[待填]` | `[1000]` | `[是/否]` | `[15]` | `[待填]` |
| `s04` | `3` | `[待填]` | `[待填]` | `[1000]` | `[是/否]` | `[15]` | `[待填]` |
| `s05` | `4` | `[待填]` | `[待填]` | `[1000]` | `[是/否]` | `[15]` | `[待填]` |

## 8. 冻结配置与主时间语义（硬性）

- [ ] 条件为 System A、八阈值和 never-speculate。
- [ ] `0.92` 预冻结且未重选。
- [ ] greedy、batch 1、`max_new_tokens=32`、`spec_chunk=12`。
- [ ] model/prompt/dtype/attention backend identity 一致。
- [ ] 输入明确为非真实音频。
- [ ] 原始时间为单调整数纳秒。

### 8.1 三事件不得混淆

- [ ] `last_segment_arrival` 是最后段到达。
- [ ] `first_token_ready` 是最终首 token 实际准备完成。
- [ ] `endpoint_accept` 是同步 oracle 接受决策，不是最后段到达别名。
- [ ] 实际受控墙钟主指标为：
  `first_token_ready_ns - last_segment_arrival_ns`。
- [ ] raw records 保存并通过 validator 复算 `arrival_to_first_token_ready_ns`。
- [ ] `oracle_preaccept_processing_ns == endpoint_accept_ns - last_segment_arrival_ns`。
- [ ] 没有使用 `endpoint_accept_ns` 伪造 `last_segment_arrival_ns`。

| 字段/指标 | Raw 字段或可复算证据 | 完整性 |
|---|---|---|
| `last_segment_arrival_ns` | `[待填]` | `[PASS/FAIL]` |
| `first_token_ready_ns` | `[待填]` | `[PASS/FAIL]` |
| `arrival_to_first_token_ready_ns` | `[待填公式/输出]` | `[PASS/FAIL]` |
| `endpoint_accept_ns` | `[待填]` | `[PASS/FAIL]` |
| `oracle_preaccept_processing_ns` | `[待填公式/输出]` | `[PASS/FAIL]` |

### 8.2 `TTFT_eff` 仅为时延的乐观下界（推测收益的上界）

- [ ] `TTFT_eff` 明确描述为候选准备后同步 oracle 接受的时延的乐观下界（推测收益的上界）。
- [ ] survived 且 ready>0 时 `TTFT_eff_ns=0`，只表示接受时可立即交付。
- [ ] 没有声称 `TTFT_eff=0` 等于最后段到达后零计算或用户零延迟。
- [ ] consumer delivery 单列，不替代主墙钟指标。

## 9. 浪费率、analysis 与 validation（硬性）

- [ ] 正式 pooled waste 主定义为：
  `sum(wasted_tokens) / sum(wasted_tokens + final_tokens)`。
- [ ] utterance waste 为 `wasted/(wasted+final)`。
- [ ] `speculative_tokens` 只作诊断，不是正式主分母。
- [ ] analyzer 以 `arrival_to_first_token_ready_ns` 为 C-E1/C-E2 主指标。
- [ ] `ttft_eff_ns` 仅在 oracle latency lower-bound / speculation-benefit upper-bound 字段中报告。
- [ ] C-E1 为 A vs B@0.92 同 session/dialogue 配对。
- [ ] C-E2 为 B@0.92 vs never-speculate，并报告九个 B 工作点。
- [ ] session→dialogue 两层 bootstrap，保留全部条件，seed/repeats/estimand/CI 完整。
- [ ] 无 outlier trimming 或未声明排除。
- [ ] validation 通过；analysis 输出未覆盖旧文件。

### C-E1 主墙钟结果（仅完全验收后填写）

| 指标 | System A | B@0.92 | 配对 A−B / CI |
|---|---:|---:|---|
| mean arrival-to-first-token-ready (ms) | `[待填]` | `[待填]` | `[待填]` |
| median (ms) | `[待填]` | `[待填]` | `[待填]` |
| IQR (ms) | `[待填]` | `[待填]` | `[待填]` |
| P95 (ms) | `[待填]` | `[待填]` | `[待填]` |

### `TTFT_eff` 时延的乐观下界（推测收益的上界）诊断

| 指标 | System A | B@0.92 | 说明 |
|---|---:|---:|---|
| mean `TTFT_eff` (ms) | `[待填]` | `[待填]` | `同步 oracle 时延的乐观下界（推测收益的上界）` |
| median `TTFT_eff` (ms) | `[待填]` | `[待填]` | `不得称实际主墙钟` |
| survival / ready tokens | — | `[待填]` | `[待填]` |

### C-E2 主比较

| 指标 | B@0.92 | Never speculate | 配对差 / CI |
|---|---:|---:|---|
| arrival-to-first-token-ready | `[待填]` | `[待填]` | `[待填]` |
| `TTFT_eff` 时延的乐观下界（推测收益的上界） | `[待填]` | `[待填]` | `[待填]` |
| pooled waste `wasted/(wasted+final)` | `[待填]` | `[待填]` | — |
| survival / invalidations | `[待填]` | `[待填]` | — |

- 完整九点表：`[待填]`
- Bootstrap seed / repeats：`[待填]`
- Analysis SHA-256：`[待填]`
- Validation SHA-256：`[待填]`

## 10. Artifact、hash 与回传（硬性）

- [ ] `checksums.sha256` 覆盖 holdout/provenance、TEN cache、campaign manifest、五 sessions、logs、snapshots、analysis、validation、acceptance。
- [ ] tarball 不含模型、凭据、旧结果或其他 campaign。
- [ ] tarball SHA-256 已复核。
- [ ] 若入库，仅 force-add accepted campaign 工件；暂存区无 Python、论文或旧 JSON 意外变化。

| Artifact | 路径 / SHA-256 |
|---|---|
| campaign manifest / SHA-256 | `[待填]` |
| campaign checksums | `[待填]` |
| tarball | `[待填]` |
| tarball SHA-256 | `[待填]` |
| result commit | `[待填]` |

## 11. 异常、失败与 resume 审计

| 时间 | Session/run | 现象 | 影响 formal | 处置 | 新 ID / 证据 |
|---|---|---|---|---|---|
| `[待填]` | `[待填]` | `[待填]` | `[是/否]` | `[待填]` | `[待填]` |

进程重启、OOM、GPU 干扰、hash mismatch 或截断 JSONL 必须记录。重启后默认新 ID 从头跑，不能静默拼接。

## 12. 主张边界签核（硬性）

- [ ] 只称受控同步文本段确认性 E1/E2。
- [ ] `0.92` 只称预冻结 candidate。
- [ ] 实际主指标明确为 `last_segment_arrival→first_token_ready`。
- [ ] `endpoint_accept` 未与最后段到达混淆。
- [ ] `TTFT_eff` 只称同步 oracle 接受时延的乐观下界（推测收益的上界）。
- [ ] 浪费率主定义为 `wasted/(wasted+final)`。
- [ ] 明确输入不是实际音频。
- [ ] 不声称真实 ASR、在线 TEN runtime、在线 TTS、播放器/声卡、声学停播、mouth-to-ear 或生产端到端。

## 13. 最终决定

- 硬性失败项：`[无 / 列出]`
- Campaign manifest identity/hash 完整：`[是/否]`
- 主墙钟字段及恒等式完整：`[是/否]`
- 接受的 formal sessions：`[待填]`
- 排除的 sessions/records：`[待填]`
- 是否允许进入论文权威 Markdown：`[是/否]`
- 是否需要新 run / 代码修正：`[待填]`
- Analysis 版本关系：`[待填]`
- 最终限定性结论：`[待填]`
