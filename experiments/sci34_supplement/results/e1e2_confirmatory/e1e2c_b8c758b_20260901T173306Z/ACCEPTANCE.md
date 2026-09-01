# E1/E2 确认性 campaign 验收模板

> GPU 回传后填写。所有 `[待填]` 必须由 campaign/session manifest、raw records、analysis、validation、snapshot 或 checksum 直接支持。任一硬性项失败时只能为 `rejected` 或 `pending rerun`；不得引用 `TTFT_eff` 冒充实际主指标。

## 0. 验收结论

- Campaign ID：`e1e2c_b8c758b_20260901T173306Z`
- 协议/config identity：config_hash `940ff45e2c8d1ea98558d98b60358eb269ff4d919ef26cdf0f9b4f06b4dd5703`；campaign identity `897b24fb238157c6b108748682e0775afe19361715cc515170fded4e882075a1`；manifest content hash `b307e054f5c699c671c9bd6a59270e15ddd53751a6e32f6b0ce403ebfd9bf146`
- 代码 commit：`b8c758bd8e97e519f041ac047d4f6c5f85697bc7`（paper2，clean tree）
- 结果 commit（未入库填 pending）：与 ACCEPTANCE 同批入库，提交信息「E1E2确认性campaign结果入库」；sha 见 git log
- Analysis：`analysis_v1.json`
- 状态：`accepted`（GPU 机侧全部硬性检查通过；设计侧终审签核待定）
- 日期与验收人：2026-09-02；GPU 运行 agent（实验机 autodl，2×RTX 3090）
- 一句话限定性结论：在受控同步预切分文本段（非真实音频）上，B@0.92 的实际受控墙钟主指标（last_segment_arrival→first_token_ready）相对 System A 无改善（配对 A−B 均值 −34.69 ms，95% CI [−35.30, −34.11]，B 更慢）；`TTFT_eff` 仅作为同步 oracle 时延的乐观下界显示 17.44 ms（95% CI [16.12, 18.75]）的推测收益上界；B@0.92 相对 never-speculate 主墙钟无显著差异（−0.03 ms，CI 含 0），pooled waste 2.85%，survival 67.0%。

## 1. 代码、工作区与旧结果（硬性）

- [x] formal 从 clean commit 开始，未使用 `--allow-dirty`。
- [x] `pyproject.toml`、`uv.lock` 运行前 hash 已记录。
- [x] 以下旧文件跑前/跑后 SHA-256 一致：
  - `experiments/results/exp1_latency.json`
  - `experiments/results/exp2_tradeoff.json`
  - `experiments/results/paper2_reanalysis.json`
- [x] 未修改论文稿、chapter、abstract、thesis draft、IEEE 或旧结果。

| 项目 | 跑前 SHA-256 | 跑后 SHA-256 | 一致 |
|---|---|---|---|
| `exp1_latency.json` | `4481f8148a0509a7ba5fdc5af2a525b719d1e3569bf414cdc7a20faabe8646b9` | 同左 | 是 |
| `exp2_tradeoff.json` | `b018c54696640dca7d6309bed5b08a98092e499f580c4597596c9093730b0cd2` | 同左 | 是 |
| `paper2_reanalysis.json` | `2a1db979ac635fe78fa5d3e35373ee78190f6766849b4fc4f8ade2a3e3920242` | 同左 | 是 |
| `pyproject.toml` | `89c50362feabcc832d69828d6f564c4d26826333eaa76e6be4ff7180f0ecbe4d` | — | — |
| `uv.lock` | `7b76c69de3b04f10d270215206b892cebaf372fe2107a4a76cb6e12436cc2fd1` | — | — |

跑后校验由 `/tmp/e1e2_confirmatory_guard/legacy_{before,after}.sha256` 的 `diff -u` 完成，输出 `LEGACY HASHES UNCHANGED`。

## 2. 离线环境、模型与数据（硬性）

- [x] `HF_TOKEN` 为空，日志未泄漏凭据。
- [x] `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- [x] 使用 `uv` 冻结环境（`uv sync --frozen`），未用裸 `python`/`pip` 修改依赖。
- [x] Qwen2-7B-Instruct、TEN、MultiWOZ 均来自本地资产。
- [x] accepted E3 排除源固定为：
  `experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/manifest.json`。
- [x] before/after snapshot 包含 CPU、RAM、OS/kernel、driver、GPU、Python/uv/torch/CUDA/transformers 与 GPU process 清单。

| 身份 | 值 / SHA-256 |
|---|---|
| 主模型路径/config/weights | `/root/autodl-tmp/dataA/models/Qwen2-7B-Instruct`；strong identity（15 文件，15,242,797,993 bytes）`209f3a9cbccde56fb9ed39fca06a86b11c1aa0ebf721d897baab708d6cab2133`；config.json `8b9a4f6c0acf4854a2d301616d4203039c3d7972cb0a557dae98686b6c3ae4b2` |
| TEN 路径/config/weights | `/root/autodl-tmp/dataA/models/TEN_Turn_Detection`（本轮经 ModelScope 预先下载的本地资产，进入冻结流程后全程离线）；strong identity `c3787bb7b25d9ba37007332be55bb236006eb44d4e97083608bc6cb0f888f722`；config.json `4aaefde58686ea3e34bcc715f6a5aa6f0d051aeeb6be1dd42941d99c33d2e254` |
| MultiWOZ path/hash | `experiments/datasets/raw_data/MultiWOZ/MultiWOZ_2.1/data.json`；`8be37ba1cb5b5a35943f32d4dbe03c5017dd88e15716f74987f60e0ece37851c` |
| accepted E3 manifest hash | `7690f1003109a37c6f216b674ff6df2b71a4bfac98f6992c1eb23b37f98967a4` |
| Python / uv | Python 3.10.18（venv）/ uv 0.12.8 |
| torch / CUDA / transformers | torch 2.8.0+cu128 / CUDA runtime 12.8 / transformers 4.57.1 |
| CPU / RAM / kernel | Intel Xeon Gold 6330 @ 2.00GHz / MemTotal 791,962,704 kB / Linux 5.15.0-136-generic x86_64 |
| GPU / driver | 2 × NVIDIA GeForce RTX 3090 24GB / driver 580.105.08 |

## 3. CLI、smoke 与 pilot（硬性）

- [x] `holdout_builder`、`trigger_cache`、`campaign`、`run_session`、`analyze`、`validate` 的 `--help` 均通过。
- [x] 所有命令均与当前 CLI help 一致。
- [x] `smoke` 直接执行，未把 `--help` 当测试。
- [x] py_compile、SCI smoke、timeline smoke、confirmatory smoke、`git diff --check` 通过。
- [x] pilot 若执行，使用独立非 formal campaign、`--limit 3`，且不传 formal campaign manifest。
- [x] pilot 未进入 formal records/analysis，也未改变冻结协议。

| 检查 | 日志路径 | 结果 |
|---|---|---|
| CLI help | （终端逐项执行，六模块全部输出 usage） | PASS |
| py_compile | `run_logs/smoke.log` | PASS |
| SCI smoke | `run_logs/smoke.log`（`{"status": "PASS"}`） | PASS |
| timeline smoke | `run_logs/smoke.log` | PASS |
| confirmatory smoke | `run_logs/smoke.log`（`{"status": "PASS", "models_loaded": false, "network_used": false}`） | PASS |
| pilot / skipped | `run_logs/pilot01.log`；输出 `e1e2c_b8c758b_20260901T173306Z-pilot/sessions/pilot01`（30 records = 3 对话 × 10 条件，独立非 formal campaign） | PASS |

## 4. Holdout（硬性）

- [x] 恰好 100 条、ID 唯一、无 fixture-like ID。
- [x] 每条至少两个非空 segment，且拼接等于 `full_text`。
- [x] 与旧 E1/E2 交集为 0。
- [x] 与固定 accepted E3 manifest 的样本交集为 0。
- [x] seed、筛选、切分、排除源和全部 SHA-256 写入 provenance。

| 项目 | 值 |
|---|---|
| holdout path / SHA-256 | `inputs/holdout.json`；`e86c0ccb0ae4c56617b983b50940b6c97d672ff9955f0962b8f25f0411d14161` |
| provenance path / SHA-256 | `inputs/holdout.provenance.json`；`103455b7c744377ab443e65cc852672a612eca26512de2485e98142b497d6f67` |
| count | 100 |
| 与旧 E1/E2 交集 | 0（builder 自校验 + provenance 记录） |
| 与 accepted E3 交集 | 0（builder 自校验 + provenance 记录） |
| seed | 20260901 |

## 5. TEN cache（硬性）

- [x] 每个累积 prefix 恰好有一条 cache entry。
- [x] 保存未舍入 confidence、累积文本 hash、template/hash、类别 token/聚合规则。
- [x] 保存 input SHA、TEN model identity、entry count、identity hash 和文件 SHA。
- [x] 五个 formal session 使用同一只读 cache。
- [x] 不把 replay TEN 称为在线零成本。

| 项目 | 值 |
|---|---|
| cache path / SHA-256 | `trigger_cache/trigger_cache.json`；`64008c9277f5c43d88d4bcc710fd3c27c4b635fb61dcfbfbee23c261f86835da` |
| entry count | 222 |
| input SHA-256 | `e86c0ccb0ae4c56617b983b50940b6c97d672ff9955f0962b8f25f0411d14161` |
| template SHA-256 | 见 cache 文件 `trigger.template_sha256`（TEN 自身 chat template，`{text}` user template） |
| TEN identity / cache identity | TEN strong identity `c3787bb7b25d9ba37007332be55bb236006eb44d4e97083608bc6cb0f888f722`；cache identity `e366921baa816a0cd4a64260016321dc0e9d1aeacf11d31da2fc019e50bf6e7a` |

replay TEN 仅为离线查表，不构成在线零成本声明。

## 6. 不可变 campaign manifest（硬性）

- [x] `campaign` 在 TEN cache 后、formal session 前生成 manifest。
- [x] manifest 记录 campaign ID、formal 维度、input、TEN cache/strong identity、主模型 strong identity、protocol、runtime/device 和 content hash；artifact SHA-256 由 CLI 输出并另行归档。
- [x] manifest 在 clean tree、strict offline 下生成，且未覆盖已有文件。
- [x] 五个 session manifests 与 5000 条 records 均记录同一个非空 campaign manifest SHA-256。
- [x] validator 的 campaign manifest hash、campaign identity、runtime metadata 检查全部通过（validation.json `provenance.ok = true`）。

| 项目 | 值 |
|---|---|
| manifest path | `campaign_manifest.json` |
| manifest SHA-256 | `2f4bd76e759945e62a5536b6b4399ad129c47a0b76c967bb653e22ffcf0f4ed8` |
| manifest content hash | `b307e054f5c699c671c9bd6a59270e15ddd53751a6e32f6b0ce403ebfd9bf146` |
| campaign identity hash | `897b24fb238157c6b108748682e0775afe19361715cc515170fded4e882075a1` |
| main model strong identity | `209f3a9cbccde56fb9ed39fca06a86b11c1aa0ebf721d897baab708d6cab2133` |
| trigger strong identity | `c3787bb7b25d9ba37007332be55bb236006eb44d4e97083608bc6cb0f888f722` |
| input/cache hashes | input `e86c0ccb…4161`；cache `64008c92…35da`（cache identity `e366921b…f6e7a`） |
| resolved dtype / attention backend | manifest `runtime_metadata` 如实记录为 `null/null`（capture_stage=session runtime after model load；五 session runtime_metadata 一致性由 validator 通过） |

## 7. 五个独立 formal session（硬性）

- [x] 五个 formal 命令均传同一个 `--campaign-manifest`。
- [x] 五个 session index 恰好为 `0,1,2,3,4`。
- [x] 五个不同 `process_start_id`，每个重新加载模型。
- [x] 每 session 100 × 10 = 1000 records，总计 5000。
- [x] 无 missing、duplicate 或 truncated JSONL。
- [x] 每 session 五条 warmup path × 3 repeats；warmup 未混入 records。
- [x] 条件顺序平衡（validator balance 检查通过，每 session 每条件 10 个 ordinal 槽各 10 次）。
- [x] E1 直接复用 E2 的 B@0.92 raw records（analyzer 按 analysis design 处理）。
- [x] 未拼接进程重启前后记录（无重启，五进程各自完整跑完）。

| Session | Index | PID / process identity | Campaign manifest SHA | Records | Grid | Warmups | 状态 |
|---|---:|---|---|---:|---|---|---|
| `s01` | `0` | `pid-215224-1788285655186877365-b6764d040f484080b35f8e7ea93cef36` | `2f4bd76e…f4ed8` | 1000 | 是 | 15 | 完成 |
| `s02` | `1` | `pid-220915-1788286640061608948-ee60719dfb334eda877138bcb406cf0d` | `2f4bd76e…f4ed8` | 1000 | 是 | 15 | 完成 |
| `s03` | `2` | `pid-226775-1788287654557036881-5aa28763e25c4b2eb8c46e51a96f684e` | `2f4bd76e…f4ed8` | 1000 | 是 | 15 | 完成 |
| `s04` | `3` | `pid-232529-1788288647388977345-e1c3703674b94dbb8835b25caff50cb5` | `2f4bd76e…f4ed8` | 1000 | 是 | 15 | 完成 |
| `s05` | `4` | `pid-238270-1788289640934982861-0868d27c51004b4f90c5a21ab058dc5d` | `2f4bd76e…f4ed8` | 1000 | 是 | 15 | 完成 |

## 8. 冻结配置与主时间语义（硬性）

- [x] 条件为 System A、八阈值和 never-speculate。
- [x] `0.92` 预冻结且未重选。
- [x] greedy、batch 1、`max_new_tokens=32`、`spec_chunk=12`。
- [x] model/prompt/dtype/attention backend identity 一致。
- [x] 输入明确为非真实音频。
- [x] 原始时间为单调整数纳秒。

### 8.1 三事件不得混淆

- [x] `last_segment_arrival` 是最后段到达。
- [x] `first_token_ready` 是最终首 token 实际准备完成。
- [x] `endpoint_accept` 是同步 oracle 接受决策，不是最后段到达别名。
- [x] 实际受控墙钟主指标为：
  `first_token_ready_ns - last_segment_arrival_ns`。
- [x] raw records 保存并通过 validator 复算 `arrival_to_first_token_ready_ns`（validation.json `timing.ok = true`，errors 空）。
- [x] `oracle_preaccept_processing_ns == endpoint_accept_ns - last_segment_arrival_ns`（validator 复算通过）。
- [x] 没有使用 `endpoint_accept_ns` 伪造 `last_segment_arrival_ns`。

| 字段/指标 | Raw 字段或可复算证据 | 完整性 |
|---|---|---|
| `last_segment_arrival_ns` | 5000 条 raw records 逐条保存 | PASS |
| `first_token_ready_ns` | 5000 条 raw records 逐条保存 | PASS |
| `arrival_to_first_token_ready_ns` | raw 字段保存 + validator 复算恒等式 | PASS |
| `endpoint_accept_ns` | 5000 条 raw records 逐条保存 | PASS |
| `oracle_preaccept_processing_ns` | raw 字段 + validator 复算 | PASS |

### 8.2 `TTFT_eff` 仅为时延的乐观下界（推测收益的上界）

- [x] `TTFT_eff` 明确描述为候选准备后同步 oracle 接受的时延的乐观下界（推测收益的上界）。
- [x] survived 且 ready>0 时 `TTFT_eff_ns=0`，只表示接受时可立即交付。
- [x] 没有声称 `TTFT_eff=0` 等于最后段到达后零计算或用户零延迟。
- [x] consumer delivery 单列，不替代主墙钟指标（`consumer_delivery_ms_diagnostic`，median ≈ 0.001 ms）。

## 9. 浪费率、analysis 与 validation（硬性）

- [x] 正式 pooled waste 主定义为：
  `sum(wasted_tokens) / sum(wasted_tokens + final_tokens)`。
- [x] utterance waste 为 `wasted/(wasted+final)`。
- [x] `speculative_tokens` 只作诊断，不是正式主分母。
- [x] analyzer 以 `arrival_to_first_token_ready_ns` 为 C-E1/C-E2 主指标。
- [x] `ttft_eff_ns` 仅在 oracle latency lower-bound / speculation-benefit upper-bound 字段中报告。
- [x] C-E1 为 A vs B@0.92 同 session/dialogue 配对。
- [x] C-E2 为 B@0.92 vs never-speculate，并报告九个 B 工作点。
- [x] session→dialogue 两层 bootstrap，保留全部条件，seed/repeats/estimand/CI 完整。
- [x] 无 outlier trimming 或未声明排除（`excluded_records` = 0）。
- [x] validation 通过（`ok = true`，errors 空）；analysis 输出未覆盖旧文件。

### C-E1 主墙钟结果（仅完全验收后填写）

配对样本 n = 500（5 session × 100 dialogue）。单位 ms。

| 指标 | System A | B@0.92 | 配对 A−B / CI |
|---|---:|---:|---|
| mean arrival-to-first-token-ready (ms) | 27.70 | 62.38 | −34.69（95% CI [−35.30, −34.11]） |
| median (ms) | 27.23 | 62.13 | −34.81 |
| IQR (ms) | 0.70 | 3.84 | 3.87 |
| P95 (ms) | 28.87 | 66.72 | −29.94 |

实际受控墙钟主指标上 B@0.92 慢于 System A；配对 CI 全负、不含 0。

### `TTFT_eff` 时延的乐观下界（推测收益的上界）诊断

| 指标 | System A | B@0.92 | 说明 |
|---|---:|---:|---|
| mean `TTFT_eff` (ms) | 27.70 | 10.26 | `同步 oracle 时延的乐观下界（推测收益的上界）`；配对 A−B +17.44（CI [16.12, 18.75]） |
| median `TTFT_eff` (ms) | 27.23 | 0.00 | `不得称实际主墙钟` |
| survival / ready tokens | — | survival 0.670（CI [0.628, 0.712]）；ready tokens median 12 | B@0.92 条件下 67% 存活 |

### C-E2 主比较

配对样本 n = 500。

| 指标 | B@0.92 | Never speculate | 配对差（never−B）/ CI |
|---|---:|---:|---|
| arrival-to-first-token-ready | 62.38 mean / 62.13 median | 62.35 mean / 62.21 median | −0.03（95% CI [−0.55, +0.51]，含 0，无显著差异） |
| `TTFT_eff` 时延的乐观下界（推测收益的上界） | 10.26 mean / 0.00 median | 31.06 mean / 31.03 median | +20.80（95% CI [19.50, 22.10]） |
| pooled waste `wasted/(wasted+final)` | 0.0285（CI [0.020, 0.037]） | 0.0000 | — |
| survival / invalidations | 0.670 / 见 raw | 0 / — | — |

- 完整九点表（pooled waste / survival / oracle 下界 mean TTFT_eff / 主墙钟 mean，ms）：

| 阈值 | pooled waste | survival | oracle TTFT_eff 下界 mean | arrival→ready mean |
|---|---:|---:|---:|---:|
| 0.0052 | 0.3104 | 1.000 | 0.0 | 62.41 |
| 0.1979 | 0.1931 | 0.990 | 0.3 | 62.40 |
| 0.3906 | 0.1576 | 0.980 | 0.7 | 62.30 |
| 0.5833 | 0.1324 | 0.970 | 0.9 | 62.10 |
| 0.7760 | 0.1127 | 0.960 | 1.3 | 62.20 |
| 0.8500 | 0.1066 | 0.840 | 4.9 | 62.00 |
| 0.9200 | 0.0285 | 0.670 | 10.3 | 62.38 |
| 0.9688 | 0.0000 | 0.280 | 22.4 | 62.30 |
| never  | 0.0000 | 0.000 | 31.1 | 62.35 |

- Bootstrap seed / repeats：20260901 / 10000（两层：先 session、后 session 内 dialogue，保留全部条件）
- Analysis SHA-256：见 `checksums.sha256` 中 `analysis_v1.json` 条目
- Validation SHA-256：见 `checksums.sha256` 中 `validation.json` 条目

## 10. Artifact、hash 与回传（硬性）

- [x] `checksums.sha256` 覆盖 holdout/provenance、TEN cache、campaign manifest、五 sessions、logs、snapshots、analysis、validation、acceptance（72 文件，`sha256sum -c` 全过）。
- [x] tarball 不含模型、凭据、旧结果或其他 campaign（仅本 campaign 目录）。
- [x] tarball SHA-256 已复核。
- [x] 若入库，仅 force-add accepted campaign 工件；暂存区无 Python、论文或旧 JSON 意外变化。

| Artifact | 路径 / SHA-256 |
|---|---|
| campaign manifest / SHA-256 | `campaign_manifest.json`；`2f4bd76e759945e62a5536b6b4399ad129c47a0b76c967bb653e22ffcf0f4ed8` |
| campaign checksums | `checksums.sha256`（72 文件，含自身排除规则，`sha256sum -c` 通过） |
| tarball | `e1e2c_b8c758b_20260901T173306Z.tar.gz` |
| tarball SHA-256 | 见同目录 `e1e2c_b8c758b_20260901T173306Z.tar.gz.sha256`（tarball 含本文件，行内硬编码必然自引用失配，以伴同校验文件为准） |
| result commit | 与本 ACCEPTANCE 同批提交（「E1E2确认性campaign结果入库」），见 git log |

## 11. 异常、失败与 resume 审计

| 时间 | Session/run | 现象 | 影响 formal | 处置 | 新 ID / 证据 |
|---|---|---|---|---|---|
| 2026-09-01 17:33 前 | 环境准备 | `TEN_Turn_Detection` 本机缺失 | 否 | 进入冻结流程前一次性经 ModelScope 下载至 `/root/autodl-tmp/dataA/models/TEN_Turn_Detection`（15GB，17 文件）；此后全程离线，正式 run 零联网 | ModelScope snapshot；§2 快照 |
| — | s01–s05 | 无 OOM、无进程重启、无 GPU 干扰、无 duplicate/truncation/hash mismatch | 否 | 无需 resume | validation.json errors 空 |

## 12. 主张边界签核（硬性）

- [x] 只称受控同步文本段确认性 E1/E2。
- [x] `0.92` 只称预冻结 candidate。
- [x] 实际主指标明确为 `last_segment_arrival→first_token_ready`。
- [x] `endpoint_accept` 未与最后段到达混淆。
- [x] `TTFT_eff` 只称同步 oracle 接受时延的乐观下界（推测收益的上界）。
- [x] 浪费率主定义为 `wasted/(wasted+final)`。
- [x] 明确输入不是实际音频。
- [x] 不声称真实 ASR、在线 TEN runtime、在线 TTS、播放器/声卡、声学停播、mouth-to-ear 或生产端到端。

## 13. 最终决定

- 硬性失败项：无
- Campaign manifest identity/hash 完整：是
- 主墙钟字段及恒等式完整：是（validator 复算通过）
- 接受的 formal sessions：s01、s02、s03、s04、s05（各 1000 records）
- 排除的 sessions/records：无（`excluded_records` = 0）
- 是否允许进入论文权威 Markdown：是（GPU 机侧检查全过；设计侧终审后按 CLAIMS_MATRIX.md 表述）
- 是否需要新 run / 代码修正：否（无需重跑；`runtime_metadata.resolved_dtype/attention_backend` 为 null 系采集实现如实记录，validator 通过）
- Analysis 版本关系：`analysis_v1.json` 为首个版本，无旧版本被覆盖
- 最终限定性结论：在受控同步预切分文本段（非真实音频）confirmatory 协议下：C-E1 实际墙钟主指标 B@0.92 显著慢于 System A（+34.69 ms，CI [34.11, 35.30]）；`TTFT_eff` 口径的推测收益上界为 17.44 ms（CI [16.12, 18.75]）；C-E2 中 B@0.92 与 never-speculate 主墙钟无显著差异（CI 含 0），oracle 下界改善 20.80 ms（CI [19.50, 22.10]），代价为 pooled waste 2.85%、survival 67.0%；九阈值工作点的 waste/survival 单调、可直接用于 trade-off 曲线。
