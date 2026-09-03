# C2 equivalence campaign 验收模板（协议 v2）

> GPU 工件返回后填写。任一硬门槛失败只能填写 `rejected` 或 `pending rerun`；失败工件不得删除。封存前必须把状态明确改为 accepted，且保留下面精确文本 `Status: accepted`。
> **填写纪律（v1 教训）**：所有通过率/计数一律以 `validation.json` 与 `records.jsonl` 的独立重算为准，不得凭印象填写（v1 ACCEPTANCE 曾把 top-1 写成 45/45，实际运行时口径 43/45、严格重算 41/45）。

## 0. 结论

- Run ID：`[待填]`
- Code commit：`[待填]`
- Protocol version：`2`
- Campaign identity：`[待填]`
- Manifest SHA-256 / content hash：`[待填]`
- Reviewer / UTC date：`[待填]`
- Status: `[pending / accepted / rejected / pending rerun]`
- 限定性结论：`[待填；仅限冻结 Qwen2-7B BF16 correctness protocol v2]`

## 1. 范围与网格（硬性）

- [ ] 恰好 1 个逻辑 session `s01`；case resume 可有多个完整记录的 process identity。
- [ ] 恰好 24 个 deterministic cases；formal 不接受删减网格。
- [ ] 无统计重复、无 bootstrap、无 outlier/case 删除。
- [ ] 512/2048/8192 context 和全部 8 scenario、3 termination 均覆盖。
- [ ] `cases.json` 与 manifest input SHA-256 一致。
- [ ] Pilot 使用不同 run ID/目录，未进入 formal raw/analysis。

## 2. Clean / offline / 模型身份（硬性）

- [ ] campaign manifest 从 exact clean commit 创建。
- [ ] formal 使用显式本地 Qwen2-7B-Instruct `--model` 路径。
- [ ] `HF_TOKEN` 与 `HUGGING_FACE_HUB_TOKEN` 为空。
- [ ] `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- [ ] `uv sync --frozen`；`uv.lock` hash 已记录。
- [ ] strong model identity 已对本地 snapshot 全文件内容寻址。
- [ ] dtype 为 BF16；attention backend、tokenizer class 已冻结。
- [ ] tokenizer `eos_token_id == <|im_end|>`，chat template hash 与 special-token IDs 已记录。

| 项目 | 值 |
|---|---|
| 本地模型路径 | `[待填]` |
| model content identity | `[待填]` |
| tokenizer/chat template hash | `[待填]` |
| EOS / EOT / PAD IDs | `[待填]` |
| dtype / attention backend | `[待填]` |
| Python / torch / CUDA / transformers | `[待填]` |
| GPU / driver / CPU / RAM / kernel | `[待填]` |

## 3. 旧结果 hash guard 与 v1 归档保护（硬性）

- [ ] `GPU_HANDOFF.md` 列出的 legacy paths（含 v1 C2 归档 run 与 `e3_exact_rescue/`）跑前/跑后 SHA-256 完全一致。
- [ ] 未修改旧 E1/E2/E3/A1/P1 工件、v1 C2 rejected 工件、其他文档或论文。
- [ ] E3 `p2_turns.json` 抢救件已在 v1 轮入库且 hash 仍为 `a2116b83...9248a0c`（本轮不重做）。

| Guard / rescue | 跑前 | 跑后 / 实际 | 一致/状态 |
|---|---|---|---|
| legacy aggregate hash manifest（含 v1 C2 归档） | `[待填]` | `[待填]` | `[PASS/FAIL]` |
| E3 `e3_exact_rescue/`（v1 已入库） | — | `[hash]` | `[READ-ONLY PASS]` |

## 4. CLI、smoke 与 pilot（硬性）

- [ ] `campaign/run/validate/analyze/seal --help` 与 GPU_HANDOFF 命令一致。
- [ ] `py_compile` 通过。
- [ ] 纯 CPU fake `smoke.py` 通过且未加载模型/联网。
- [ ] core KV/timeline/speculative/supplement smoke 通过。
- [ ] integration pilot 为独立 non-formal run，仅作兼容性/成本预检。
- [ ] Pilot 未被用于授予或补足 formal termination 资格，也未用于调整 frozen tolerance、case、token 数或 protocol。
- [ ] formal 的 24 条 record 均各自重新执行并通过 termination probe；未引用 pilot probe 结果。

## 5. Raw termination 与 retained-token correctness（逐 record/checkpoint 硬性，v2 门槛）

- [ ] `termination_probes.required == observed == qualified == 24`，且 `natural_eos.genuine >= 5`；每个 checkpoint 的 probe 与所属 record 完全一致。
- [ ] `natural_eos` 全部为真实模型 greedy（cap 256）：genuine 者在上限内观测 EOS 且 phase 为 `ASSISTANT_EOT_PENDING`；未命中者 `requalified=true`、`MAX_TOKENS`、内容恰为 cap、phase 为 `ASSISTANT_OPEN`，且仍完成全部等价 checkpoint。
- [ ] `eos_at_cap` 全部明确 `controlled=true`，cap=4 且 EOT 精确位于最后一步；fixture 末 token 为 EOT，内容 token 仍走 production KV append。
- [ ] `max_tokens` 全部为真实模型 greedy，预算 2 内未 EOS，显式 `MAX_TOKENS` 且 phase 为 `ASSISTANT_OPEN`。
- [ ] 所有 EOS probe 的 pending EOT 均未进入 KV、完整 token ledger 或 assistant 内容 ledger；内容 IDs/hash/count 与长度关系可独立复算。
- [ ] `crop_pending_eot` 在 teacher-force 内容后真实调用一次受控 EOT 的 `generate_accumulating(max_new_tokens=1)`，`pending_before_crop=true`、EOT 不进 ledger，截断 crop 后 pending 清除，再 reopen。
- [ ] `reply_tail_noop` 同样真实进入 EOT pending，`crop_to_token(current_seq)` 后 `no_op_preserved_pending=true`，再 reopen；record/checkpoint 的 scenario execution 完全一致。
- [ ] canonical 与 crop path token IDs 100% exact，token hash 可独立复算。
- [ ] first mismatch 对所有 checkpoint 为 null。
- [ ] `seq == mask == KV == token ledger`。
- [ ] assistant 内容 spans 与内容账本 exact；结构 EOT 未混入内容账本。
- [ ] role phase/end reason 合法。
- [ ] 每个 assistant→user boundary 恰好一个 EOT，EOT 位置 exact。
- [ ] next-token top-1：exact，或 canonical margin ≤ 近并列限（`top1_flip_near_tie=true`）；翻转 token 必须在 top-5 集合内。
- [ ] top-5 overlap 全部 `>=4/5`，完整分布已报告。
- [ ] v2 相对门槛：每个 checkpoint `path max_abs <= 2.0×max(control max_abs, 0.05)` 且 `mean_abs <= 2.0×max(control mean_abs, 0.01)`；绝对安全上限 `max_abs<=2.0`、`mean_abs<=0.5`；control 统计与 `checkpoints/*.npz` 三数组由 validator 独立重算一致。
- [ ] 32-token greedy continuation：exact，或首个发散步 canonical margin ≤ 近并列限；每步 top1/top2/margin 已记录且与发出的 token 一致；所有 `continuation_source=actual_crop_cache`，clean side 为 `clean_prefill_cache`，checkpoint state/logits 在 continuation mutation 前捕获。
- [ ] `full_rollback_p0` 保留 assistant header、提交 empty assistant EOT，assistant boundary=1；`speculation_full_invalidation` 删除完整 transition、保持原 user open，assistant boundary=0；两类 token 序列不混同且 next-user 未被人为加换行。
- [ ] next-turn continuation 与后续第二 crop checkpoint 适用同样的 v2 门槛。

| 指标 | 结果 | worst case/checkpoint |
|---|---:|---|
| Cases / checkpoints | `[24 / 待填]` | — |
| Termination probes qualified | `[24 / 24]` | `[待填]` |
| Natural EOS genuine/requalified | `[待填；须 >=5/10 genuine]` | `[待填]` |
| Controlled EOS-at-cap positions | `[待填；预期全为 4/4]` | `[待填]` |
| MAX_TOKENS observed/budget | `[待填；预期全为 2]` | `[待填]` |
| Pending EOT in KV/full/content ledger | `[待填；预期均 0]` | `[待填]` |
| Token/state exact rate | `[待填；预期 45/45]` | `[待填]` |
| Top-1 exact（运行时口径）/ near-tie flips | `[待填]` | `[待填]` |
| Top-5 overlap min/mean | `[待填]` | `[待填]` |
| Path max_abs / control max_abs（worst） | `[待填]` | `[待填]` |
| path/control 比值 worst（0.5B dry-run 参考 1.08） | `[待填]` | `[待填]` |
| 噪声相对门槛通过率 | `[待填；预期 45/45]` | `[待填]` |
| Continuation exact rate（描述性） | `[待填]` | `[待填]` |
| 发散点 margin 全部 ≤ 近并列限 | `[待填]` | `[待填]` |
| Unique EOT failures | `[待填；预期 0]` | `[待填]` |

## 6. 失败、attempt 与 resume 审计

| UTC time | Case | Attempt / process identity | 现象 | 工件 | 处置 |
|---|---|---|---|---|---|
| `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` |

- [ ] 所有异常 attempt 均在 `attempts.jsonl`，无半条 records。
- [ ] Resume 只跳过已有完整 case，manifest/cases/model/code identity 未变化。
- [ ] 每 checkpoint 的 `checkpoints/*.npz`（path/canonical/control 三数组）全量保留，共 45 个。
- [ ] 若存在任一 formal failure，本表状态不是 accepted，且未 seal。

## 7. Validate → analyze → acceptance → seal → tar（硬性顺序）

- [ ] `summary.json` 存在，formal validator 已从 records 独立核对其 case/checkpoint/failed/process/identity/probe 计数和 verdict。
- [ ] `validation.json` 首先生成且 `ok=true`、`acceptance_eligible=true`。
- [ ] `analysis_v1.json` 之后生成；不含 bootstrap；按 context/scenario/termination/checkpoint 汇总。
- [ ] analysis 列出 worst cases 与全部失败索引。
- [ ] 本验收由 raw records 独立复核后填写。
- [ ] 状态变为 accepted 后才运行 `seal --create`；seal 已确认全套 raw/log/snapshot 工件存在并直接 formal 重算 validation/analysis，stored 核心 verdict/provenance 与重算一致。
- [ ] `seal --verify` 通过；相对路径按字典序，checksums 不自包含。
- [ ] 最后创建 tarball 与 `.tar.gz.sha256`，回传完整目录而非 summary 子集。

| Artifact | 路径 / SHA-256 |
|---|---|
| validation | `[待填]` |
| analysis_v1 | `[待填]` |
| ACCEPTANCE | `[待填]` |
| checksums.sha256 | `[待填]` |
| tarball | `[待填]` |
| tarball SHA-256 | `[待填]` |

## 8. 最终签核

- 硬失败项：`[无 / 列出]`
- Formal failed cases/checkpoints：`[待填]`
- 接受的 run：`[待填]`
- 是否允许用于论文正确性主张：`[是/否；后续另行授权]`
- 是否需要新 run：`[待填]`
- 最终限定性结论：`[待填]`

封存所需精确状态行（仅全部通过后取消注释并保留一行）：

<!-- Status: accepted -->
