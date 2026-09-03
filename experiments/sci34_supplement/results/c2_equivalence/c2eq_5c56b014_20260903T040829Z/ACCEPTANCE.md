# C2 equivalence campaign 验收模板（协议 v2）

> GPU 工件返回后填写。任一硬门槛失败只能填写 `rejected` 或 `pending rerun`；失败工件不得删除。封存前必须把状态明确改为 accepted，且保留下面精确文本 `Status: accepted`。
> **填写纪律（v1 教训）**：所有通过率/计数一律以 `validation.json` 与 `records.jsonl` 的独立重算为准，不得凭印象填写（v1 ACCEPTANCE 曾把 top-1 写成 45/45，实际运行时口径 43/45、严格重算 41/45）。

## 0. 结论

- Run ID：`c2eq_5c56b014_20260903T040829Z`
- Code commit：`5c56b0144c822e4a05ba4eeec167684363e8828e`
- Protocol version：`2`
- Campaign identity：`165c91f98c734a2cdbaa63ce534047c6bfb40db034ca87c5ee5c8b9b7f8176dd`
- Manifest SHA-256 / content hash：`7a636016c7959566f47303fb47c3da7c68775bda08b12abc6df59ed192ef178d` / `b7d8a4747a348f14f016d4dff8f3322dedd4677989df537fda415d65094ddefe`
- Reviewer / UTC date：GPU 实验机执行者（ZCode）/ 2026-09-03
- Status: `rejected`
- 限定性结论：在冻结 Qwen2-7B-Instruct（content identity `209f3a9c…`）、BF16、sdpa、RTX 3090 的协议 v2 下：token/状态/顶层层全部等价成立——24/24 termination probe 合格（natural_eos 6 genuine / 4 requalified，≥5 门槛达成）、canonical↔path token IDs 45/45 exact、KV/mask/seq/ledger 状态与 unique EOT 45/45、scenario execution 24/24、v2 continuation 近并列 margin 规则 45/45（30 完全 exact）、top-1 运行时口径 43/45（2 个近并列翻转均在 margin 限内且落在 top-5）、top-5 overlap min 4/5。但 v2 噪声相对门槛 **42/45 通过**：3 个 checkpoint 边际超阈（path/control 比值 2.38/2.36/2.63 × 对照臂噪声，超过冻结的 2.0× 倍数），绝对安全上限（max≤2.0、mean≤0.5）全过。campaign 级 verdict 为 FAIL，未 seal。

## 1. 范围与网格（硬性）

- [x] 恰好 1 个逻辑 session `s01`；case resume 可有多个完整记录的 process identity。（单进程 `pid-48291-1788408567427447242-e34758f9a04c41fbabada03876f4799a`，24 attempts，无中断无 resume）
- [x] 恰好 24 个 deterministic cases；formal 不接受删减网格。（24/24 落盘）
- [x] 无统计重复、无 bootstrap、无 outlier/case 删除。
- [x] 512/2048/8192 context 和全部 8 scenario、3 termination 均覆盖。
- [x] `cases.json` 与 manifest input SHA-256 一致。（`acda9afb83ff393d222fc06eb433c1a307dba1eb213fcbbbedfdb860f7500696`）
- [x] Pilot 使用不同 run ID/目录，未进入 formal raw/analysis。（本 commit 的 pilot `c2pilot_5c56b014_20260903T040650Z`；前一 commit `a501df43` 的 pilot `c2pilot_a501df43_20260903T033106Z` 因探针实现缺陷被拦截，已在 commit `899462c` 归档，与本轮 formal 隔离）

## 2. Clean / offline / 模型身份（硬性）

- [x] campaign manifest 从 exact clean commit 创建。（dirty=false）
- [x] formal 使用显式本地 Qwen2-7B-Instruct `--model` 路径。
- [x] `HF_TOKEN` 与 `HUGGING_FACE_HUB_TOKEN` 为空。
- [x] `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- [x] `uv sync --frozen`；`uv.lock` hash 已记录。（env_before/after 一致）
- [x] strong model identity 已对本地 snapshot 全文件内容寻址。
- [x] dtype 为 BF16；attention backend、tokenizer class 已冻结。（`torch.bfloat16` / `sdpa` / `Qwen2TokenizerFast`）
- [x] tokenizer `eos_token_id == <|im_end|>`，chat template hash 与 special-token IDs 已记录。

| 项目 | 值 |
|---|---|
| 本地模型路径 | `/root/autodl-tmp/dataA/models/Qwen2-7B-Instruct` |
| model content identity | `209f3a9cbccde56fb9ed39fca06a86b11c1aa0ebf721d897baab708d6cab2133`（accepted artifact hash `fae2ece1…` 与 D-017 一致） |
| tokenizer/chat template hash | `Qwen2TokenizerFast` / `793202280c0910ab26bc6eb57c8212c2417324c6d8c4efd0d10d59092ab3e3eb` |
| EOS / EOT / PAD IDs | 151645 / 151645 / 151643 |
| dtype / attention backend | `torch.bfloat16` / `sdpa` |
| Python / torch / CUDA / transformers | 3.10.18 / 2.8.0+cu128 / CUDA runtime 12.8 / 4.57.1 |
| GPU / driver / CPU / RAM / kernel | NVIDIA GeForce RTX 3090（cuda:0）/ 580.105.08 / Intel Xeon Gold 6330 / 755 GiB / 5.15.0-136-generic |

## 3. 旧结果 hash guard 与 v1 归档保护（硬性）

- [x] `GPU_HANDOFF.md` 列出的 legacy paths（含 v1 C2 归档 run 与 `e3_exact_rescue/`）跑前/跑后 SHA-256 完全一致。（289 个 git 跟踪结果文件，diff 为空）
- [x] 未修改旧 E1/E2/E3/A1/P1 工件、v1 C2 rejected 工件、其他文档或论文。
- [x] E3 `p2_turns.json` 抢救件已在 v1 轮入库且 hash 仍为 `a2116b83...9248a0c`（本轮不重做）。

| Guard / rescue | 跑前 | 跑后 / 实际 | 一致/状态 |
|---|---|---|---|
| legacy aggregate hash manifest（含 v1 C2 归档） | `/tmp/c2_equivalence_guard_v2b/legacy_before.sha256`（289 文件） | `legacy_after.sha256` diff 为空 | PASS |
| E3 `e3_exact_rescue/`（v1 已入库） | — | `a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c` | READ-ONLY PASS |

## 4. CLI、smoke 与 pilot（硬性）

- [x] `campaign/run/validate/analyze/seal --help` 与 GPU_HANDOFF 命令一致。
- [x] `py_compile` 通过。
- [x] 纯 CPU fake `smoke.py` 通过且未加载模型/联网。（`protocol_version=2`、`models_loaded=false`、`network_used=false`、24 cases PASS、`checkpoint_sidecars=45`）
- [x] core KV/timeline/speculative/supplement smoke 通过。
- [x] integration pilot 为独立 non-formal run，仅作兼容性/成本预检。（`--limit 3`，3/3 通过，含 genuine natural_eos c2_01）
- [x] Pilot 未被用于授予或补足 formal termination 资格，也未用于调整 frozen tolerance、case、token 数或 protocol。（注：本 commit 前的 a501df4 pilot 暴露的探针分支缺陷为代码修复输入，非 protocol/常数变更——修复 diff 仅 runtime.py 一行 `else:`→`elif case.termination == "max_tokens":` 与 smoke 覆盖，由设计侧 commit `5c56b01` 提供）
- [x] formal 的 24 条 record 均各自重新执行并通过 termination probe；未引用 pilot probe 结果。（24/24 qualified，validator 独立复核）

## 5. Raw termination 与 retained-token correctness（逐 record/checkpoint 硬性，v2 门槛）

- [x] `termination_probes.required == observed == qualified == 24`，且 `natural_eos.genuine >= 5`；每个 checkpoint 的 probe 与所属 record 完全一致。（genuine 6 / requalified 4，gate passed）
- [x] `natural_eos` 全部为真实模型 greedy（cap 256）：genuine 者在上限内观测 EOS 且 phase 为 `ASSISTANT_EOT_PENDING`（steps 21/8/20/111/98/80）；未命中者 `requalified=true`、`MAX_TOKENS`、内容恰为 cap、phase 为 `ASSISTANT_OPEN`，且仍完成全部等价 checkpoint。（4 个 requalified 全自洽）
- [x] `eos_at_cap` 全部明确 `controlled=true`，cap=4 且 EOT 精确位于最后一步；fixture 末 token 为 EOT，内容 token 仍走 production KV append。（6/6，eos_step 全为 4）
- [x] `max_tokens` 全部为真实模型 greedy，预算 2 内未 EOS，显式 `MAX_TOKENS` 且 phase 为 `ASSISTANT_OPEN`。（8/8，content 全为 2）
- [x] 所有 EOS probe 的 pending EOT 均未进入 KV、完整 token ledger 或 assistant 内容 ledger；内容 IDs/hash/count 与长度关系可独立复算。（0 违例）
- [x] `crop_pending_eot` 在 teacher-force 内容后真实调用一次受控 EOT 的 `generate_accumulating(max_new_tokens=1)`，`pending_before_crop=true`、EOT 不进 ledger，截断 crop 后 pending 清除，再 reopen。（scenario_execution 24/24 passed）
- [x] `reply_tail_noop` 同样真实进入 EOT pending，`crop_to_token(current_seq)` 后 `no_op_preserved_pending=true`，再 reopen；record/checkpoint 的 scenario execution 完全一致。
- [x] canonical 与 crop path token IDs 100% exact，token hash 可独立复算。（45/45 `token_ids_exact=true`）
- [x] first mismatch 对所有 checkpoint 为 null。
- [x] `seq == mask == KV == token ledger`。（state errors 全空）
- [x] assistant 内容 spans 与内容账本 exact；结构 EOT 未混入内容账本。
- [x] role phase/end reason 合法。
- [x] 每个 assistant→user boundary 恰好一个 EOT，EOT 位置 exact。（`unique_eot.ok=true` 45/45）
- [x] next-token top-1：exact，或 canonical margin ≤ 近并列限（`top1_flip_near_tie=true`）；翻转 token 必须在 top-5 集合内。（运行时口径 43/45 exact；2 个翻转 checkpoint（`c2_16/post_recovery`、`c2_24/next_assistant`）均 `top1_flip_near_tie=true`、`top1_ok=true`）
- [x] top-5 overlap 全部 `>=4/5`，完整分布已报告。（min 4、mean 4.8667/5）
- [ ] v2 相对门槛：每个 checkpoint `path max_abs <= 2.0×max(control max_abs, 0.05)` 且 `mean_abs <= 2.0×max(control mean_abs, 0.01)`；绝对安全上限 `max_abs<=2.0`、`mean_abs<=0.5`；control 统计与 `checkpoints/*.npz` 三数组由 validator 独立重算一致。（**42/45 通过**；3 个失败见下表与 §6；绝对安全上限 45/45 通过；validator 重算一致）
- [x] 32-token greedy continuation：exact，或首个发散步 canonical margin ≤ 近并列限；每步 top1/top2/margin 已记录且与发出的 token 一致；所有 `continuation_source=actual_crop_cache`，clean side 为 `clean_prefill_cache`，checkpoint state/logits 在 continuation mutation 前捕获。（exact 30/45；15 个发散点 margin 全部 ≤ 各自 checkpoint 近并列限（观测 0–0.25，限 0.125–0.5），`continuation_ok` 45/45）
- [x] `full_rollback_p0` 保留 assistant header、提交 empty assistant EOT，assistant boundary=1；`speculation_full_invalidation` 删除完整 transition、保持原 user open，assistant boundary=0；两类 token 序列不混同且 next-user 未被人为加换行。
- [x] next-turn continuation 与后续第二 crop checkpoint 适用同样的 v2 门槛。（按同一门槛判定）

| 指标 | 结果 | worst case/checkpoint |
|---|---:|---|
| Cases / checkpoints | 24 / 45 | — |
| Termination probes qualified | 24 / 24 | 无失败 |
| Natural EOS genuine/requalified | 6 / 4（≥5/10 达标） | genuine steps 21/8/20/111/98/80 |
| Controlled EOS-at-cap positions | 6/6 全为 4/4 | 无 |
| MAX_TOKENS observed/budget | 8/8 全为 2 | 无 |
| Pending EOT in KV/full/content ledger | 0 / 0 / 0 | 无 |
| Token/state exact rate | 45/45 = 100% | 无失败 |
| Top-1 exact（运行时口径）/ near-tie flips | 43/45 / 2（均在 margin 限内） | `c2_16_second_medium_eos/post_recovery`、`c2_24_second_long_max/next_assistant` |
| Top-5 overlap min/mean | 4 / 4.8667 | 门槛 ≥4 全过 |
| Path max_abs / control max_abs（worst） | 0.96875 / 0.40625 | `c2_06_invalidate_short_max/next_assistant`（限 0.8125，**超**） |
| path/control 比值 worst（0.5B dry-run 参考 1.08） | 2.632 | `c2_21_pending_long_max/post_recovery`（mean 0.08432/0.03204） |
| 噪声相对门槛通过率 | 42/45 | 失败：`c2_06/next_assistant`（max 比值 2.38）、`c2_10/post_recovery`（mean 比值 2.36）、`c2_21/post_recovery`（mean 比值 2.63） |
| Continuation exact rate（描述性） | 30/45 | 首个发散位 0–25 |
| 发散点 margin 全部 ≤ 近并列限 | 是（15/15，margin 0–0.25 ≤ 限 0.125–0.5） | 无违例 |
| Unique EOT failures | 0 | 无 |

## 6. 失败、attempt 与 resume 审计

| UTC time | Case | Attempt / process identity | 现象 | 工件 | 处置 |
|---|---|---|---|---|---|
| 2026-09-03T03:31Z | a501df4 pilot（c2_01） | pilot 独立进程 | v2 探针 `else` 分支误伤 genuine natural_eos（实现缺陷） | `c2pilot_a501df43_20260903T033106Z/`（commit `899462c`） | 拦截 formal、上报；设计侧修复入 `5c56b01` |
| 2026-09-03T04:08–04:14Z | c2_06_invalidate_short_max | `pid-48291-…`（单进程，24 attempts 无 resume） | `next_assistant`：path max_abs 0.96875 > 限 0.8125（比值 2.38×） | records + `checkpoints/c2_06…next_assistant.npz` | 保留；判定 rejected |
| 同上 | c2_10_clean_medium_eos | 同上 | `post_recovery`：path mean_abs 0.09014 > 限 0.07626（比值 2.36×） | records + `checkpoints/c2_10…post_recovery.npz` | 同上 |
| 同上 | c2_21_pending_long_max | 同上 | `post_recovery`：path mean_abs 0.08432 > 限 0.06407（比值 2.63×） | records + `checkpoints/c2_21…post_recovery.npz` | 同上 |

- [x] 所有异常 attempt 均在 `attempts.jsonl`，无半条 records。（24 attempts，1 process identity，无 resume）
- [x] Resume 只跳过已有完整 case，manifest/cases/model/code identity 未变化。（未发生中断）
- [x] 每 checkpoint 的 `checkpoints/*.npz`（path/canonical/control 三数组）全量保留，共 45 个。
- [x] 若存在任一 formal failure，本表状态不是 accepted，且未 seal。（状态 `rejected`；`seal --create` 以文档方式试跑一次并确认 fail-closed，未写任何状态）

## 7. Validate → analyze → acceptance → seal → tar（硬性顺序）

- [x] `summary.json` 存在，formal validator 已从 records 独立核对其 case/checkpoint/failed/process/identity/probe 计数和 verdict。（`status=FAIL`、`acceptance_eligible=false`，10 errors，复核一致）
- [x] `validation.json` 首先生成且 `ok=true`、`acceptance_eligible=true`。（**实际 `ok=false`、`acceptance_eligible=false`**——fail closed 如设计；validator 从 `checkpoints/*.npz` 三数组独立重算与 stored 一致）
- [ ] `analysis_v1.json` 之后生成；不含 bootstrap；按 context/scenario/termination/checkpoint 汇总。（**analyzer fail-closed 拒绝**，文件不存在，`logs/analysis_v1.log` 保留退出栈）
- [ ] analysis 列出 worst cases 与全部失败索引。（未生成，同上；失败索引见 §6）
- [x] 本验收由 raw records 独立复核后填写。（全部计数由 `records.jsonl`/`checkpoints/*.npz` 字段重算，非凭印象）
- [ ] 状态变为 accepted 后才运行 `seal --create`。（未 accepted，**未 seal**；`checksums_return.sha256` 为回传完整性清单，非封存印章）
- [ ] `seal --verify` 通过。（未封存）
- [x] 最后创建 tarball 与 `.tar.gz.sha256`，回传完整目录而非 summary 子集。

| Artifact | 路径 / SHA-256 |
|---|---|
| validation | `validation.json`（`ok=false`；文件 sha 见 `checksums_return.sha256`） |
| analysis_v1 | 未生成（analyzer fail-closed 拒绝） |
| ACCEPTANCE | `ACCEPTANCE.md`（本文件） |
| checksums.sha256 | 未生成（未 seal）；`checksums_return.sha256` 提供回传完整性 |
| tarball | `experiments/sci34_supplement/results/c2_equivalence/c2eq_5c56b014_20260903T040829Z.tar.gz` |
| tarball SHA-256 | 见同目录 `.tar.gz.sha256`（伴同校验文件为准） |

## 8. 最终签核

- 硬失败项：3/45 checkpoint 未过 v2 噪声相对门槛（2.0× 倍数）——`c2_06_invalidate_short_max/next_assistant`（max_abs 0.96875，比值 2.38）、`c2_10_clean_medium_eos/post_recovery`（mean_abs 0.09014，比值 2.36）、`c2_21_pending_long_max/post_recovery`（mean_abs 0.08432，比值 2.63）。其余所有门槛（probe 24/24、token/state/unique-EOT/top1 规则/top5/continuation 规则/绝对安全上限/scenario）全部通过。
- Formal failed cases/checkpoints：3 cases / 3 checkpoints（共 24 cases / 45 checkpoints）。
- 接受的 run：无（本 run `rejected`；pilots 不具 formal 资格）。
- 是否允许用于论文正确性主张：否——未达 campaign 级全部通过；设计侧另行决策与授权。
- 是否需要新 run：需设计侧决策。本轮失败集中于 mean_abs/max_abs 的 2.0× 相对倍数边际（比值 2.36–2.63；0.5B dry-run 参考 1.08，本机 7B@8192 上下文量级更高），绝对安全上限与全部结构层门槛均通过；若设计侧评估该倍率需调整，须以新协议版本冻结（红线 3 禁止事后改本 run 常数）。
- 最终限定性结论：仅限冻结 Qwen2-7B-Instruct snapshot、BF16、sdpa、本机环境、协议 v2：crop/recovery 路径与同 retained IDs 的 canonical clean re-prefill 在 token 序列、KV/mask/seq/ledger、内容账本、unique EOT、role phase 与 v2 margin 感知的 next-token/continuation 规则上全部等价；3/45 checkpoint 的 logit 数值差超过 2.0× 对照臂噪声冻结倍数（不超过绝对安全上限）。不外推其他模型/backend/dtype/硬件。

封存所需精确状态行（仅全部通过后取消注释并保留一行）：

<!-- Status: accepted -->
