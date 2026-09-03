# C2 equivalence campaign 验收模板

> GPU 工件返回后填写。任一硬门槛失败只能填写 `rejected` 或 `pending rerun`；失败工件不得删除。封存前必须把状态明确改为 accepted，且保留下面精确文本 `Status: accepted`。

## 0. 结论

- Run ID：`c2eq_563dd22a_20260903T013547Z`
- Code commit：`563dd22a55544e042826290f9dde736fa7fef458`
- Campaign identity：`1f07a2e91bd97e6c3ff5f497d0f017a27321c712c1d27fd2285b82861eba8a36`
- Manifest SHA-256 / content hash：`f4960a20364de8c4e78bdb51c3accdd02576b5ce61e2375c36bc5831054c9670` / `e0dc511acbd5e77a34affe0fddae5cb1ce2f46dd1762921e575120750192e054`
- Reviewer / UTC date：GPU 实验机执行者（ZCode）/ 2026-09-03
- Status: `rejected`
- 限定性结论：在冻结 Qwen2-7B-Instruct（content identity `209f3a9c…`）、BF16、sdpa、torch 2.8.0+cu128、RTX 3090 的 24-case correctness protocol 下：**token/state 层等价 100% 成立**（canonical 与 crop path token IDs、KV/mask/seq/ledger、assistant 内容账本、unique EOT、role phase、next-token top-1、top-5 overlap 全部 exact），但**数值 logit 等价的冻结门槛未通过**——全部 45 个 checkpoint 的 FP32 比较 max_abs 0.15625–0.96875 > 0.1、mean_abs 0.0202–0.1565 > 0.01（RMS 0.0268–0.1643），15/45 checkpoint 的 32-token greedy continuation 发散（首个发散位 0–25）；另有 4/10 `natural_eos` probe 在冻结 128-token 上限内未观测 EOS（greedy 持续生成），termination 资格 20/24。失败工件全部保留，未 seal。

## 1. 范围与网格（硬性）

- [x] 恰好 1 个逻辑 session `s01`；case resume 可有多个完整记录的 process identity。（实际：单进程一次完成，`process_start_id=pid-10378-…`，24 attempts 全属同一进程，无 resume）
- [x] 恰好 24 个 deterministic cases；formal 不接受删减网格。（24/24 跑完并落盘）
- [x] 无统计重复、无 bootstrap、无 outlier/case 删除。
- [x] 512/2048/8192 context 和全部 8 scenario、3 termination 均覆盖。
- [x] `cases.json` 与 manifest input SHA-256 一致。（`acda9afb83ff393d222fc06eb433c1a307dba1eb213fcbbbedfdb860f7500696`）
- [x] Pilot 使用不同 run ID/目录，未进入 formal raw/analysis。（`c2pilot_563dd22a_20260903T013205Z`，独立目录）

## 2. Clean / offline / 模型身份（硬性）

- [x] campaign manifest 从 exact clean commit 创建。（dirty=false；`git status --porcelain` 为空）
- [x] formal 使用显式本地 Qwen2-7B-Instruct `--model` 路径。
- [x] `HF_TOKEN` 与 `HUGGING_FACE_HUB_TOKEN` 为空。
- [x] `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
- [x] `uv sync --frozen`；`uv.lock` hash 已记录。（guard env_before/after 一致）
- [x] strong model identity 已对本地 snapshot 全文件内容寻址。（15 文件逐文件 sha256）
- [x] dtype 为 BF16；attention backend、tokenizer class 已冻结。（`torch.bfloat16` / `sdpa` / `Qwen2TokenizerFast`）
- [x] tokenizer `eos_token_id == <|im_end|>`，chat template hash 与 special-token IDs 已记录。

| 项目 | 值 |
|---|---|
| 本地模型路径 | `/root/autodl-tmp/dataA/models/Qwen2-7B-Instruct` |
| model content identity | `209f3a9cbccde56fb9ed39fca06a86b11c1aa0ebf721d897baab708d6cab2133`（accepted artifact hash `fae2ece10b76512237cf28957f98e7b0d2c609455a173031e3bd16b3dff7c5ab`，与 D-017 一致） |
| tokenizer/chat template hash | `Qwen2TokenizerFast` / `793202280c0910ab26bc6eb57c8212c2417324c6d8c4efd0d10d59092ab3e3eb` |
| EOS / EOT / PAD IDs | 151645 / 151645 / 151643 |
| dtype / attention backend | `torch.bfloat16` / `sdpa` |
| Python / torch / CUDA / transformers | 3.10.18 / 2.8.0+cu128 / CUDA runtime 12.8 / 4.57.1 |
| GPU / driver / CPU / RAM / kernel | NVIDIA GeForce RTX 3090（cuda:0）/ 580.105.08 / Intel Xeon Gold 6330 / 755 GiB / 5.15.0-136-generic |

补充说明（环境资产，不影响 protocol）：本机无 `Qwen/Qwen2.5-0.5B-Instruct` HF 缓存，§2 的 `run_kvcrop_test`/`run_speculative_test` smoke 需要 0.5B 模型；已在该 smoke 阶段（formal 之前）经 HF 下载入 `/root/autodl-tmp/hfhome`（HF_HUB_DISABLE_XET=1），并以符号链接对齐 `cache_dir=$HF_HOME` 的查找布局。formal 期间严格离线，无任何联网。

## 3. 旧结果 hash guard 与 E3 数据抢救（硬性）

- [x] `GPU_HANDOFF.md` 列出的 legacy paths 跑前/跑后 SHA-256 完全一致。（186 个 git 跟踪旧结果文件，diff 为空）
- [x] 未修改旧 E1/E2/E3/A1/P1 工件、其他文档或论文。
- [x] 从 E3 manifest 的原始路径抢救 exact `p2_turns.json`。
- [x] 保存 `p2_turns.json` SHA-256，并核对为 manifest 中 `a2116b83...9248a0c`。（一致）
- [x] 保存 raw MultiWOZ path/hash、builder/provenance、模型 snapshot 身份；若缺失明确记录 missing，不伪造。

| Guard / rescue | 跑前 | 跑后 / 实际 | 一致/状态 |
|---|---|---|---|
| legacy aggregate hash manifest | `/tmp/c2_equivalence_guard/legacy_before.sha256`（186 文件） | `legacy_after.sha256` diff 为空 | PASS |
| E3 `p2_turns.json` | — | `/tmp/e3_exact_rescue/p2_turns.json`，SHA `a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c` | RECOVERED |
| raw MultiWOZ | — | `experiments/datasets/raw_data/MultiWOZ/MultiWOZ_2.1/data.json`（hash 见 `provenance.sha256`） | RECOVERED |
| builder/provenance | — | `experiments/scripts/prepare_multiwoz_data.py`（hash 见 `provenance.sha256`）；未发现 `*p2_turns*provenance*.json` 文件 | RECOVERED（builder only） |
| E3 model snapshot | — | `/root/autodl-tmp/dataA/models/Qwen2-7B-Instruct`，rehash `209f3a9c…`（与 E3 manifest 记录一致） | RECOVERED |

## 4. CLI、smoke 与 pilot（硬性）

- [x] `campaign/run/validate/analyze/seal --help` 与 GPU_HANDOFF 命令一致。
- [x] `py_compile` 通过。
- [x] 纯 CPU fake `smoke.py` 通过且未加载模型/联网。（`models_loaded=false`、`network_used=false`、24 cases PASS）
- [x] core KV/timeline/speculative/supplement smoke 通过。（kvcrop 43 项 PASS；timeline 71 项 PASS；speculative 54 项 PASS；sci34 smoke PASS）
- [x] integration pilot 为独立 non-formal run，仅作兼容性/成本预检。（`--limit 3`，transformers 7B）
- [x] Pilot 未被用于授予或补足 formal termination 资格，也未用于调整 frozen tolerance、case、token 数或 protocol。
- [ ] formal 的 24 条 record 均各自重新执行并通过 termination probe；未引用 pilot probe 结果。（formal 各自重跑 probe 属实，但 **20/24 合格**：4 个 `natural_eos` 未在 128 内 EOS，见 §5/§6）

Pilot 记录（不进入 formal）：3 cases 均 probe 合格，但 BF16 logit 门槛失败（max_abs 0.25–0.4375）——与 formal 失败同模式，已在 formal 前如实预警；按 handoff §5 该情形不属允许提前停止的 probe 不可行类，formal 照常执行。

## 5. Raw termination 与 retained-token correctness（逐 record/checkpoint 硬性）

- [ ] `termination_probes.required == observed == qualified == 24`；每个 checkpoint 的 probe 与所属 record 完全一致。（required=observed=24，**qualified=20**；validator 复核一致）
- [ ] `natural_eos` 全部为真实模型 greedy，128-token 冻结上限内观测 EOS，结束 phase 为 `ASSISTANT_EOT_PENDING`。（**6/10 合格**：eos step 8/20/21/80/98/111；4 个未命中）
- [x] `eos_at_cap` 全部明确 `controlled=true`，cap=4 且 EOT 精确位于最后一步；fixture 末 token 为 EOT，内容 token 仍走 production KV append。（6/6，eos_step 全为 4）
- [x] `max_tokens` 全部为真实模型 greedy，预算 2 内未 EOS，显式 `MAX_TOKENS` 且 phase 为 `ASSISTANT_OPEN`。（8/8）
- [x] 所有 EOS probe 的 pending EOT 均未进入 KV、完整 token ledger 或 assistant 内容 ledger；内容 IDs/hash/count 与长度关系可独立复算。（0 违例）
- [x] `crop_pending_eot` 在 teacher-force 内容后真实调用一次受控 EOT 的 `generate_accumulating(max_new_tokens=1)`，`pending_before_crop=true`、EOT 不进 ledger，截断 crop 后 pending 清除，再 reopen。（scenario_execution 24/24 passed）
- [x] `reply_tail_noop` 同样真实进入 EOT pending，`crop_to_token(current_seq)` 后 `no_op_preserved_pending=true`，再 reopen；record/checkpoint 的 scenario execution 完全一致。
- [x] canonical 与 crop path token IDs 100% exact，token hash 可独立复算。（45/45 checkpoint `token_ids_exact=true`）
- [x] first mismatch 对所有 checkpoint 为 null。
- [x] `seq == mask == KV == token ledger`。（state errors 全空）
- [x] assistant 内容 spans 与内容账本 exact；结构 EOT 未混入内容账本。
- [x] role phase/end reason 合法。
- [x] 每个 assistant→user boundary 恰好一个 EOT，EOT 位置 exact。（`unique_eot.ok=true` 45/45）
- [x] next-token top-1 100% exact。（45/45 `top1_exact=true`）
- [x] top-5 overlap 全部 `>=4/5`，完整分布已报告。（min 4、mean 4.87/5）
- [ ] BF16 FP32 logit diff 全部 `max_abs<=0.1` 且 `mean_abs<=0.01`；RMS 已报告。（**45/45 超阈**：max_abs 0.15625–0.96875，mean_abs 0.02017–0.15647，RMS 0.02677–0.16429）
- [ ] 32-token greedy continuation 100% exact；所有 `continuation_source=actual_crop_cache`，clean side 为 `clean_prefill_cache`，checkpoint state/logits 在 continuation mutation 前捕获。（**30/45 exact**；15 个发散，首个发散位 0/2/4/5/8/14/16/17/18/19/25；`continuation_source` 核对为 actual_crop_cache）
- [x] `full_rollback_p0` 保留 assistant header、提交 empty assistant EOT，assistant boundary=1；`speculation_full_invalidation` 删除完整 transition、保持原 user open，assistant boundary=0；两类 token 序列不混同且 next-user 未被人为加换行。（p0 boundary=1、invalidation boundary=0 均如协议）
- [ ] next-turn continuation 与后续第二 crop checkpoint 100% exact。（next_assistant checkpoint 中 11 个 continuation 发散）

| 指标 | 结果 | worst case/checkpoint |
|---|---:|---|
| Cases / checkpoints | 24 / 45 | — |
| Termination probes qualified | 20 / 24 | `c2_07_next_short_eos`、`c2_13_pending_medium_eos`、`c2_16_second_medium_eos`、`c2_19_mid_long_eos`（natural EOS 128 内未命中） |
| Natural EOS observed/cap | 6/10 合格（step 8–111 / 128） | 4 case 无 EOS |
| Controlled EOS-at-cap positions | 6/6 全为 4/4 | 无 |
| MAX_TOKENS observed/budget | 8/8 全为 2 | 无 |
| Pending EOT in KV/full/content ledger | 0 / 0 / 0 | 无 |
| Token/state exact rate | 45/45 = 100% | 无失败 |
| Top-1 exact rate | 45/45 = 100% | 无失败 |
| Top-5 overlap min/mean | 4 / 4.867 | 门槛 ≥4 全过 |
| Max absolute logit diff | 全部 45 个 checkpoint 超阈 0.1 | 0.96875 @ `c2_06_invalidate_short_max/next_assistant` |
| Worst mean absolute logit diff | 全部超阈 0.01 | 0.15647 @ `c2_11_mid_medium_max/post_recovery` |
| Worst RMS | — | 0.16429 @ `c2_11_mid_medium_max/post_recovery` |
| Continuation exact rate | 30/45 = 66.7% | 首个发散位 0 @ `c2_16_second_medium_eos`、`c2_24_second_long_max` |
| Unique EOT failures | 0 | 无 |

## 6. 失败、attempt 与 resume 审计

| UTC time | Case | Attempt / process identity | 现象 | 工件 | 处置 |
|---|---|---|---|---|---|
| 2026-09-03T01:32Z | `c2pilot…`（3 cases） | pilot 独立进程 | probe 合格；BF16 logit 门槛超阈（同 formal 模式） | `c2pilot_563dd22a_20260903T013205Z/` | 保留；不进 formal；不据此改 protocol |
| 2026-09-03T01:36–01:40Z | 全部 24 formal cases | `pid-10378-1788399408595489786-9d247206ebc54669aa90cf0945e818a5`（单进程 24 attempts，无 resume） | 4 个 `natural_eos` 未在 128 内 EOS；45/45 checkpoint logit 超阈；15/45 continuation 发散 | `records.jsonl` 24 条 + `failures/*.npz` 45 个 sidecar + `logs/formal.log` | 全部保留；runner 非零退出；不 acceptance/seal |

- [x] 所有异常 attempt 均在 `attempts.jsonl`，无半条 records。（24 attempts，1 个 process identity）
- [x] Resume 只跳过已有完整 case，manifest/cases/model/code identity 未变化。（未发生中断，无 resume）
- [x] 失败 logits `.npz` sidecar 已保留。（45 个）
- [x] 若存在任一 formal failure，本表状态不是 accepted，且未 seal。（状态 `rejected`；`seal --create` 未运行——见 §7）

## 7. Validate → analyze → acceptance → seal → tar（硬性顺序）

- [x] `summary.json` 存在，formal validator 已从 records 独立核对其 case/checkpoint/failed/process/identity/probe 计数和 verdict。（`status=FAIL`、`acceptance_eligible=false` 复核一致）
- [x] `validation.json` 首先生成且 `ok=true`、`acceptance_eligible=true`。（**实际 `ok=false`、`acceptance_eligible=false`、227 errors、probes 24/24/20**——fail closed 如设计）
- [ ] `analysis_v1.json` 之后生成；不含 bootstrap；按 context/scenario/termination/checkpoint 汇总。（**analyzer 拒绝为失败 campaign 生成 analysis**，`analysis_v1.json` 不存在，退出非零——如设计）
- [ ] analysis 列出 worst cases 与全部失败索引。（未生成，同上）
- [x] 本验收由 raw records 独立复核后填写。（token/state/top1/top5/continuation/probe 计数均由 raw `records.jsonl` 复算）
- [ ] 状态变为 accepted 后才运行 `seal --create`。（未 accepted，**未 seal**；`checksums.sha256` 由回传打包清单另行提供，非封存印章）
- [ ] `seal --verify` 通过。（未封存）
- [x] 最后创建 tarball 与 `.tar.gz.sha256`，回传完整目录而非 summary 子集。

| Artifact | 路径 / SHA-256 |
|---|---|
| validation | `validation.json`（`ok=false`；文件 sha 见 tarball 内 `checksums_return.sha256`） |
| analysis_v1 | 未生成（analyzer fail-closed 拒绝；`logs/analysis_v1.log` 保留退出栈） |
| ACCEPTANCE | `ACCEPTANCE.md`（本文件） |
| checksums.sha256 | 未生成（seal 拒绝后以 `checksums_return.sha256` 提供回传完整性清单，非封存） |
| tarball | `experiments/sci34_supplement/results/c2_equivalence/c2eq_563dd22a_20260903T013547Z.tar.gz` |
| tarball SHA-256 | 见同目录 `.tar.gz.sha256`（伴同校验文件为准） |

## 8. 最终签核

- 硬失败项：① 45/45 checkpoint BF16 logit 门槛（max_abs>0.1 且 mean_abs>0.01 全部成立）；② 15/45 checkpoint 32-token continuation 非 exact；③ 4/24 termination probe（natural_eos 128 内未 EOS）。
- Formal failed cases/checkpoints：24 cases / 45 checkpoints 中——probe 失败 4 cases（`c2_07/c2_13/c2_16/c2_19`）；logit 门槛失败 24 cases 全部 45 checkpoints；continuation 失败 11 cases 15 checkpoints（首个发散位 0–25）。
- 接受的 run：无（本 run `rejected`；pilot 不具 formal 资格）。
- 是否允许用于论文正确性主张：否——数值 logit 等价与 continuation 门槛未通过；token/state 层等价的正向观察可作后续设计侧讨论素材，但是否引用由设计侧另行授权。
- 是否需要新 run：需设计侧决策。失败两类成因不同：(a) 4 个 natural-EOS probe 未命中属冻结 cap 与该 snapshot greedy 行为的组合，确定性可复现，同机重跑无法改变；(b) logit 超阈在本机（RTX 3090 / cu128 / sdpa）呈系统性、量级 0.1–0.97，疑为增量 append 与整段 prefill 的 BF16 核归约差异叠加长上下文放大；若设计侧预期该路径应通过，需在设计侧环境复核或调整协议（后者须新协议版本，不得事后改本 run 容差）。
- 最终限定性结论：仅限冻结 Qwen2-7B-Instruct snapshot、BF16、sdpa、本机环境：crop/recovery 路径与同 retained IDs 的 canonical clean re-prefill 在 token 序列、KV/mask/seq/ledger 状态、内容账本、unique EOT、role phase 与 next-token top-1/top-5 上 100% 等价；在冻结 BF16 数值阈值（max_abs≤0.1、mean_abs≤0.01）与 32-token continuation exact 门槛上未通过；4/10 natural-EOS probe 在 128-token cap 内不终止。不外推其他模型/backend/dtype/硬件。

封存所需精确状态行（仅全部通过后取消注释并保留一行）：

<!-- Status: accepted -->
