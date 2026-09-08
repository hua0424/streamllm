# C2 正确性 campaign 实验计划

## 0. 协议 v2（2026-09-03，D-019；本节优先于下文与 v1 不同的细节）

**v1 formal run `c2eq_563dd22a_20260903T013547Z`（commit `1a47ac1`）判定 rejected 并永久归档**。它通过了全部 token/state 层门槛（token IDs、KV/mask/seq/ledger、内容账本、unique EOT、role phase 100% exact），但暴露出两类 v1 门槛的设计错误，而非实现缺陷：

1. **绝对 BF16 logit 阈（max_abs≤0.1、mean_abs≤0.01）与 32-token greedy continuation 100% exact 对任何正确实现都不可达成。** 机理（本地 0.5B/RTX 3060 与 7B/RTX 3090 双环境实证）：同形状重复计算差异精确为 0，而"增量 append vs 整段 prefill"因 kernel 归约顺序不同产生 max_abs 0.16–0.97 / mean_abs 0.02–0.16 的固有差异；纯 transformers 分块 append（零 crop 代码）给出同量级（本地测得 max_abs 0.20–0.33）。v1 中仅有的 top-1 翻转（运行时口径 43/45，GPU 报告的 45/45 系误统计）全部发生在 canonical top1–top2 margin ≤ 0.125（|logit|∈[16,32) 的 BF16 ulp）的精确近并列处，top-5 集合不变、仅前两名互换；continuation 15/45 发散全部始于 margin 受限的平缓区。
2. **4/10 `natural_eos` probe 的 greedy 在冻结 128-token cap 内 run-on**，属 cap×snapshot 行为组合，确定性可复现，同机重跑不可解；与等价性本身无关。

**v2 以先验冻结的机制性门槛替换绝对阈**（不得在看到 formal 结果后调整任何常数）：

- **噪声对照臂（新增）**：每个 checkpoint 把同一 canonical 序列按结构 seam 分块增量 re-prefill（镜像路径的 append 结构、以同样的单 token refresh 前向收尾，**不含任何 crop/生产恢复代码**），与整段 prefill 的 FP32 logit 差即本环境固有增量噪声 `control`。
- **相对门槛**：`path max_abs ≤ 2.0 × max(control max_abs, 0.05)`，`path mean_abs ≤ 2.0 × max(control mean_abs, 0.01)`；另设宽松绝对安全上限 `max_abs ≤ 2.0`、`mean_abs ≤ 0.5` 只拦粗错。本地 0.5B dry-run 实测 path/control 比值最差 **1.08**。
- **top-1 门槛**：须 exact，或 canonical top1–top2 margin ≤ `min(max(2.0 × max(control max_abs, 0.05), 0.125), 0.5)`（近并列翻转）。
- **continuation 门槛**：32-token greedy 须 exact，或首个发散步的 canonical margin 在上述近并列限内；每步记录 top1/top2/margin，greedy exact 率仅作描述性统计。
- **natural_eos**：cap 128→256；cap 内未 EOS 的 case 确定性**重资格化**为 max_tokens 语义并照常完成全部等价比较；campaign 级门槛要求 10 个 `natural_eos` 中**至少 5 个真实命中 EOS**（真实 EOS 分支覆盖；controlled `eos_at_cap` 6 例与 scenario pending-EOT 探针另行覆盖 EOS 分支）。
- **每 checkpoint 全量 logits sidecar**：`checkpoints/*.npz` 恒保存 path/canonical/control 三个 FP32 数组（不再只在失败时保存），validator 从 sidecar 独立重算全部统计、top-1/top-5/margin 与门槛。

不变项：24-case 冻结矩阵、8 scenario、3 termination 类、32-token continuation、top-5 ≥4/5、全部 token/state/账本/EOT/role 门槛、模型 snapshot 身份（D-017 artifact）、BF16、单逻辑 session、fail-closed 与封存规则。v1 工件不动，v2 以新 run-id 重跑。

## 1. 研究问题

在冻结的 Qwen2-7B-Instruct、BF16、Transformers 实现上，播放感知 KV 路径执行“保留 assistant 内容 → crop → 唯一 EOT 关闭 → 打开 user/assistant role”后，其模型状态是否与使用**同一 retained token IDs**从空 cache 做 canonical clean re-prefill 等价？

这是实现正确性验收，不是性能估计。固定 1 个 deterministic session、24 cases、无统计重复、无 bootstrap、无显著性检验。

## 2. 独立 termination probe 与两条比较路径

### T：每个 formal case 自身的 termination 资格探针

探针与 retained-token comparison 使用独立 KV，从同一冻结 user context 开始，并真实消费 `StreamLLMInference.generate_accumulating()` 到结束：

- `natural_eos`：真实模型 greedy（temperature=0），v2 冻结上限 256；cap 内命中 EOS 记 `genuine_eos=true`，未命中则确定性重资格化为 max_tokens 语义（`requalified=true`），campaign 级要求 ≥5/10 真实命中；
- `eos_at_cap`：明确标为 controlled 的确定性 decode-token fixture，固定 cap=4，前三步为非 EOT 内容并走 production KV append，第四步返回 EOT；必须由 `generate_accumulating()` 的 EOS 分支在最后一步留下 `ASSISTANT_EOT_PENDING`；
- `max_tokens`：真实模型 greedy，固定小预算 2，两个内容 token 均进入 KV/ledger，且未见 EOT、结束原因为 `MAX_TOKENS`。

EOS/EOT 被预测时不得进入 KV、完整 token ledger 或 assistant 内容 ledger。每条 record 和每个 checkpoint 都保存同一份可审计探针：observed end reason、内容 token IDs/hash/count、cap/EOS step、EOT ledger/KV 标志、role phase、controlled 说明、pass/errors。Formal runner 与独立 validator 均逐 case 硬校验；pilot 不承担 formal 资格。

### A：正式 crop + role 路径

1. 用项目 `StreamLLMInference` 建立另一份初始 canonical user→assistant KV。
2. 将冻结 `assistant_text` teacher-force 到实际 KV；`assistant_token_ids` 只记录内容 token。此步骤只为让两条等价路径共享完全相同的 retained IDs，不作为 termination 证据。
3. 根据 case 的片段边界决定 retained assistant token-ID 前缀。
4. 调用 `crop_to_token()`：播放期 `full_rollback_p0` 裁到 assistant content 起点，保留完整 assistant header；`speculation_full_invalidation` 才裁到 assistant role 起点（即 raw user 内容末端），删除 user-close/assistant-header transition；其余裁到 retained 内容末端。
5. `full_rollback_p0` 及有 retained 内容的路径调用 `reopen_user_role()`，由该操作唯一提交一个 EOT 并打开 user role；speculation full invalidation 保持原 user role open，直接追加后续 raw user segment，不注入换行。
6. 多轮 case 再 `prefill_user_text()`、`open_assistant_role()`，必要时执行第二次 crop/recovery。

### B：canonical clean oracle

1. 全部结构边界来自同一 tokenizer 的 `apply_chat_template(tokenize=True)`。
2. retained assistant token IDs 从路径 A 的原始 token-ID 账本直接切片并嵌入；禁止 assistant `decode→encode`。
3. 完整 canonical token IDs 从空 KV 一次性 prefill。
4. 比较当前 token 序列、状态不变量、next-token logits 与 continuation。

路径 B 不调用 crop/recovery API，不复用路径 A 的 KV。

## 3. 状态语义

- 完整账本：`token_ids`，要求 `len(token_ids) == seq_length == attention_mask length == DynamicCache length`。
- assistant 内容账本：结构 token（role header、EOT）不得混入。
- `ASSISTANT_EOT_PENDING`：模型已预测 EOT，但 EOT 尚未写入内容/KV。`crop_pending_eot` 在 teacher-force 内容后临时强制下一次 decode 为 EOT，并真实调用 `generate_accumulating(max_new_tokens=1)`，确认 pending/EOT 不入 ledger 后截断到 retained boundary，crop 必须清除 pending；`reply_tail_noop` 以同样方式进入 pending，再调用 `crop_to_token(current_seq)`，必须保留 pending，随后 reopen 唯一提交 EOT。
- `reopen_user_role()`：唯一允许正式提交 assistant EOT 的关闭操作。
- crop 不得落入 role 结构 token 中间；只允许可解释边界。
- 每个已关闭 assistant→user 边界在完整 token 序列中恰好一个 EOT。

## 4. 冻结 case 矩阵

`cases.json` 固定 24 个案例，全部 case ID 唯一且无运行时抽样。

覆盖维度：

- context target：512、2048、8192 canonical tokens；
- crop：p=0 全回滚、clean fragment boundary、mid-fragment 向命中片段末端吸附、reply-tail/no-op；
- termination：natural EOS、EOS 恰好位于 generation cap、max-token 无 EOS；
- state：crop 删除 pending EOT、speculation full invalidation；
- turn：下一 user、下一 assistant、后续轮第二次 crop。

每个 context target 有 8 cases；每个 scenario 在三个 context 档均出现。自然 EOS/预算末端 EOS/max-token 均跨 context 覆盖。`controlled_fixture=true` 明确标出依赖确定性状态注入而非自然生成时机的案例。

`termination` 不是描述性标签。每个 formal case 必须由其自身独立 probe 取得资格；formal 的 `natural_eos` 若在 128-token 冻结上限内未 EOS、`max_tokens` 若两步内出现 EOS、或 controlled `eos_at_cap` 未恰在第 4 步命中 EOS，runner 必须落盘 failed record 并阻止 acceptance。Pilot 仅用于预检性能/兼容性，不能替代或补足 formal probe。

## 5. 每 checkpoint 原始证据

每条 record 与其每个 checkpoint 保存 termination probe 的 observed end reason、内容 token IDs/hash/count、EOT 是否进入 KV/完整 ledger/内容 ledger、cap 与 EOS 位置、mode/controlled 说明、pass/errors；另保存 `scenario_execution`，审计 pending-before-crop、EOT ledger 状态、crop target/no-op、pending-after-crop、reopen 与 pass/errors。每个 checkpoint 还保存：

- canonical/path 完整 token IDs、数量和 SHA-256；
- 首个 token mismatch；
- KV/mask/seq/ledger 长度与 role phase/end reason；
- assistant 内容 spans 与内容账本检查；
- EOT 位置及 assistant boundary 数；
- next-token top-1 与 top-5，及 canonical/path top1–top2 margin；
- logits 转 FP32 后的 max absolute、mean absolute、RMS difference；
- v2 噪声对照臂：有效 seam 位置、chunk 数、control diff 统计；
- v2 门槛判定明细（`logit_gates`）；
- 32-token greedy continuation、hash、首个 divergence，及双侧逐步 top1/top2/margin。

v2 起每个 checkpoint 的完整 logits 恒保存为 `checkpoints/*.npz`（path/canonical/control 三个 FP32 数组），validator 从中独立重算。失败记录与 sidecar 不得删除。

## 6. 冻结硬门槛

先逐 record 且在每个 checkpoint 重复核对 termination probe；`eos_at_cap`/`max_tokens` 不合格即 case 失败，`natural_eos` 未命中 cap 者重资格化（须自洽），campaign 级要求 ≥5/10 真实 EOS。随后所有等价门槛逐 checkpoint 执行（validator 一律从 `checkpoints/*.npz` sidecar 独立重算）：

- `natural_eos`：真实 greedy、cap 256；genuine 或自洽 requalified，且 campaign ≥5/10 genuine；pending EOT 不进 KV/ledger；
- controlled `eos_at_cap`：EOT 精确位于 cap=4 最后一步并走 production EOS 分支；
- `max_tokens`：真实 greedy、预算 2、无 EOS、显式 `MAX_TOKENS`；
- canonical/path token IDs：100% exact；
- KV/mask/seq/ledger：100% exact；
- assistant 内容 span/账本：100% exact；
- role phase：100% exact；
- 每个已关闭 assistant boundary 恰好一个 EOT；
- next-token top-1：exact，或 canonical margin ≤ 近并列限（top-5 集合内互换）；
- top-5 overlap：至少 4/5，且完整报告；
- v2 相对 logit 门槛：`max_abs ≤ 2.0 × max(control max_abs, 0.05)`、`mean_abs ≤ 2.0 × max(control mean_abs, 0.01)`；绝对安全上限 `max_abs ≤ 2.0`、`mean_abs ≤ 0.5`；
- 32-token greedy continuation：exact，或首个发散步 canonical margin ≤ 近并列限 `min(max(2.0 × max(control max_abs, 0.05), 0.125), 0.5)`；path continuation 必须直接从已 crop/recovery 并刷新到 checkpoint 原长度的实际 KV 出发，禁止按 path IDs 从空 cache 重建副本；所有 state/logit/EOT 指标必须在 continuation 原地推进 cache 前捕获；
- 下一轮 continuation 与第二次 crop checkpoint：同样适用上述门槛。

任一 formal case/checkpoint 失败：

1. raw record、attempt 日志和失败 logits sidecar 保留；
2. `summary.acceptance_eligible=false`；
3. validator 非零退出；
4. analyzer 拒绝生成 accepted analysis；
5. seal 拒绝封存。

不得删除 case、事后修改同一协议容差或只报告通过子集。

## 7. 工件与 resume

正式目录：

```text
experiments/sci34_supplement/results/c2_equivalence/<run_id>/
```

至少包含 `campaign_manifest.json`、`cases.json`、`records.jsonl`、`attempts.jsonl`、`progress.json`、`summary.json`、`validation.json`、`analysis_v1.json`、logs、snapshots、`ACCEPTANCE.md`、`checksums.sha256`。Tarball 及其 hash 位于结果根目录同级。

`run.py --resume` 按完整 case key 跳过已完成记录。每个 case 在所有 checkpoint 完成后，才通过原子替换写入 `records.jsonl`；异常 attempt 单独写入 `attempts.jsonl`，因此进程重启不会留下半条 accepted record。Resume 允许新的 process identity，但仍属于唯一逻辑 session `s01`，所有 process/attempt 身份均保留审计。

## 8. 允许与禁止主张

允许：在冻结 Qwen2-7B-Instruct snapshot、BF16、attention backend、tokenizer/chat template、24-case 协议 v2 下，报告 crop/recovery 与 token-ID clean re-prefill 是否通过 token/state 等价门槛，以及 logit 分布是否在环境固有增量噪声的 2 倍以内（近并列翻转/发散仅在受限 margin 内允许）。

禁止：

- 跨模型、跨 dtype、跨 backend、跨 Transformers 版本泛化；
- 数值逐位 KV tensor 等价（本实验验收的是 token/state/logit/continuation）；
- 延迟、吞吐、质量提升或统计总体结论；
- 真实 ASR/TTS/播放器、声学停播或生产端到端正确性；
- 用 fake smoke 代替 7B forward evidence。
