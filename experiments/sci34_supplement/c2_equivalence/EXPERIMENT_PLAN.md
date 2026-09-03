# C2 正确性 campaign 实验计划

## 1. 研究问题

在冻结的 Qwen2-7B-Instruct、BF16、Transformers 实现上，播放感知 KV 路径执行“保留 assistant 内容 → crop → 唯一 EOT 关闭 → 打开 user/assistant role”后，其模型状态是否与使用**同一 retained token IDs**从空 cache 做 canonical clean re-prefill 等价？

这是实现正确性验收，不是性能估计。固定 1 个 deterministic session、24 cases、无统计重复、无 bootstrap、无显著性检验。

## 2. 独立 termination probe 与两条比较路径

### T：每个 formal case 自身的 termination 资格探针

探针与 retained-token comparison 使用独立 KV，从同一冻结 user context 开始，并真实消费 `StreamLLMInference.generate_accumulating()` 到结束：

- `natural_eos`：真实模型 greedy（temperature=0），冻结上限 128，必须在上限内观测 `EOS`；
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
- next-token top-1 与 top-5；
- logits 转 FP32 后的 max absolute、mean absolute、RMS difference；
- 32-token greedy continuation、hash、首个 divergence。

完整 logits 只在 checkpoint 失败时保存到 `failures/*.npz` 压缩 sidecar。失败记录与 sidecar不得删除。

## 6. 冻结硬门槛

先逐 record 且在每个 checkpoint 重复核对 termination probe；任一 probe 不合格即 case 失败。随后所有等价门槛逐 checkpoint 执行：

- `natural_eos`：真实 greedy、128 内 EOS、pending EOT 不进 KV/ledger；
- controlled `eos_at_cap`：EOT 精确位于 cap=4 最后一步并走 production EOS 分支；
- `max_tokens`：真实 greedy、预算 2、无 EOS、显式 `MAX_TOKENS`；
- canonical/path token IDs：100% exact；
- KV/mask/seq/ledger：100% exact；
- assistant 内容 span/账本：100% exact；
- role phase：100% exact；
- 每个已关闭 assistant boundary 恰好一个 EOT；
- next-token top-1：100% exact；
- top-5 overlap：至少 4/5，且完整报告；
- BF16 logits（比较前转 FP32）：`max_abs <= 0.1`、`mean_abs <= 0.01`，同时报告 RMS；
- 32-token greedy continuation：100% exact；path continuation 必须直接从已 crop/recovery 并刷新到 checkpoint 原长度的实际 KV 出发，禁止按 path IDs 从空 cache 重建副本；所有 state/logit/EOT 指标必须在 continuation 原地推进 cache 前捕获；
- 下一轮 continuation 与第二次 crop checkpoint：同样 100% exact。

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

允许：在冻结 Qwen2-7B-Instruct snapshot、BF16、attention backend、tokenizer/chat template、24-case 协议下，报告 crop/recovery 与 token-ID clean re-prefill 是否通过状态/next-token/continuation 等价门槛。

禁止：

- 跨模型、跨 dtype、跨 backend、跨 Transformers 版本泛化；
- 数值逐位 KV tensor 等价（本实验验收的是 token/state/logit/continuation）；
- 延迟、吞吐、质量提升或统计总体结论；
- 真实 ASR/TTS/播放器、声学停播或生产端到端正确性；
- 用 fake smoke 代替 7B forward evidence。
