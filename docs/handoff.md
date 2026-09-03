# 二期 C2 正确性补强交接文档（协议 v2 复跑）

> 面向实验机执行者和后续论文修订会话。当前唯一待执行的正式 GPU 工作是 **C2 协议 v2 的 Qwen2-7B 正确性验收复跑**。既有 E1/E2、E3、A1、P1 已完成；C2 v1 run 已执行并判定 rejected（归档保留），不得重跑或覆盖。

**更新时间**：2026-09-03

**分支**：`paper2`

**状态**：C2 v1 formal `c2eq_563dd22a_20260903T013547Z` 已执行并 rejected（token/state 层 100% 等价；v1 绝对 logit 阈与 greedy-exact 门槛被证明对任何正确实现不可达成，4/10 natural_eos greedy run-on 属 cap×snapshot 组合）。`src/` 无缺陷、不改。协议 v2（噪声对照臂 + 相对门槛 + margin 规则 + natural cap 256/重资格化）已实现并在本地通过 fake smoke 与 0.5B CUDA 真模型 dry-run（path/control 比值最差 1.08）。v2 Qwen2-7B formal 待实验机执行。

**GPU 唯一入口**：`experiments/sci34_supplement/c2_equivalence/GPU_HANDOFF.md`（协议 v2 版）

---

## 一、本轮为什么需要 GPU

第二次 SCI Q3/Q4 审稿指出两个必须闭环的问题，D-018 已修复代码并冻结 C2 协议，v1 轮（2026-09-03）完成了首次 formal 执行：

1. 原实现把 Qwen 的 `<|im_end|>` 同时当作生成 EOS 和 assistant 关闭 token，可能形成重复 EOT。显式 role phase / generation end reason 已修复，v1 formal 在 7B 上验证 token/state 层 100% 等价。
2. 尚缺"crop+role recovery 与 clean re-prefill 数值等价"的正式证据。v1 的绝对阈设计错误（见下），v2 以机制性相对门槛重跑。

**v1 rejected 的定性与证据（D-019，设计侧独立审计 raw records/NPZ + 本地 0.5B 对照实验）**：

- 增量 append 与整段 prefill 的 BF16 kernel 归约差异是环境固有噪声：同形状重复计算差精确为 0；纯 transformers 分块 append（零 crop 代码）在 0.5B/RTX 3060 上给出 max_abs 0.20–0.33，与 7B/RTX 3090 观测（0.16–0.97）同阶。v1 冻结的 `max_abs<=0.1 / mean_abs<=0.01` 与 32-token greedy 100% exact 对任何正确实现不可达成。
- 仅有的 top-1 翻转（运行时 43/45，报告的 45/45 系误统计）全部发生在 margin ≤ 0.125（BF16 ulp）的近并列处、top-5 集合内互换；continuation 15 处发散全部始于平缓/退化 greedy 区。
- 4/10 natural_eos 在 128 cap 内 run-on，确定性可复现，与等价性无关。

这不是新的时延实验，不产生 TTFT、mouth-to-ear 或生产 barge-in headline。

---

## 二、v2 正式协议（相对 v1 的差异）

不变项：模型（D-017 accepted Qwen2-7B artifact）、BF16、单卡、24-case 冻结矩阵、8 scenario、3 termination、32-token continuation、top-5 ≥4/5、全部 token/state/账本/unique-EOT/role 门槛、单逻辑 session、fail-closed 与封存规则。

v2 变更（常数先验冻结）：

- **噪声对照臂**：每 checkpoint 将 canonical 序列按结构 seam 分块增量 re-prefill（不含 crop 代码），其与整段 prefill 的 FP32 diff 即 `control`；
- **相对门槛**：path max_abs ≤ 2.0×max(control max_abs, 0.05)、mean_abs ≤ 2.0×max(control mean_abs, 0.01)；绝对安全上限 2.0/0.5；
- **margin 规则**：top-1 翻转与 continuation 发散仅允许在近并列 margin `min(max(2.0×max(control max_abs,0.05),0.125),0.5)` 内；逐步记录 top1/top2/margin；
- **natural_eos**：cap 256；未命中者确定性重资格化为 max_tokens 语义并完成全部等价比较；campaign 级 ≥5/10 genuine；
- **工件**：每 checkpoint 恒存 `checkpoints/*.npz`（path/canonical/control 三 FP32 数组），validator 从 sidecar 独立重算。

详细定义见 `experiments/sci34_supplement/c2_equivalence/EXPERIMENT_PLAN.md` §0 与 `ACCEPTANCE_TEMPLATE.md`。

---

## 三、实验机执行顺序

不要从本文复制零散命令。按以下唯一入口逐节执行：

```text
experiments/sci34_supplement/c2_equivalence/GPU_HANDOFF.md
```

要点：协议版本预检（必须 `PROTOCOL_VERSION == 2`）、旧结果 guard（**含 v1 C2 归档 run 与 `e3_exact_rescue/`，全部只读**）、严格离线、CLI/smoke（fake smoke 须报 `protocol_version=2`、`checkpoint_sidecars=45`）、模型预检、v2 pilot、冻结 manifest、单 session formal run、validate→analyze→acceptance→seal→tar、完整回传。E3 抢救在 v1 轮已完成入库，本轮默认跳过。

预计 GPU 用时 10 分钟内（v1 约 5 分钟，v2 增加对照臂 prefill）；tarball 约 60–100MB。

Formal 启动前工作树必须 clean。执行者应使用设计侧推送的 v2 代码 commit，不得用 dirty tree 生成论文证据。

---

## 四、回传件与验收门槛

必须回传完整 formal run（含 `checkpoints/` 45 个三数组 sidecar）、tarball 与 tarball SHA-256。

硬门槛（v2）：

- token serialization、结构边界、KV/mask/ledger：100% exact；
- 每个 assistant→user 边界恰好一个 EOT；
- termination probe 与 case 标签/状态一致（natural genuine 或自洽 requalified，campaign ≥5/10 genuine）；EOT 不进入内容 ledger/KV；
- next-token top-1：exact，或近并列翻转（margin 受限，top-5 集合内）；
- top-5 overlap：每 checkpoint 至少 4/5；
- 相对 logit 门槛：2× 控制臂噪声 + 绝对安全上限，validator 从 NPZ 独立重算一致；
- 32-token continuation：exact，或首个发散步 margin 在近并列限内；
- validation、analysis、acceptance、seal 和旧结果 guard（含 v1 归档）全绿。

只有全部通过后，`ACCEPTANCE.md` 才能包含精确状态行 `Status: accepted`。

---

## 五、允许与禁止的结论

通过后只允许说明：

> 在冻结的 Qwen2-7B snapshot、tokenizer/chat template、BF16/backend 和 24-case 协议 v2 下，修复后的 crop/recovery 路径与 canonical token-ID clean re-prefill 在 token/state 层 100% 等价，且 logit 分布偏差不超过环境固有增量 append BF16 噪声的 2 倍（top-1 翻转与 greedy 发散仅出现在近并列处）。

不得据此声称：

- 所有模型、模板、dtype 或推理引擎普适等价；
- 逐位 KV tensor 等价；
- 真实声卡或声学“用户实际听到”边界正确；
- 真实 ASR/TTS/播放器并发链路正确；
- 生产 barge-in、TTFT、mouth-to-ear 或用户体验改善。

---

## 六、旧结果与论文冻结规则

以下既有工件不重跑、不覆盖（v1 C2 归档含失败工件与 NPZ 全量保留）：

- C2 v1 formal（rejected 归档）：`c2eq_563dd22a_20260903T013547Z` 及 pilot `c2pilot_563dd22a_*`；
- E3 抢救件：`results/e3_exact_rescue/`；
- C-E1/E2：`e1e2c_b8c758b_20260901T173306Z`；
- 固定轨迹 E3：`sci34_f11ccba_20260901_e3`；
- 联合 A1：`sci34_f11ccba_20260901_a1`；
- P1 v2：`sci34_dc52978_20260901_async_prepared_v2`；
- `experiments/results/` 中全部旧 JSON 与 `paper2_reanalysis.json`。

GPU 正式结果验收前，不修改 `paper2/` 正文、图表与数字。结果回传后统一处理二审的统计重分析、主张收窄和全文修订。

---

## 七、关键文件

- 核心状态机（本轮不改）：`src/llm/stream_llm_inference.py`
- 编排适配（本轮不改）：`src/dialogue/orchestrator.py`
- timeline 合同（本轮不改）：`src/dialogue/timeline.py`
- C2 campaign（v2）：`experiments/sci34_supplement/c2_equivalence/`
- 正式 GPU 入口：`experiments/sci34_supplement/c2_equivalence/GPU_HANDOFF.md`
- 协议决策：`docs/decisions.md` D-018（v1 冻结）、D-019（v1 rejected + v2）
- 二审意见：`paper2/review/sci_q3_q4_full_review_2026-09-02.md`
- v1 执行记录：`experiments/sci34_supplement/results/GPU_RUN_NOTES.md` C2 节、`results/c2_equivalence/c2eq_563dd22a_20260903T013547Z/ACCEPTANCE.md`
