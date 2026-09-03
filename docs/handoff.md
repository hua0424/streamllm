# 二期 C2 正确性补强交接文档

> 面向实验机执行者和后续论文修订会话。当前唯一待执行的正式 GPU 工作是 **C2 crop/recovery 与 canonical clean re-prefill 的 Qwen2-7B 正确性验收**。既有 E1/E2、E3、A1、P1 已完成，不得重跑或覆盖。

**更新时间**：2026-09-02

**分支**：`paper2`

**状态**：EOS/EOT 与角色状态机代码已修复；C2 独立 campaign 已实现；本地 CPU/fake smoke 与 0.5B 模型回归已通过；Qwen2-7B formal 证据待实验机执行。

**GPU 唯一入口**：`experiments/sci34_supplement/c2_equivalence/GPU_HANDOFF.md`

---

## 一、本轮为什么需要 GPU

第二次 SCI Q3/Q4 审稿指出两个必须闭环的问题：

1. 原实现把 Qwen 的 `<|im_end|>` 同时当作生成 EOS 和 assistant 关闭 token。生成器曾先把 EOT 写入 KV/assistant 账本，`reopen_user_role()` 随后又写一次关闭边界，正常 EOS 后可能形成重复 EOT。
2. 既有测试只证明 KV、mask 和长度结构合法以及 crop 后仍能生成；A1 只测时延，P1 只测软件控制路径。它们没有证明 crop+role recovery 与相同 retained history 的 clean re-prefill 在规范 token 序列、logits 和后续生成上等价。

代码现已采用显式 role phase 和 generation end reason：assistant EOT 不进入内容账本或 KV，而由 `reopen_user_role()` 唯一提交一次。正式 GPU campaign 用独立 token-ID clean oracle 验收这条状态路径。

这不是新的时延实验，不产生 TTFT、mouth-to-ear 或生产 barge-in headline。

---

## 二、正式协议

正式协议冻结为：

- 模型：显式本地 Qwen2-7B-Instruct snapshot；
- 运行：Transformers、BF16、单卡、batch size 1；
- 网格：24 个确定性 case，1 个逻辑 session，无统计重复、无 bootstrap；
- 上下文：512、2048、8192 canonical tokens；
- 覆盖：p=0 全回滚、clean fragment boundary、mid-fragment 吸附、reply-tail/no-op、pending EOT crop、推测全作废、下一 user/assistant、后续轮第二次 crop；
- 结束分支：真实 natural EOS、受控 EOS-at-cap、真实 max-token；每个 formal case 自身必须通过 termination probe，不能只依赖标签或 pilot；
- 比较：实际 crop/recovery 路径与相同 retained token IDs 的 canonical clean re-prefill；
- 输出：完整 token hashes/边界、role/end state、next-token top-1/top-5、FP32 logit 差、32-token greedy continuation、失败 sidecar；
- 失败处理：任何 case/checkpoint 失败均保留 raw/attempt/NPZ，并判定 formal 未通过；不得删 case 或事后放宽同一协议。

详细定义见：

- `experiments/sci34_supplement/c2_equivalence/EXPERIMENT_PLAN.md`
- `experiments/sci34_supplement/c2_equivalence/ACCEPTANCE_TEMPLATE.md`

---

## 三、实验机执行顺序

不要从本文复制零散命令。按以下唯一入口逐节执行：

```text
experiments/sci34_supplement/c2_equivalence/GPU_HANDOFF.md
```

它包含：

1. checkout 包含本轮代码的 exact clean commit；
2. 旧结果跑前 SHA-256 guard；
3. 严格离线环境、空 HF token、显式本地模型目录；
4. CLI `--help`、CPU smoke、核心状态机回归；
5. tokenizer/chat template/EOS/EOT identity 检查；
6. 独立 integration pilot；
7. 创建不可变 formal manifest；
8. 单 session formal run 与 case 原子 resume；
9. 跑后环境 snapshot 和旧结果 hash guard；
10. `validate → analyze → acceptance → seal → tar`；
11. 完整目录、tarball/hash 和 E3 数据抢救件回传。

Formal 启动前工作树必须 clean。当前会话不负责 commit/push；执行者应使用后续明确提供的代码 commit，不得用 `--allow-dirty` 生成论文证据。

---

## 四、同时从原实验机抢救 E3 输入

正式 E3 使用的 exact 文件目前未入库：

```text
/root/autodl-tmp/dataA/streamllm/experiments/datasets/processed/p2_turns.json
```

E3 manifest 记录的 SHA-256：

```text
a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c
```

按 GPU handoff 的“E3 exact 数据抢救”步骤保存：

- exact `p2_turns.json` 及 hash；
- raw MultiWOZ 路径/hash；
- `prepare_multiwoz_data.py`、命令和可用 provenance；
- E3 manifest 与模型 snapshot identity。

若原路径不存在，只能如实记录 `MISSING`；不得重新生成一个文件后冒充 exact 输入。

---

## 五、回传件与验收门槛

必须回传完整 formal run，而不是 summary 子集：

```text
results/c2_equivalence/<run_id>/
├── campaign_manifest.json
├── cases.json
├── records.jsonl
├── attempts.jsonl
├── progress.json
├── summary.json
├── failures/*.npz            # 仅失败 checkpoint
├── logs/
├── snapshots/before|after/
├── validation.json
├── analysis_v1.json
├── ACCEPTANCE.md
└── checksums.sha256
```

另回传 tarball、tarball SHA-256 和 E3 rescue 目录。

硬门槛：

- token serialization、结构边界、KV/mask/ledger：100% exact；
- 每个 assistant→user 边界恰好一个 EOT；
- termination probe 与 case 标签/状态一致，EOT 不进入内容 ledger/KV；
- next-token top-1：100% exact；
- top-5 overlap：每 checkpoint 至少 4/5；
- BF16 logits 转 FP32 后 `max_abs <= 0.1`、`mean_abs <= 0.01`；
- 32-token continuation 及下一轮 continuation：100% exact；
- validation、analysis、acceptance、seal 和旧结果 guard 全绿。

只有全部通过后，`ACCEPTANCE.md` 才能包含精确状态行 `Status: accepted`。

---

## 六、允许与禁止的结论

通过后只允许说明：

> 在冻结的 Qwen2-7B snapshot、tokenizer/chat template、BF16/backend 和 24-case 协议下，修复后的 crop/recovery 路径与 canonical token-ID clean re-prefill 满足预定义的结构、next-token 和 continuation 正确性门槛。

不得据此声称：

- 所有模型、模板、dtype 或推理引擎普适等价；
- 真实声卡或声学“用户实际听到”边界正确；
- 真实 ASR/TTS/播放器并发链路正确；
- 生产 barge-in、TTFT、mouth-to-ear 或用户体验改善。

---

## 七、旧结果与论文冻结规则

以下既有 campaign 不重跑、不覆盖：

- C-E1/E2：`e1e2c_b8c758b_20260901T173306Z`；
- 固定轨迹 E3：`sci34_f11ccba_20260901_e3`；
- 联合 A1：`sci34_f11ccba_20260901_a1`；
- P1 v2：`sci34_dc52978_20260901_async_prepared_v2`；
- `experiments/results/` 中全部旧 JSON 与 `paper2_reanalysis.json`。

GPU 正式结果验收前，不修改：

- `paper2/abstract.md`；
- `paper2/chapter1_introduction.md` 至 `chapter8_conclusion.md`；
- `paper2/thesis_draft.md`；
- IEEE 衍生稿、图表和现有论文数字。

结果回传后再统一处理二审的统计重分析、主张收窄和全文修订。

---

## 八、关键文件

- 核心状态机：`src/llm/stream_llm_inference.py`
- 编排适配：`src/dialogue/orchestrator.py`
- timeline 合同：`src/dialogue/timeline.py`
- 核心模型回归：`src/llm/run_kvcrop_test.py`
- 推测回归：`src/dialogue/run_speculative_test.py`
- timeline 负向回归：`src/dialogue/run_timeline_test.py`
- C2 campaign：`experiments/sci34_supplement/c2_equivalence/`
- 正式 GPU 入口：`experiments/sci34_supplement/c2_equivalence/GPU_HANDOFF.md`
- 协议冻结决策：`docs/decisions.md` D-018
- 二审意见：`paper2/review/sci_q3_q4_full_review_2026-09-02.md`
