# 二期 C2 v3 crop-integrity GPU 交接

> 当前唯一待执行的正式 GPU 工作是 **C2 protocol v3 exact-only crop-integrity addendum**。C2 v1/v2 均已执行并按各自冻结门槛 rejected，完整归档保留；既有 E1/E2/E3/A1/P1 不重跑、不覆盖。

**更新时间**：2026-09-03

**分支**：`paper2`

**GPU 唯一入口**：`experiments/sci34_supplement/c2_crop_integrity/GPU_HANDOFF.md`

## 当前结论

- `src/` 没有发现新的实现缺陷，不改。
- C2 v2 run `c2eq_5c56b014_20260903T040829Z` 保持 rejected：24/24 probe、45/45 token/state/EOT/scenario、top-k/近并列行为均符合预期，但预注册的单控制 2× raw-logit 比值仅 42/45。
- 不能把 2.0× 事后放宽到 2.7×。独立审计确认 v2 control 与 production forward/chunk 拓扑不匹配，且三项失败分别由无预测意义的尾部 token 极值或 softmax 不变的常数偏移主导，不能归因于 crop。
- v3 改为直接、无经验容差地回答“crop 是否改变已保留 KV”：从同一个 pre-crop cache 出发，production `crop_to_token` 与独立 K/V prefix clone oracle 配对，全部要求 bitwise/exact。
- 7B v3 pilot `c2crop_pilot_b2c6f22b_20260903T064135Z`（`91ea218`）7/8 cases 通过并验证 K/V exact；唯一暴露生产状态缺陷：invalidation crop 后 `prefill_user_text` 残留陈旧 `CROPPED`。D-022 已修生产 API（新 user 内容成功追加后清为 `NONE`）并加真实 seam 回归；v3 协议不变。

## v3 冻结设计

- 精确复制 v2 的 24 cases（SHA-256 `acda9afb83ff393d222fc06eb433c1a307dba1eb213fcbbbedfdb860f7500696`），24 records / 27 crop events；覆盖全部 8 scenario、512/2048/8192 与三次 second crop。
- 不重跑 v2 termination probe；v2 run ID 仅作为固定 provenance，runtime 不读取其工件。
- assistant fixture 以受控非 EOT token selection 逐 token 走真实 `generate_accumulating → _prefill_ids_p2`，硬验每 token 一次 forward；第一轮与第二轮 ledger 均独立验证。
- crop 前记录全层 retained key/value 的 shape、dtype、device 与 SHA-256；独立 clone oracle 不调用 production crop；production arm 唯一调用 `crop_to_token`。
- crop 后、恢复前要求 pre-prefix = production post = oracle 的 K/V 逐张量 `torch.equal`/hash exact；mask、token ledger、seq/KV 长度 exact；validator 独立从 case/fragment/fraction 推导 keep，并检查所有层 `shape[-2] == keep`。
- 之后 production role API 与 oracle 使用相同 token-ID chunk、position/mask forward；每步要求 K/V、logits、mask、ledger、retained prefix 与 role/end/content 状态 bitwise/exact。
- wrong-length negative control 必须检出；fake smoke 还篡改 wrong keep、layer hash、duplicate EOT、missing event，并验证缺 snapshots 不能 seal。
- 无 clean re-prefill 数值门槛、无 2×/3× 容差、无统计重复。

## 本地验证

- `py_compile/compileall`：PASS。
- fake full workflow：PASS（24 cases / 27 crop events，四类 tamper 全检出）。
- 0.5B CUDA 真模型代表性 pilot：c2_01 + c2_08，共 3 次真实 crop（含 next-turn 和 second crop）全部 bitwise/exact；production post、pre-prefix、oracle K/V 相等，恢复后的 K/V/logits/mask/token/state 全相等。

## GPU 执行

只按以下文件逐节执行：

```text
experiments/sci34_supplement/c2_crop_integrity/GPU_HANDOFF.md
```

该文档包含 exact clean commit、`uv sync --frozen`、严格离线、本地 Qwen2-7B、全部旧结果 SHA guard、五项 smoke、8-case pilot、before/after snapshot、不可变 manifest、case 原子 resume、validate→analyze→acceptance→seal→tar 与防覆盖规则。

GPU 侧使用修复 commit，从 handoff §3 的 8-case pilot 重新开始；pilot 不可 resume/复用旧目录，因为 code identity 已变化。预计低于或接近 v2 的约 6 分钟 formal：v3 不跑 256-token termination 和双 32-token continuation，但增加短 assistant 的逐-token forward 与 KV hash。

## 允许的最终主张

若 v3 24/27 全过，只允许：

> 在冻结 Qwen2-7B snapshot、BF16/SDPA/backend 和 24-case/27-event addendum 下，生产 crop 保留的 K/V 前缀与 crop 前前缀及独立切片 oracle 逐张量 bitwise exact；相同 chunk 恢复后的 K/V、logits、mask、token ledger 和 role/end 状态也 bitwise/exact 一致。

不得主张：clean re-prefill 数值等价、v2 通过、跨模型/backend 普适、真实 ASR/TTS/声卡、生产端到端正确性、时延或质量提升。

## 只读工件

- C2 v1 formal：`c2eq_563dd22a_20260903T013547Z`；
- C2 v2 formal：`c2eq_5c56b014_20260903T040829Z` 及其 pilots；
- E3 rescue：`results/e3_exact_rescue/`；
- C-E1/E2、固定轨迹 E3、联合 A1、P1 v2；
- `experiments/results/` 全部旧 JSON；
- `paper2/` 正文、图表与现有数字。

GPU v3 结果回传验收后，再统一改论文与做二审统计重分析。
