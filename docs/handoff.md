# 二期论文统一修订交接（C2 v3 已正式接受）

> GPU 补实验阶段已结束。当前没有待执行的无条件 GPU 任务。下一阶段是按第二次审稿意见统一修改论文；既有 E1/E2/E3/A1/P1 与 C2 v1/v2/v3 工件全部只读，不得重跑或覆盖。

**更新时间**：2026-09-03

**分支**：`paper2`

**当前状态**：C2 v3 formal `c2crop_82103004_20260903T080512Z` 已按 D-023 正式接受并封存。24/24 cases、27/27 crop events 全 exact；validation 零错误；ACCEPTANCE accepted；seal 通过。GPU 正确性阻塞解除。

## 一、已接受的 C2 v3 证据

- Code commit：`82103004637dce8f98688f4a685d33ebee363a3b`
- Result commit：`7d50624`
- Manifest SHA-256：`d8c3db4d609234a072064162a5caa443e25171b2311d84afa48b7b6a4f1d4bc2`
- Records SHA-256：`f775ba238f17439b2b1831f31cbb97eb8ade87ddc7e2517c8eba427ee8b21725`
- Seal SHA-256：`e0997d41793f510fc1120a7c3f08c420097813cc627f08d47716e76b4489f4a9`
- Formal directory：`experiments/sci34_supplement/results/c2_crop_integrity/c2crop_82103004_20260903T080512Z/`

设计侧独立复核：24 records / 27 events / 3 no-op / 3 second-crop ledgers / 60 recovery steps / 28 层 K/V 全部 exact；308 个 assistant fixture token 均逐 token 走 production append；27/27 wrong-length negative control 检出；381-file legacy guard 一致；模型为 accepted Qwen2-7B、BF16、SDPA、strict offline、clean commit。

## 二、论文允许写入的 C2 结论

只能写：

> 在冻结 Qwen2-7B snapshot、BF16/SDPA/Transformers backend 和 24-case/27-event v3 addendum 下，production `crop_to_token` 保留的 K/V 前缀与 crop 前前缀及独立切片 oracle 逐张量 bitwise exact；使用相同 token-ID chunk 恢复后的 K/V、logits、attention mask、token ledger 与 role/end 状态也 exact 一致。

不得写：

- clean re-prefill 数值等价；
- C2 v2 通过；
- 32-token continuation 等价；
- 跨模型、dtype、backend 或硬件普适；
- 真实 ASR/TTS/声卡或“用户实际听到”边界正确；
- 生产端到端正确性、时延或质量提升。

C2 v1/v2 必须透明保留为 rejected 描述性证据：两轮均支持 token/state/EOT 正确性，但不同 forward topology 下的 BF16 clean-prefill 数值比较不构成 crop 的可识别 oracle。v3 是 direct crop-integrity addendum，不得包装成 v2 clean-reprefill 结论。

## 三、下一阶段统一论文修订

以 `paper2/thesis_draft.md` 和分章 Markdown 为权威源，按二审意见集中处理：

1. **C2 贡献重构**：把 exact crop-integrity、显式 EOT/role state 与 playback-aware retention 提升为主要技术贡献；C1 timing characterization 降为支持性结果；C3/A2 降为 exploratory/negative extension。
2. **E1/E2 crossed reanalysis**：用 session × dialogue crossed/product bootstrap 生成 versioned `analysis_v2`，不得覆盖 accepted raw/analysis_v1。审核阶段估计的区间仅作核对，最终数值从正式 raw 重算。
3. **事件命名**：27.70/62.38 ms 改称内部 first-token selection/compute readiness，不称 generator deliverability。同步 harness 的 consumer/yield 延迟不作生产 headline。
4. **C-E1 限定**：明确是 implementation-path comparison，不声称 token-equivalent；报告 full output 280/500 same、first token 465/500 same，B@0.92 vs B-never 500/500 same（以正式离线分析为准）。
5. **E3 weighting/sensitivity**：统一点估计与区间 estimand；补 label-weighted/dialogue-weighted/unique-boundary 去重敏感性。现有证据不支持 superiority/equivalence/noninferiority/harm。
6. **播放边界术语**：始终区分 software playback cursor、device-presented audio 与 acoustically heard content；将保留边界称为 TTS-fragment-level software boundary。
7. **novelty search**：完成可复现的 targeted/scoping literature search，并收窄 novelty 语言。
8. **artifact 完成**：LICENSE、release/tag、exact E3 input（已在 `results/e3_exact_rescue/`）、references、declarations、复现说明。
9. **全文同步**：先改权威分章/摘要，再合并 `thesis_draft.md`，最后同步 IEEE 衍生稿与图表。避免局部改稿造成口径分裂。

## 四、只读实验工件

以下均不得覆盖或无条件重跑：

- C2 v3 accepted：`c2crop_82103004_20260903T080512Z`；
- C2 v1/v2 rejected 及全部 pilots/failure sidecars；
- C-E1/E2 accepted：`e1e2c_b8c758b_20260901T173306Z`；
- 固定轨迹 E3：`sci34_f11ccba_20260901_e3`；
- 联合 A1：`sci34_f11ccba_20260901_a1`；
- P1 v2：`sci34_dc52978_20260901_async_prepared_v2`；
- E3 exact rescue：`experiments/sci34_supplement/results/e3_exact_rescue/`；
- `experiments/results/` 中全部旧 JSON。

统计 reanalysis 必须新建 versioned 输出，不修改 raw、manifest、validation、analysis_v1、acceptance 或 seal。

## 五、跨平台复核陷阱

Windows 的 `core.autocrlf` 会把结果文本转成 CRLF，直接在普通 Windows checkout 上跑 byte-level cases/seal 校验会误报。正式复核必须采用以下任一方式：

- `git -c core.autocrlf=false clone ...` 的 LF 保留 checkout；
- `git show HEAD:<path>` 的 Git blob 原字节；
- GPU 返回的原 tarball。

设计侧已在 LF 保留临时 clone 中重跑 formal validator：`ok=true/errors=0/cases=24/events=27/all_exact=true`；seal verify：`files=30/ok=true/seal_sha256=e0997d41…f4a9`。

## 六、关键文档

- 二审意见：`paper2/review/sci_q3_q4_full_review_2026-09-02.md`
- 决策：`docs/decisions.md` D-018～D-023
- C2 v3 plan：`experiments/sci34_supplement/c2_crop_integrity/EXPERIMENT_PLAN.md`
- C2 v3 accepted artifacts：`experiments/sci34_supplement/results/c2_crop_integrity/c2crop_82103004_20260903T080512Z/`
- GPU 执行日志：`experiments/sci34_supplement/results/GPU_RUN_NOTES.md`
- 论文总稿：`paper2/thesis_draft.md`
