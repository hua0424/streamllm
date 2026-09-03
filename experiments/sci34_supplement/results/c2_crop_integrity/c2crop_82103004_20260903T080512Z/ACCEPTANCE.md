# C2 v3 crop-integrity acceptance record

Run ID: `c2crop_82103004_20260903T080512Z`  
Code commit: `82103004637dce8f98688f4a685d33ebee363a3b`  
Campaign manifest SHA-256: `d8c3db4d609234a072064162a5caa443e25171b2311d84afa48b7b6a4f1d4bc2`  
Records SHA-256: `f775ba238f17439b2b1831f31cbb97eb8ade87ddc7e2517c8eba427ee8b21725`  
Validation SHA-256: `d0e9809bf34f9b45cc270adcbc5af5a113fcc0e006d66de2c3297b1e208f3f62`  
Analysis SHA-256: `153eb65fd321fa02f432d6c1a6836249a8439cc493010d15d3e9abda41e0c412`

## Independent review checklist

- [x] Formal manifest: clean `paper2` tree, strict offline, local accepted model, empty HF tokens.（dirty=false；`HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`；`--model /root/autodl-tmp/dataA/models/Qwen2-7B-Instruct`；HF_TOKEN 空）
- [x] Model: accepted Qwen2-7B artifact hash, Qwen2ForCausalLM/Qwen2, `torch.bfloat16`.（content identity `209f3a9c…`；artifact hash `fae2ece1…` 与 D-017 一致）
- [x] Cases: exact copied hash `acda9afb83ff393d222fc06eb433c1a307dba1eb213fcbbbedfdb860f7500696`.（与 `c2_equivalence/cases.json` byte-copy 一致，preflight 校验通过）
- [x] Grid: exactly 24 ordered case records and 27 ordered crop events.（24/24 records、27/27 events，独立重算一致；summary `protocol_version=3, status=PASS`）
- [x] Prior evidence: v2 run `c2eq_5c56b014_20260903T040829Z` recorded only as immutable provenance; no local dependency; zero termination-probe reruns.（PRIOR_V2_RUN_ID 仅作 provenance；v3 不读取其工件；guard 显示 v1/v2 归档零改动）
- [x] Assistant fixtures: all non-EOT and appended via `generate_accumulating`, exactly one `_prefill_ids_p2` forward per token.（fixture token 逐 token 走 production KV append，事件账本 API 记录在案）
- [x] Every crop: pre-prefix, post-production, and independent clone-oracle per-layer manifests/hash aggregates match exactly; direct runtime `torch.equal` gates passed.（27/27 events 的 `pre_prefix_equals_oracle`/`post_equals_oracle`/`post_equals_pre_prefix`/`retained_prefix_hash_exact`/`keep_length_exact` 及逐层 manifest 全 exact，独立重算 True）
- [x] No-op crops receive the same exact proof.（3 个 no-op crop（reply_tail 类）同获全套 exact 证明）
- [x] Every recovery event: production API and direct-forward oracle use identical token-ID chunks; K/V, logits, mask, token ledger, and retained prefix are exact.（全部 recovery_check `kv_exact/logits_exact/masks_exact/production_state_exact/passed` 均 True；production API 为 `StreamLLMInference.reopen_user_role/prefill_user_text/open_assistant_role`，oracle 为 direct forward，token chunk 逐块一致）
- [x] Canonical final ledger, role boundaries, and unique EOT gates all pass.（final token hash/IDs、role 边界、unique EOT 门全过——validator 零错误复核）
- [x] Deterministic wrong-length negative control passes and states wrong crop length would be detected.（27/27 events `negative_control_detected=True`）
- [x] `validation.json`: `ok=true`, `acceptance_eligible=true`, with no errors.（`{"errors": [], "ok": true}`）
- [x] `analysis_v1.json`: accepted, descriptive only, no clean-reprefill gate or overclaim.（`acceptance.passed=true, failed_cases=0, failed_crop_events=0, all_exact_gates=true`；`rejected_descriptive_evidence` 明确保留 v2 rejected 证据为描述性）
- [x] Complete logs/attempts/progress/summary retained; seal verifies.（全套工件在 run 目录；seal --create/--verify 见下）

补充（本轮执行说明）：前序 pilot `c2crop_pilot_b2c6f22b_20260903T064135Z` 曾暴露 `speculation_full_invalidation` 路径 `prefill_user_text` 残留 `CROPPED` 终因（v3 oracle 期望 NONE），设计侧以 commit `8210300` 在生产侧修复（`prefill_user_text` 现重置 `generation_end_reason=NONE`，含 orchestrator 断言与 kvcrop 测试覆盖）；修复后新 pilot `c2crop_pilot_82103004_20260903T080321Z` 8 records/9 events 全过（validate `ok=true`），随后 formal 一次通过。两次 pilot 均独立归档，未进入 formal。

## Claim boundary

Accepted evidence, if all checks pass, directly proves crop/truncation integrity and matched recovery determinism only for the frozen Qwen2-7B snapshot/dtype/Transformers backend and 24-case protocol v3 grid. It does not prove clean-reprefill numerical equivalence, cross-model correctness, or online ASR/TTS/player correctness. Existing v2 clean-prefill comparisons remain descriptive rejected evidence for the v3 exact gate.

<!-- Replace only after completing every checklist item. The seal requires an exact standalone accepted line. -->
Status: accepted
