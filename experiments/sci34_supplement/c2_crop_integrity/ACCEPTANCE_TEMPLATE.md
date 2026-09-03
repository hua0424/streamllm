# C2 v3 crop-integrity acceptance record

Run ID: `<run-id>`  
Code commit: `<full-commit>`  
Campaign manifest SHA-256: `<sha256>`  
Records SHA-256: `<sha256>`  
Validation SHA-256: `<sha256>`  
Analysis SHA-256: `<sha256>`

## Independent review checklist

- [ ] Formal manifest: clean `paper2` tree, strict offline, local accepted model, empty HF tokens.
- [ ] Model: accepted Qwen2-7B artifact hash, Qwen2ForCausalLM/Qwen2, `torch.bfloat16`.
- [ ] Cases: exact copied hash `acda9afb83ff393d222fc06eb433c1a307dba1eb213fcbbbedfdb860f7500696`.
- [ ] Grid: exactly 24 ordered case records and 27 ordered crop events.
- [ ] Prior evidence: v2 run `c2eq_5c56b014_20260903T040829Z` recorded only as immutable provenance; no local dependency; zero termination-probe reruns.
- [ ] Assistant fixtures: all non-EOT and appended via `generate_accumulating`, exactly one `_prefill_ids_p2` forward per token.
- [ ] Every crop: pre-prefix, post-production, and independent clone-oracle per-layer manifests/hash aggregates match exactly; direct runtime `torch.equal` gates passed.
- [ ] No-op crops receive the same exact proof.
- [ ] Every recovery event: production API and direct-forward oracle use identical token-ID chunks; K/V, logits, mask, token ledger, and retained prefix are exact.
- [ ] Canonical final ledger, role boundaries, and unique EOT gates all pass.
- [ ] Deterministic wrong-length negative control passes and states wrong crop length would be detected.
- [ ] `validation.json`: `ok=true`, `acceptance_eligible=true`, with no errors.
- [ ] `analysis_v1.json`: accepted, descriptive only, no clean-reprefill gate or overclaim.
- [ ] Complete logs/attempts/progress/summary retained; seal verifies.

## Claim boundary

Accepted evidence, if all checks pass, directly proves crop/truncation integrity and matched recovery determinism only for the frozen Qwen2-7B snapshot/dtype/Transformers backend and 24-case protocol v3 grid. It does not prove clean-reprefill numerical equivalence, cross-model correctness, or online ASR/TTS/player correctness. Existing v2 clean-prefill comparisons remain descriptive rejected evidence for the v3 exact gate.

<!-- Replace only after completing every checklist item. The seal requires an exact standalone accepted line. -->
Status: pending
