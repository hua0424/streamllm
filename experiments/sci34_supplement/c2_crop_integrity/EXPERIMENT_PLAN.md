# C2 v3 crop-integrity addendum — frozen experiment plan

## 0. Status

Protocol version 3 is a new independent addendum. It leaves C2 equivalence v1/v2 and all existing results immutable. Constants and gates are frozen before formal execution.

## 1. Evidence question

For the accepted Qwen2-7B-Instruct snapshot (`artifact_hash=fae2ece10b76512237cf28957f98e7b0d2c609455a173031e3bd16b3dff7c5ab`), Qwen2 architecture, Transformers backend, batch size 1, and `torch.bfloat16`:

1. Does production `StreamLLMInference.crop_to_token` leave every retained K/V tensor element bitwise unchanged?
2. Starting from independently cloned retained-prefix K/V, do production role APIs and direct model forwards over identical token-ID chunks remain bitwise equal after every recovery event?
3. Do token/mask/role/EOT ledgers remain exact?

There is no numeric tolerance and no clean-from-empty re-prefill logit gate.

## 2. Frozen cases and grid

`cases.json` is an exact byte copy of `../c2_equivalence/cases.json`; both must hash to `acda9afb83ff393d222fc06eb433c1a307dba1eb213fcbbbedfdb860f7500696`. Formal execution requires all 24 cases in source order. Every case contributes `crop_1`; cases `c2_08_second_short_cap`, `c2_16_second_medium_eos`, and `c2_24_second_long_max` also contribute `crop_2`, for exactly 27 events.

Termination probes are not rerun. Immutable v2 run `c2eq_5c56b014_20260903T040829Z` is cited as prior evidence by fixed metadata and is not a runtime artifact dependency.

## 3. Production fixture path

For each case, the initial production cache is created through `StreamLLMInference.cache_prompt(..., is_end=True)` and `to_accum_cache`. Frozen assistant text is tokenized once; EOT is forbidden in fixture IDs. `_decode_logits` is temporarily controlled to select each fixture ID. `generate_accumulating` must call `_prefill_ids_p2` exactly once per selected token and each call must contain exactly one token. The production fixture ledger records every call and length transition.

Second-crop cases recover from crop 1, open the next assistant role in the real case topology, and append the second assistant fixture token-by-token through the same controlled generation path before crop 2.

## 4. Crop oracle

Immediately before each crop:

- retain the full token ledger and mask;
- hash every layer's retained-prefix key and value tensor, including shape/dtype/device;
- clone each retained prefix tensor to construct a new `DynamicCache` using `DynamicCache.from_legacy_cache` (or an equivalent clone-only constructor);
- never invoke production `crop_to_token` on the oracle arm.

The production arm then invokes `crop_to_token`. Pre-prefix, post-production, and oracle manifests must have equal per-layer metadata, equal SHA-256 hashes, equal aggregate manifest hashes, and direct `torch.equal` for all tensors. A no-op crop is held to the same proof.

## 5. Recovery topology and comparison

Recovery chunks follow the exact case topology:

- assistant turn retained (including empty-but-real p=0): `reopen_user_role`;
- speculation full invalidation: no assistant close/reopen chunk;
- cases with next user: `prefill_user_text(case.next_user)`, then `open_assistant_role`;
- crop 2: `reopen_user_role` after tokenwise second assistant fixture.

For each production API call, the oracle appends the exact same token-ID chunk in one direct model forward using the same attention-mask extension and position IDs. After each event, production and oracle K/V tensors, logits, masks, and complete token IDs must be bitwise/exact. The original retained-prefix manifest must remain exact in both arms. Production and oracle event ledgers record token IDs/hashes, operation, API, ordinal, and before/after lengths.

## 6. Canonical ledger and role/EOT gates

The runtime independently composes the final canonical token ledger from the retained IDs and frozen recovery chunks. The validator recomputes token hashes, length chains, expected assistant boundary count, and unique EOT count. Duplicate/missing EOT, duplicated ledger chunks, altered chunks, role-boundary mismatch, or final-token mismatch fails the event.

## 7. Raw artifacts and persistence

One case-atomic `records.jsonl` record contains its fixture ledger and one/two crop-event objects. Each crop event includes token IDs/hashes, keep length, three crop manifests, recovery manifests and logits hashes, both event ledgers, exact booleans, and errors. Tensor bytes are never persisted. `attempts.jsonl`, `progress.json`, `summary.json`, immutable manifest, validation, analysis, acceptance, logs, and checksum seal complete the campaign.

Resume is fail-closed on manifest SHA-256, campaign identity, code/model/runtime identity, cases hash, unique case IDs, and per-record JSON content hashes.

## 8. Independent validation

The validator cannot reconstruct tensor bytes without rerunning and makes no such claim. It independently:

- recomputes every JSON content/token/manifest aggregate hash;
- checks exact 24/27 case/event grid and source order;
- verifies cases, protocol, code, model, dtype, backend, offline, and prior-v2 provenance;
- cross-compares every pre/post/oracle and recovery manifest;
- checks every exact boolean and ledger chain;
- verifies no termination probe reran;
- verifies the deterministic wrong-length negative control.

Any formal validation failure blocks analysis and seal.

## 9. Controls and smoke

Pure-CPU fake smoke executes campaign creation, 24 cases/27 events, validation, analysis, acceptance, and seal without a model or network. It requires detection of wrong keep length, altered layer hash, duplicate EOT/ledger, and missing crop event. Positive-control metadata states that a wrong crop length is detected. Formal manifest creation and formal execution each rerun a deterministic disposable wrong-length manifest control without mutating evidence caches or paths.

## 10. Permitted conclusion

A passing formal run directly proves crop/truncation integrity and matched recovery determinism for the frozen Qwen2-7B snapshot/backend. It does not prove clean-reprefill numerical equivalence; v2 clean-prefill data remains descriptive rejected evidence for this gate. It does not establish cross-model correctness or online ASR/TTS/player correctness.
