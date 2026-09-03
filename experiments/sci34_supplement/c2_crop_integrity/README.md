# C2 protocol v3: crop-integrity addendum

This directory is an independent addendum to `../c2_equivalence/`; it does not alter the v1/v2 protocol or any result artifact.

## Question and claim boundary

C2 v3 asks whether, for the frozen Qwen2-7B snapshot and Transformers/BF16 backend, production `crop_to_token` retains exactly the K/V prefix that existed immediately before crop, and whether production role recovery is bitwise deterministic against an oracle that starts from an independently cloned retained prefix and appends identical token-ID chunks directly through model forwards.

A passing run supports only **crop/truncation integrity plus matched recovery determinism for this frozen model/backend**. It does not prove clean-reprefill numerical equivalence, correctness across models/templates/dtypes/backends, or online ASR/TTS/player correctness.

## Frozen inputs

- Protocol version: 3.
- Cases: exact byte copy of `../c2_equivalence/cases.json`.
- Frozen SHA-256: `acda9afb83ff393d222fc06eb433c1a307dba1eb213fcbbbedfdb860f7500696`.
- Formal grid: 24 records and 27 crop events (one in every case; second crop in cases 08, 16, and 24).
- Model identity/dtype: same accepted Qwen2-7B artifact hash and BF16 identity as C2 v2.
- Prior termination evidence: immutable v2 run `c2eq_5c56b014_20260903T040829Z`; v3 records fixed provenance only and never reads that run at runtime.

## Workflow

Run from repository root with `uv`:

```bash
uv run python -m experiments.sci34_supplement.c2_crop_integrity.smoke
uv run python -m experiments.sci34_supplement.c2_crop_integrity.campaign \
  --run-id "$RUN_ID" --output-dir "$RUN_DIR" --model "$MODEL" --device cuda:0
uv run python -m experiments.sci34_supplement.c2_crop_integrity.run \
  --campaign-dir "$RUN_DIR" --model "$MODEL" --device cuda:0
uv run python -m experiments.sci34_supplement.c2_crop_integrity.validate \
  --campaign-dir "$RUN_DIR"
uv run python -m experiments.sci34_supplement.c2_crop_integrity.analyze \
  --campaign-dir "$RUN_DIR"
# Complete ACCEPTANCE.md from ACCEPTANCE_TEMPLATE.md only after validation review.
uv run python -m experiments.sci34_supplement.c2_crop_integrity.seal \
  --campaign-dir "$RUN_DIR" --create
```

Formal creation and execution require a clean tree, an explicit local model directory, offline environment variables, and empty Hugging Face tokens. Case persistence is atomic JSONL with an append-only attempt ledger; `--resume` accepts only matching manifest, code, model, cases, and record hashes.

Raw JSON records contain token IDs and hashes, lengths, pre/post/oracle per-layer shape/dtype/device/hash manifests and aggregate hashes, production/oracle event ledgers, exact booleans, and errors. Tensor bytes are not dumped.
