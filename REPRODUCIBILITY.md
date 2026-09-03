# Paper 2 artifact and reproducibility guide

This file is the authoritative repository-level index for Paper 2 campaign status and artifacts. It records what is in Git; it does not promote rejected runs or expand the claims authorized by the campaign acceptance records. Paths are relative to the repository root. Paper chapters and accepted result files are immutable inputs to this guide.

## 1. Status matrix

| Campaign | Status | Formal run ID | Experiment code commit | Result commit | Result directory | Manifest / key integrity artifacts | Validation, analysis, acceptance, seal | Entrypoints and GPU requirement | Allowed claim boundary |
|---|---|---|---|---|---|---|---|---|---|
| Confirmatory E1/E2 | **accepted** (D-017); post-run crossed reanalysis v2 generated | `e1e2c_b8c758b_20260901T173306Z` | Formal GPU code `b8c758bd8e97e519f041ac047d4f6c5f85697bc7`; `analyze_v2.py` is currently uncommitted, so its code commit is **not recorded** | Formal results `62508dc79a8843e5dbe58677750f2c22010a1e44`; `analysis_v2` result commit **not recorded** | [`experiments/sci34_supplement/results/e1e2_confirmatory/e1e2c_b8c758b_20260901T173306Z/`](experiments/sci34_supplement/results/e1e2_confirmatory/e1e2c_b8c758b_20260901T173306Z/) | Manifest SHA-256 `2f4bd76e759945e62a5536b6b4399ad129c47a0b76c967bb653e22ffcf0f4ed8`; content hash `b307e054f5c699c671c9bd6a59270e15ddd53751a6e32f6b0ce403ebfd9bf146`; campaign identity `897b24fb238157c6b108748682e0775afe19361715cc515170fded4e882075a1`; historical `checksums.sha256` covers 72 v1 files and intentionally does not cover later v2 files | `validation.json` `ok=true`; accepted `analysis_v1.json`; post-run `analysis_v2.json` SHA-256 `9bce6db5d93c1faccb4069b295df32ce5ee0778899b31ac6be17526bfb644456`; `ACCEPTANCE.md`; design-side acceptance [`paper2/e1e2_confirmatory_acceptance_2026-09-02.md`](paper2/e1e2_confirmatory_acceptance_2026-09-02.md); no separate seal | Builder/cache/campaign/run/analyze/validate modules under `e1e2_confirmatory`; full GPU commands in its [`GPU_HANDOFF.md`](experiments/sci34_supplement/e1e2_confirmatory/GPU_HANDOFF.md). Formal generation requires local Qwen2-7B-Instruct plus TEN_Turn_Detection and CUDA (run used 2×RTX 3090); v1/v2 analysis and validation do not require GPU. | Controlled synchronous pre-segmented text only. V2 renames the raw alias `arrival_to_first_token_ready_ns` as candidate-selection/compute readiness, not generator/production deliverability. `TTFT_eff` is an oracle latency lower bound/speculation-benefit upper bound. No real ASR, online TEN cost, TTS/player/sound-card, mouth-to-ear, production end-to-end, or globally optimal-threshold claim. |
| Fixed-trajectory E3 | **accepted**; post-review weighting/dedup reanalysis generated | `sci34_f11ccba_20260901_e3` | Formal GPU code `f11ccbadac807e083fc46977ab45288c336361d0`; `analyze_e3_v2.py` commit **not recorded until this revision is committed** | Formal results `728ca369a9bab2f0a2f9d4d364334b8f76002390`; post-review analysis result commit **not recorded until this revision is committed** | [`experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/`](experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3/) | Manifest SHA-256 `7690f1003109a37c6f216b674ff6df2b71a4bfac98f6992c1eb23b37f98967a4`; config hash `3c60d57594b91ee3fbb96239af0fdf9ee123b20fffa917c86b40c916dd44c2a1`; records SHA-256 `9406ea42d1112b5ad97c94e7e27856946acc79c14dd4ebe5762d0b702fe458e9`; exact input preserved in the [E3 rescue bundle](experiments/sci34_supplement/results/e3_exact_rescue/) | 100 trajectories / 800 condition records; historical `analysis_metric_specific_eligibility_v1.json`; post-review `analysis_weighting_dedup_v2.json` SHA-256 `5776db23e534767c6ca266872967e228962eef59924cec3a1e3dd5cdbcd30366`; accepted by [`paper2/supplement_acceptance_2026-09-01.md`](paper2/supplement_acceptance_2026-09-01.md); no campaign seal | `e3_fixed_trajectory`, `analyze_e3`, `analyze_e3_v2`; formal run requires local Qwen2-7B-Instruct and CUDA (run used RTX 3090); reanalysis is CPU-only. | Controlled fixed shared trajectories. V2 aligns point estimates and cluster intervals for label/dialogue weighting and adds target-specific unique-boundary sensitivity. The evidence does **not** establish superiority, equivalence, non-inferiority, harm, human semantic validity, or acoustic playback truth. |
| E3 judge v2 | **completed and used in accepted E3 analysis** | `sci34_f11ccba_20260901_judge_v2` | `ca627c7e525e96635b3ff4b168fe63ece1c33b96` | `728ca369a9bab2f0a2f9d4d364334b8f76002390` | [`experiments/sci34_supplement/results/judge/sci34_f11ccba_20260901_judge_v2/`](experiments/sci34_supplement/results/judge/sci34_f11ccba_20260901_judge_v2/) | Manifest SHA-256 `f103963a300dc8f668161d1c1147d9c8e0d0f488a8d66ff76f59dee55cbc91c4`; source E3 records SHA-256 `9406ea42d1112b5ad97c94e7e27856946acc79c14dd4ebe5762d0b702fe458e9`; judge model identity `033f608001d19e152db47f66053656f7d59e73870aa6578baa686c25ef5fb627` | 1,600 records; `parse_failures=0`, `retried=0`; incorporated into E3 analysis and acceptance; no separate seal. The earlier 919-record judge run is an incomplete audit artifact, not accepted output. | `e3_judge`; requires local Mistral-7B-Instruct-v0.3 and CUDA (run used RTX 3090). Reading existing summary needs no GPU. | Single-model `specific-reference-v3` proxy only; not human double annotation and not an independent human validation. v3 added output-format enforcement and one bounded retry without changing the semantic criterion. |
| Joint A1 | **accepted** | `sci34_f11ccba_20260901_a1` | Manifest records `ca627c7e525e96635b3ff4b168fe63ece1c33b96`; A1 implementation was unchanged from `f11ccba` | `728ca369a9bab2f0a2f9d4d364334b8f76002390` | [`experiments/sci34_supplement/results/a1/sci34_f11ccba_20260901_a1/`](experiments/sci34_supplement/results/a1/sci34_f11ccba_20260901_a1/) | Manifest SHA-256 `39073e83c727d0e4522d714209aee4f2f5ce6fe3754b39362b0477f208039012`; config hash `c7980e7adc17c6bf4f835beec298d87bf2fe72fe14ca4991cf98968f9ae7454b`; records SHA-256 `c525c3a94ab75899c7d5004bb83ecbd148d9bddeafee3f7c8cdac1267777fbb5` | `analysis.json`; accepted by [`paper2/supplement_acceptance_2026-09-01.md`](paper2/supplement_acceptance_2026-09-01.md); no campaign validator or seal | `a1_joint_latency`, `analyze_latency --kind a1`; formal timings require local Qwen2-7B-Instruct and CUDA (run used RTX 3090); reanalysis is CPU-only. | Synchronized model-side joint crop+role microbenchmark. Not full barge-in latency; excludes playback stop, timeline lookup, service transport, concurrent ASR/LLM/TTS, and acoustic stop. |
| P1 v2 prepared state | **accepted** (D-015) | `sci34_dc52978_20260901_async_prepared_v2` | `dc529788e86ecd3e2e4203ba16b1076d6b231ec1` | `ee1dcc71c18d0c6886161a8db692708a61c0c0ae` | [`experiments/sci34_supplement/results/async_bargein/sci34_dc52978_20260901_async_prepared_v2/`](experiments/sci34_supplement/results/async_bargein/sci34_dc52978_20260901_async_prepared_v2/) | Manifest SHA-256 `0358af6cb3a7796d35091322a3075bab322679d65fe4a8a563b278159e7deef9`; records `2dc68896dc52ce2c777b1a6375f1a5c3090f9baffd8f07a6ac1ed0f1769a3b67`; analysis `b9705d58f36909604e3e0df94d2190b3a5050c6a62d35fee1c29987fff4db20a`; tarball hash recorded as `4c6188249f1226e5692a85468cf1e9c3b05e648494a5ce9a6e5a475b264c0bc8` | `analysis.json`; accepted by [`paper2/p1_v2_acceptance_2026-09-01.md`](paper2/p1_v2_acceptance_2026-09-01.md); no campaign validator or seal | `async_bargein`, `analyze_latency --kind async`; formal path requires local Qwen2-7B-Instruct and CUDA but no sound card (run used RTX 3090); reanalysis is CPU-only. | Headless wall-clock-paced software playback control path only. Not sound-card/speaker stopping, user-heard last sample, online TTS cancellation, real concurrent pipeline, or production end-to-end barge-in latency. P1 v1 remains protocol-invalid. |
| C2 v1 clean-prefill equivalence | **rejected** | `c2eq_563dd22a_20260903T013547Z` | `563dd22a55544e042826290f9dde736fa7fef458` | `1a47ac1bb8a377a9cda8f3679e86ece63fc66488` | [`experiments/sci34_supplement/results/c2_equivalence/c2eq_563dd22a_20260903T013547Z/`](experiments/sci34_supplement/results/c2_equivalence/c2eq_563dd22a_20260903T013547Z/) | Manifest `f4960a20364de8c4e78bdb51c3accdd02576b5ce61e2375c36bc5831054c9670`; content hash `e0dc511acbd5e77a34affe0fddae5cb1ce2f46dd1762921e575120750192e054`; campaign identity `1f07a2e91bd97e6c3ff5f497d0f017a27321c712c1d27fd2285b82861eba8a36`; `checksums_return.sha256` | `validation.json` `ok=false`; analyzer refused output; `ACCEPTANCE.md` says rejected; no seal | C2 `campaign/run/validate/analyze/seal`; formal execution requires exact local Qwen2-7B snapshot, BF16/sdpa, CUDA (run used RTX 3090); validation/attempted analysis are CPU-only. | Not accepted correctness evidence. Retained only as descriptive failed evidence: structural/token observations passed, but frozen logit, continuation, and termination gates failed. Do not cite it as numerical equivalence. |
| C2 v2 clean-prefill equivalence | **rejected** | `c2eq_5c56b014_20260903T040829Z` | `5c56b0144c822e4a05ba4eeec167684363e8828e` | `8d9b863791630b462936be58aa7503107a53a054` | [`experiments/sci34_supplement/results/c2_equivalence/c2eq_5c56b014_20260903T040829Z/`](experiments/sci34_supplement/results/c2_equivalence/c2eq_5c56b014_20260903T040829Z/) | Manifest `7a636016c7959566f47303fb47c3da7c68775bda08b12abc6df59ed192ef178d`; content hash `b7d8a4747a348f14f016d4dff8f3322dedd4677989df537fda415d65094ddefe`; campaign identity `165c91f98c734a2cdbaa63ce534047c6bfb40db034ca87c5ee5c8b9b7f8176dd`; `checksums_return.sha256` | `validation.json` `ok=false`; analyzer refused output; `ACCEPTANCE.md` says rejected; no seal | Same C2 entrypoints; formal execution requires exact local Qwen2-7B snapshot, BF16/sdpa, CUDA (run used RTX 3090); validation/attempted analysis are CPU-only. | Not accepted correctness evidence. 42/45 relative-noise gates passed, but 3/45 failed; no campaign-level correctness claim. May be described only as rejected diagnostic evidence within its frozen environment. |
| C2 v3 crop integrity | **accepted and sealed** (D-023) | `c2crop_82103004_20260903T080512Z` | `82103004637dce8f98688f4a685d33ebee363a3b` | `7d506240a7592321425e40aae92e90795e4693fb` | [`experiments/sci34_supplement/results/c2_crop_integrity/c2crop_82103004_20260903T080512Z/`](experiments/sci34_supplement/results/c2_crop_integrity/c2crop_82103004_20260903T080512Z/) | Manifest `d8c3db4d609234a072064162a5caa443e25171b2311d84afa48b7b6a4f1d4bc2`; cases `acda9afb83ff393d222fc06eb433c1a307dba1eb213fcbbbedfdb860f7500696`; records `f775ba238f17439b2b1831f31cbb97eb8ade87ddc7e2517c8eba427ee8b21725`; `checksums.sha256`; seal hash recorded as `e0997d41793f510fc1120a7c3f08c420097813cc627f08d47716e76b4489f4a9` | `validation.json` `ok=true`; `analysis_v1.json` accepted/descriptive-only; `ACCEPTANCE.md` `Status: accepted`; experiment-machine `seal --verify` passed (Windows CRLF worktree verification needs LF/Git-blob bytes) | `c2_crop_integrity` `campaign/run/validate/analyze/seal`; formal execution requires exact local Qwen2-7B snapshot, BF16/sdpa, CUDA (run used RTX 3090); validation, analysis, and seal verification are CPU-only. | Proves crop/truncation integrity and matched-recovery determinism only for the frozen Qwen2-7B snapshot/dtype/Transformers backend and 24-case/27-event v3 grid. It does not prove clean-reprefill numerical equivalence, cross-model/backend correctness, or online ASR/TTS/player correctness. |

### Superseded and non-authoritative items

- Legacy Paper 2 JSON files under `experiments/results/` remain immutable exploratory records. Confirmatory E1/E2 above corrects the old timing-semantics interpretation; do not mix their absolute timings.
- P1 v1 `sci34_f11ccba_20260901_async` is preserved but its joint numeric path is protocol-invalid.
- Judge v1 `sci34_f11ccba_20260901_judge` stopped after 919/1600 format records and is audit-only.
- C2 v1 and v2 are rejected and unsealed. C2 v3 does not retroactively accept or rewrite them.

## 2. Inputs, model identities, and execution environment

### Shared E3 input

The exact accepted E3 input is tracked in [`experiments/sci34_supplement/results/e3_exact_rescue/p2_turns.json`](experiments/sci34_supplement/results/e3_exact_rescue/p2_turns.json). Its canonical experiment-machine/Git-blob SHA-256 is `a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c`. It was deterministically derived from MultiWOZ 2.1 `data.json` with [`experiments/scripts/prepare_multiwoz_data.py`](experiments/scripts/prepare_multiwoz_data.py), seed `42`, `--max-dialogues 100`, and the builder default `--min-user-turns 3`. The raw source is not redistributed. See the [rescue README](experiments/sci34_supplement/results/e3_exact_rescue/README.md).

### Accepted 7B model snapshots

Model weights are not redistributed by this repository. Campaign manifests record local paths and content identities:

- Qwen2-7B-Instruct strong content identity: `209f3a9cbccde56fb9ed39fca06a86b11c1aa0ebf721d897baab708d6cab2133`. Earlier E3/A1/P1 manifests also record inventory identity `7feb5a62bd0a65d0741eac46fc0ce2a0328aa5e8dec23fec92079346857347bc`. C2 records accepted model artifact hash `fae2ece10b76512237cf28957f98e7b0d2c609455a173031e3bd16b3dff7c5ab`.
- Mistral-7B-Instruct-v0.3 E3 judge inventory identity: `033f608001d19e152db47f66053656f7d59e73870aa6578baa686c25ef5fb627`.
- TEN_Turn_Detection strong identity for confirmatory E1/E2: `c3787bb7b25d9ba37007332be55bb236006eb44d4e97083608bc6cb0f888f722`.

The accepted formal campaigns used Python 3.10.18, PyTorch 2.8.0+cu128, Transformers 4.57.1, CUDA 12.8, cuDNN 9.10.2, and RTX 3090 GPUs. Exact host snapshots are stored where available in each result directory. Absolute wall-clock values must not be pooled across campaigns.

## 3. Reproducing environments safely

Always start from the exact experiment code commit in the matrix, not current `paper2`, when reproducing a formal run:

```bash
git clone <AUTHOR CONFIRM PUBLIC URL>
cd streamllm_p2
git checkout <experiment-code-commit>
uv venv --python 3.10
uv sync --frozen
```

`uv sync --frozen` prevents `uv` from rewriting `uv.lock`, but it does **not** make the current branch equivalent to a historical campaign. The lock changed between the E3/A1 run (`uv.lock` SHA-256 `8d550698a8003a55d7b595c4ce576172becdcf4215830b7cb91e010c32eab02f`) and P1 v2/E1-E2/C2 (`7b76c69de3b04f10d270215206b892cebaf372fe2107a4a76cb6e12436cc2fd1`). Checkout the listed code commit first and verify its lock hash. Do not run `uv lock`, un-frozen `uv sync`, bare `pip`, or bare `python` for formal reproduction.

Formal GPU reruns additionally require local third-party model assets, strict offline mode after asset preparation, an empty token environment, and an idle compatible CUDA GPU. Use the campaign-specific handoff/runbook rather than guessing flags. The current repository status is **no unconditional GPU rerun pending**; all accepted campaigns are archived, C2 v1/v2 are intentionally rejected, and C2 v3 is accepted/sealed.

## 4. Analysis-only verification (no model loading, no GPU)

Run these commands from the **current Paper 2 revision/analysis release checkout**, not from an older GPU experiment commit: the current release contains the post-run analyzers while each manifest continues to bind its GPU records to the historical code commit shown in the matrix. Commands write only to a temporary directory or run immutable self-tests; they never overwrite accepted artifacts.

```bash
set -euo pipefail
TMP="$(mktemp -d)"

E12=experiments/sci34_supplement/results/e1e2_confirmatory/e1e2c_b8c758b_20260901T173306Z
uv run --frozen python -m experiments.sci34_supplement.e1e2_confirmatory.validate \
  --campaign-dir "$E12" --out "$TMP/e1e2_validation.json"
uv run --frozen python -m experiments.sci34_supplement.e1e2_confirmatory.analyze \
  --campaign-dir "$E12" --out "$TMP/e1e2_analysis_v1.json" \
  --bootstrap-repeats 10000 --bootstrap-seed 20260901 \
  --expected-sessions 5 --expected-dialogues 100

diff -u "$E12/validation.json" "$TMP/e1e2_validation.json"
diff -u "$E12/analysis_v1.json" "$TMP/e1e2_analysis_v1.json"

# V2 is immutable at its formal default path. Its built-in self-test is portable;
# verify the already-generated artifact hash without rewriting it.
uv run --frozen python -m experiments.sci34_supplement.e1e2_confirmatory.analyze_v2 --self-test
printf '%s  %s\n' \
  9bce6db5d93c1faccb4069b295df32ce5ee0778899b31ac6be17526bfb644456 \
  "$E12/analysis_v2.json" | sha256sum -c -

E3=experiments/sci34_supplement/results/e3/sci34_f11ccba_20260901_e3
JUDGE=experiments/sci34_supplement/results/judge/sci34_f11ccba_20260901_judge_v2/judge_records.jsonl
uv run --frozen python -m experiments.sci34_supplement.analyze_e3 \
  --e3-run-dir "$E3" --judge-records "$JUDGE" \
  --out "$TMP/e3_analysis.json" --bootstrap-repeats 10000 --seed 20260831
diff -u "$E3/analysis_metric_specific_eligibility_v1.json" "$TMP/e3_analysis.json"

# Post-review v2 is immutable at its formal default path; run self-tests and verify hash.
uv run --frozen python -m experiments.sci34_supplement.analyze_e3_v2 --self-test
printf '%s  %s\n' \
  5776db23e534767c6ca266872967e228962eef59924cec3a1e3dd5cdbcd30366 \
  "$E3/analysis_weighting_dedup_v2.json" | sha256sum -c -

A1=experiments/sci34_supplement/results/a1/sci34_f11ccba_20260901_a1
P1=experiments/sci34_supplement/results/async_bargein/sci34_dc52978_20260901_async_prepared_v2
uv run --frozen python -m experiments.sci34_supplement.analyze_latency \
  --run-dir "$A1" --kind a1 --out "$TMP/a1_analysis.json"
uv run --frozen python -m experiments.sci34_supplement.analyze_latency \
  --run-dir "$P1" --kind async --out "$TMP/p1_analysis.json"
diff -u "$A1/analysis.json" "$TMP/a1_analysis.json"
diff -u "$P1/analysis.json" "$TMP/p1_analysis.json"

C2V1=experiments/sci34_supplement/results/c2_equivalence/c2eq_563dd22a_20260903T013547Z
C2V2=experiments/sci34_supplement/results/c2_equivalence/c2eq_5c56b014_20260903T040829Z
# Both commands are expected to exit non-zero because the campaigns are rejected.
uv run --frozen python -m experiments.sci34_supplement.c2_equivalence.validate \
  --campaign-dir "$C2V1" --out "$TMP/c2_v1_validation.json" || test $? -ne 0
uv run --frozen python -m experiments.sci34_supplement.c2_equivalence.validate \
  --campaign-dir "$C2V2" --out "$TMP/c2_v2_validation.json" || test $? -ne 0

C2V3=experiments/sci34_supplement/results/c2_crop_integrity/c2crop_82103004_20260903T080512Z
uv run --frozen python -m experiments.sci34_supplement.c2_crop_integrity.validate \
  --campaign-dir "$C2V3" --out "$TMP/c2_v3_validation.json"
uv run --frozen python -m experiments.sci34_supplement.c2_crop_integrity.analyze \
  --campaign-dir "$C2V3" --out "$TMP/c2_v3_analysis_v1.json"
diff -u "$C2V3/validation.json" "$TMP/c2_v3_validation.json"
diff -u "$C2V3/analysis_v1.json" "$TMP/c2_v3_analysis_v1.json"
# The original seal was verified on the LF Linux experiment machine. On a
# checkout that preserves LF bytes, the following should also pass:
# uv run --frozen python -m experiments.sci34_supplement.c2_crop_integrity.seal \
#   --campaign-dir "$C2V3" --verify
```

The analyzers are intentionally immutable by default. Always provide a fresh `--out` outside accepted result directories. C2 v1/v2 analyzers fail closed and should not be forced to emit an accepted-looking analysis.

For repository hashes on Windows, `core.autocrlf=true` can change worktree bytes. The recorded campaign hashes use the Linux experiment-machine/Git-blob byte stream. To verify such a file portably, hash the blob, for example:

```bash
git show HEAD:experiments/sci34_supplement/results/e3_exact_rescue/p2_turns.json | sha256sum
```

## 5. Remaining artifact and submission blockers

- Repository code/data license: **not confirmed**. No root `LICENSE` is present and none is created here. The stale README claim that the repository is MIT-licensed has been removed. The legal rights holder(s), license choice, and third-party redistribution review require author/institution confirmation.
- Public archival URL/DOI and immutable release/tag: **not recorded**. Author must supply them before a public data/code availability statement is final.
- E1/E2 crossed session×dialogue `analysis_v2` and E3 weighting/dedup `analysis_weighting_dedup_v2` are generated and hash-pinned, but their analyzers and result artifacts are currently uncommitted/ignored, so immutable Git commit provenance is **not recorded** until the author intentionally stages and commits them. Historical v1 analyses remain archived; the post-review analyses supersede only the stated uncertainty/weighting methods and do not alter raw data.
- Ethics/consent, funding, conflicts of interest, and CRediT roles require author confirmation; see [`paper2/declarations.md`](paper2/declarations.md).
- Third-party dataset/model licenses and download terms remain authoritative. Raw MultiWOZ and model weights are not redistributed by this repository.
- `pyproject.toml` now points to the repository's actual `README.MD` casing and has a factual project description. `uv lock --check` confirms that these metadata-only changes require no lock update.

## 6. Packaging note

The repository contains accepted raw records, manifests, analyses, logs, and available snapshots. Some experiment-machine tarballs and their sidecar files are referenced by acceptance records but are not tracked in Git. Their hashes are retained above only where the repository records them. A missing result-archive commit or file is reported as **not recorded**, never inferred.
