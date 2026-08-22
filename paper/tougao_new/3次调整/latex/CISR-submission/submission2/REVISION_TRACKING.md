# CISR Revision Tracking

## Paper information

| Field | Value |
|---|---|
| Paper | Latency Optimization of Cascaded Voice Dialogue Systems with a Pipeline-Parallel Streaming Architecture |
| Revision date | 2026-08-22 |
| Target | CISR conference submission |
| Raw experiment lock | Existing checkpoint/CSV/JSON/JSONL artifacts; no model experiment rerun in the final correction |
| Derived analysis | `experiments/results/revision/minimal_cpu_reanalysis` |
| Revised manuscript | `main.tex` / `main.pdf` |
| Reviewer response | `response_to_reviewers.tex` / `response_to_reviewers.pdf` |

## Revision tracking table

| ID | Concern | Final resolution | Evidence | Status |
|---|---|---|---|---|
| R1 | Human recordings and acoustic variation | Added LibriSpeech/AISHELL-1 concatenated human-read sets, noise/speed/endpoints, WER/CER, and babble failure boundary; no natural-dialogue or quality-noninferiority claim | Table VI; `r2_real_speech` | RESOLVED with boundary |
| R2 | First audible output | Added one-clock server-side first-playable-PCM timeline and six-component closure; now explicitly a B-first-sentence versus A-capped-full-response policy comparison, excluding client playback | Table VIII; `r7_ttfa_unified` | RESOLVED as policy-level proxy |
| R3 | Variability, P95/P99, repeats, contention | Added std/tails, repeated CV, and dialogue-cluster inference on the locked 498-turn valid set; contention not run | Tables III–V; `minimal_cpu_reanalysis` | RESOLVED except deliberate contention limitation |
| R4 | Stronger streaming baseline | Added configured project-internal LA-2-style comparator with matched weights/engine/segmenter/hardware/subset; trigger policy is not matched and no method-family superiority is claimed | Table VII; `r3_baseline_la`; `la_cluster_inference.csv` | RESOLVED with comparator boundary |
| R5 | Corrections, tokenizer, KV cache, downstream meaning | Added append-only contract, 224 internal changes, 25/50 seams, and exploratory response analysis; no quality, task, or usability equivalence | State-analysis subsection; `r4_commit`; `r5_semantic` | RESOLVED with exploratory limitation |
| R6 | ASR trigger implementation mismatch | Documented actual historical policy: cumulative admitted duration is not decremented, so the 2.0-s startup gate latches; no corrected-current-queue performance is claimed | Methods; `src/asr/faster_whisper_streamer.py`; Git history | RESOLVED by accurate historical-policy reporting |
| R7 | Candidate-run exclusions | Run-log review confirmed seven executions were contaminated by concurrent external programs; they remain in the audit ledger but are excluded from the locked 498-turn/99-dialogue valid set | Table IV; `sample_exclusions.csv`; `sample_flow.csv` | RESOLVED by decontaminated analysis set |
| R8 | Dependent accumulated turns | Joined 1,133 sample IDs to `(dataset,dialog_id)` and used dialogue-cluster bootstrap/Wilcoxon; reports turn and dialogue counts | Statistical protocol; `cluster_summary.csv` | RESOLVED |
| R9 | TTFA policy/language/resource ambiguity | Reports 204.10/17.74 mean TTS characters, 0/50 vs 25/50 greetings, 41/50 vs 43/50 caps, 19/25 English-input B strings without Latin letters, and ASR/TTS GPU0 co-residency | Table VIII; `ttfa_policy_descriptives.csv`; platform record | RESOLVED descriptively |

## Final analysis commitments

```yaml
- commitment: report historical latched trigger, not intended corrected trigger
  status: fulfilled
- commitment: do not infer corrected-trigger performance
  status: fulfilled
- commitment: distinguish physical-speech-end TTFT, legacy post-feed residual TTFT, and server-side first-playable PCM
  status: fulfilled
- commitment: document 505 candidates, seven run-log-confirmed externally contaminated executions, and the locked 498-turn valid set
  status: fulfilled
- commitment: retain contaminated records in the audit ledger but exclude them from system inference
  status: fulfilled
- commitment: use source dialogue as the primary synthetic inference cluster
  status: fulfilled
- commitment: treat incremental prefill as an architectural component, not the dominant proven source
  status: fulfilled
- commitment: call LA a configured operating point and disclose trigger mismatch
  status: fulfilled
- commitment: describe semantic evidence as exploratory and not quality/usability equivalence
  status: fulfilled
- commitment: disclose TTFA response policy, text length, cap, language, playback, and GPU boundaries
  status: fulfilled
- commitment: run GPU contention
  status: not_fulfilled
  rationale: retained as an explicit deployment limitation; no long model experiment was run
```

## Integrity constraints

- Tables III–V and Fig.6 use the original platform; repeated measurements and Tables VI–VIII use the second platform.
- No cross-platform scaling or mixing is used within a comparison.
- All streaming results describe the historical latched startup-duration implementation; no corrected current-queue trigger result exists.
- Legacy TTFT is post-feed residual TTFT, not a separately detected physical endpoint.
- Seven candidate executions confirmed as concurrent-external-program contamination are excluded; all reported ablation inference uses the locked 498-turn valid set.
- Cluster inference uses `(dataset, dialog_id)` and preserves all accumulated turns when a dialogue is sampled.
- Table VIII uses only the R7 unified timeline; `Language` means input language.
- System B first sentence versus System A capped full response is stated wherever the TTFA headline appears.
- `t_feed_to_close_wait` is feed end to input close, not flush-compute time.
- R7 ASR and TTS share GPU0; Qwen2 uses GPU1; no unrelated GPU job was present.
- Append-only applies only to downstream delivery; internal drift and tokenizer seams remain visible.
- No quality noninferiority, task success, user-experience, production-reliability, or general LocalAgreement superiority claim is made.
- Fig.6 was relabeled from the unchanged 24 bin-statistic rows; the current PDF embeds CID TrueType and contains no raster image.
- External gates remain similarity checking and IEEE PDF eXpress on the final PDF.
