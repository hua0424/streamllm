# CISR Revision Tracking

## Paper information

| Field | Value |
|---|---|
| Paper | Latency Optimization of Cascaded Voice Dialogue Systems with a Pipeline-Parallel Streaming Architecture |
| Revision date | 2026-08-22 |
| Target | CISR conference submission |
| Data-lock baseline | `7c93b77` |
| Revised manuscript | `main.tex` / `main.pdf` |
| Reviewer response | `response_to_reviewers.tex` / `response_to_reviewers.pdf` |

## Revision tracking table

| ID | Reviewer concern | Resolution | Manuscript location | Locked evidence | Status |
|---|---|---|---|---|---|
| R1 | Validate on human recordings and acoustic variation | Added deterministic LibriSpeech/AISHELL-1 concatenated human-read sets, 12 augmentation conditions, real-speech WER/CER, and babble failure boundary | Section IV-E p.8; Section V-D and Table VI pp.10–11; Limitations p.12 | `experiments/results/revision/r2_real_speech/ttft_real.csv`; `wer_real.csv`; build/augmentation manifests; manual spot check | RESOLVED with disclosed boundary |
| R2 | Report first audible output including endpointing, communication, and TTS | Added timestamp-defined TTFA, unified one-clock experiment, total distribution and six-component closed decomposition | Introduction p.1; Section III-C p.4; Section V-G p.11; Table VIII p.12 | `experiments/results/revision/r7_ttfa_unified/table_viii/TABLE_VIII_ASSEMBLED.md` | RESOLVED |
| R3 | Add std/P95/P99 and repeated measurements; assess contention/jitter | Expanded Tables III–V, replaced Fig.6, added paired inference and 3-run CV distribution. GPU contention was not run and is acknowledged in Limitations | Fig.6 and repeatability p.9; Tables III–V pp.10–11; statistical protocol p.4; Limitations p.12 | `r1_stats/table3_latency_percentiles.csv`; `table4_ablation_percentiles.csv`; `table5_context_percentiles.csv`; `repeat_cv_summary.csv`; `stats_inference/paired_inference.csv` | RESOLVED except GPU contention, recorded as DELIBERATE_LIMITATION |
| R4 | Add matched stronger streaming baseline | Added project-internal LocalAgreement-2-style baseline using matched weights, engine, segmenter, hardware, and paired sample subset | Related work/contributions p.2; Section V-E pp.10–11; Table VII p.11 | `r3_baseline_la/ttft_la_vs_b.csv`; `wer_la_vs_b.csv`; `LA_METHOD_AND_EXCLUSION.md` | RESOLVED |
| R5 | Clarify corrections/tokenizer/KV behavior and evaluate downstream meaning | Defined two-layer commit contract; measured rollback/internal corrections and tokenizer seams; added three-track exploratory response-quality evaluation | Section IV-B p.6; Section V-F pp.10–11; Limitations p.12 | `r4_commit/commit_divergence.json`; `tokenizer_seams.csv`; `r5_semantic/semantic_consistency.csv`; `REPRO_METADATA.md` | RESOLVED with exploratory-evidence limitation |

## Commitment ledger

```yaml
- concern_id: R1
  commitment_extracted:
    - commitment_text: "Add human-recorded speech with speaker, accent, noise, speed, and endpoint variation."
      commitment_type: add_experiment
      required_evidence_type: new_table
      fulfillment_status: fulfilled
    - commitment_text: "Disclose real-speech boundary conditions."
      commitment_type: add_analysis
      required_evidence_type: discussion_paragraph
      fulfillment_status: fulfilled

- concern_id: R2
  commitment_extracted:
    - commitment_text: "Define and measure time to first playable audio on a unified timeline."
      commitment_type: add_experiment
      required_evidence_type: new_table
      fulfillment_status: fulfilled
    - commitment_text: "Account for communication and TTS boundaries."
      commitment_type: add_clarification
      required_evidence_type: methods_paragraph
      fulfillment_status: fulfilled

- concern_id: R3
  commitment_extracted:
    - commitment_text: "Add standard deviations and P95/P99 to the main tables."
      commitment_type: add_analysis
      required_evidence_type: new_table
      fulfillment_status: fulfilled
    - commitment_text: "Add repeated measurements and characterize jitter."
      commitment_type: add_experiment
      required_evidence_type: discussion_paragraph
      fulfillment_status: fulfilled
    - commitment_text: "Test GPU contention."
      commitment_type: add_experiment
      required_evidence_type: discussion_paragraph
      fulfillment_status: partial
      unfulfilled_rationale: "The revision did not run a contention workload. Experiments used otherwise exclusive GPUs, and the missing contention condition is explicitly acknowledged in Limitations and Future Work."

- concern_id: R4
  commitment_extracted:
    - commitment_text: "Add a stronger streaming baseline with matched models and hardware."
      commitment_type: add_experiment
      required_evidence_type: new_table
      fulfillment_status: fulfilled

- concern_id: R5
  commitment_extracted:
    - commitment_text: "Clarify committed text, tokenizer seams, and KV-cache behavior under ASR changes."
      commitment_type: add_clarification
      required_evidence_type: methods_paragraph
      fulfillment_status: fulfilled
    - commitment_text: "Report rollback/correction frequency."
      commitment_type: add_analysis
      required_evidence_type: discussion_paragraph
      fulfillment_status: fulfilled
    - commitment_text: "Evaluate downstream response meaning beyond WER/CER."
      commitment_type: add_experiment
      required_evidence_type: discussion_paragraph
      fulfillment_status: fulfilled
```

## Integrity constraints retained

- Tables III–V and Fig.6 use the original platform; repeated measurements and Tables VI–VIII use the second platform.
- No cross-platform latency scaling is used.
- Table VIII uses only the R7 unified timeline; the historical cross-run TTFA assembly is excluded.
- `t_feed_to_close_wait` means feed end to pipeline input close and is not flush-compute time.
- Append-only applies only to downstream delivery; internal ASR drift is reported.
- Semantic evidence is described as exploratory, not equivalent, lossless, or statistically indistinguishable.
- Real speech is described as concatenated human-read speech, not spontaneous dialogue.
