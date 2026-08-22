# CISR 2026 LaTeX Submission Compliance — Final Local Status after CPU Reanalysis

- **Requirements**: https://www.icisr.com/submission/
- **Local instructions**: `templates/投稿须知 IEEE.docx`
- **Official template**: `templates/ieee-template-latex.zip`
- **Manuscript**: `submission2/main.tex`, `main.pdf`, `refs.bib`
- **Date**: 2026-08-22

## 1. Current verdict

The revised manuscript satisfies the locally checkable CISR LaTeX, structure, figure/table, bibliography, metadata, and font requirements. The latest scientific corrections used only deterministic CPU reanalysis of locked records; no model or data-generation experiment was rerun. Similarity checking and IEEE PDF eXpress remain external gates.

| Dimension | Result |
|---|---|
| Official CISR/IEEE LaTeX template | PASS |
| English manuscript and required structure | PASS |
| Letter, two-column layout | PASS |
| Minimum body-page requirement | PASS |
| Editable tables/equations | PASS |
| All figures/tables cited | PASS |
| Reference count, recency, diversity | PASS |
| Citation/BibTeX closure | PASS (31/31) |
| Claim-to-evidence scope | PASS after historical-policy/failure/TTFA correction |
| PDF metadata | PASS |
| All fonts embedded | PASS |
| Type 3 fonts | 0 |
| Similarity check | PENDING external |
| IEEE PDF eXpress | PENDING external |
| Local readiness | **READY FOR EXTERNAL GATES** |

## 2. Requirements and official template

The conference requires English, at least four body pages excluding references, use of its template, diverse scholarly references, high-resolution English figures, and editable formulas/tables. The local instructions additionally require at least ten references, recent literature, sources from at least three countries, explicit citation of all figures/tables/references, similarity below 24% overall and 3% per source, and IEEE PDF eXpress.

`submission2/IEEEtran.cls` and `IEEEtran.bst` remain byte-identical to the downloaded official files. The manuscript uses `\documentclass[conference]{IEEEtran}` on US Letter.

## 3. Final PDF structure

- 14 pages;
- US Letter, 612 × 792 pt;
- IEEE conference two-column layout;
- no blank page or visible clipping;
- title, ordered authors, affiliations, corresponding author, abstract, keywords, sections, acknowledgment, and references present;
- PDF metadata title: `Latency Optimization of Cascaded Voice Dialogue Systems with a Pipeline-Parallel Streaming Architecture`;
- author: `Haihua Mo; Zhengyou Liang`;
- subject and keywords synchronized with the revised paper.

## 4. Figures and Fig.6

- Six figures, all explicitly referenced and captioned in English.
- Fig.1–5 retain their existing high-resolution sources.
- Fig.6 was regenerated only from the existing aggregate bins; no experiment or audio/model processing occurred.
- All 24 rows in `Fig6.bins.csv` are byte-identical to the prior version.
- The new axis names the metric `Post-feed residual TTFT (ms)` and the figure states `Historical latched-trigger policy`.

Artifacts:

```text
Fig6.pdf SHA-256  d38927c9b25139268b2fb00003f573184a0250ede1d502638a1f57d25d573cbb
Fig6.svg SHA-256  ca9d3fd1efcb22a37cd23fc21a78e54bca55b4a752d6ce8edc7631fbcadb91b2
```

Verification:

- experiment and manuscript PDF copies are byte-identical;
- PDF and SVG use the same physical dimensions;
- PDF contains no raster image;
- DejaVuSans is embedded/subset as CID TrueType;
- SVG is self-contained/path based;
- manuscript Type 3 count is zero.

## 5. Tables and equations

- Eight editable LaTeX tables, numbered I–VIII and explicitly referenced.
- Table IV documents 505 candidates, seven run-log-confirmed externally contaminated executions, the locked 498-turn valid set, its cluster interval, and non-causal arm labels.
- Table VII states that the LA-2-style result is one configured operating point with unmatched trigger policy.
- Table VIII labels input language and makes the first-sentence/full-response policy asymmetry visible.
- Equations remain editable, numbered by IEEEtran, and referenced through `\eqref`.
- No table or equation produces an overfull box.

## 6. Methods and evidence corrections

The final paper distinguishes:

1. conceptual physical-speech-end TTFT;
2. legacy post-feed residual TTFT;
3. server-side time to first playable PCM.

The historical ASR policy is accurately reported:

- `total_duration` accumulates speech admitted to the ASR cache;
- removed segments do not decrement it;
- the 2.0-s condition is a one-time cumulative-seen-duration startup gate;
- after the first crossing it remains open, and later calls depend mainly on segment count/final processing;
- all existing results measure this historical policy;
- no result is claimed for a corrected current-queue-duration trigger.

No long experiment was rerun. The new derived results come from `experiments/scripts/cpu_revision_analysis.py` and locked JSON/CSV/JSONL inputs.

## 7. Deterministic CPU reanalysis

Output: `experiments/results/revision/minimal_cpu_reanalysis`.

Key locked results:

- 505 candidate executions − 7 run-log-confirmed concurrent-external-program contaminations = 498 valid three-arm turns;
- locked 498 turns/99 dialogues: 74.34% [72.62%, 75.76%];
- contaminated records remain in `sample_exclusions.csv` for audit but are not analyzed as system outcomes;
- second-platform retained subset: 70.36% [68.50%, 71.90%];
- configured LA-2-style minus B: 541.11 ms [485.66, 599.14];
- overall Baseline-minus-ASR-only contrast: 3332.13 ms [3064.38, 3572.61];
- ASR-only-minus-Full contrast: 15.50 ms [10.95, 19.66].

The script records input hashes, validates all 1,133 metadata IDs, fails closed on pairing/ledger violations, and uses 10,000 dialogue-cluster resamples. A second formal run produced byte-identical hashes for all ten generated outputs.

## 8. TTFA policy disclosure

Table VIII is explicitly server-side and policy-level:

- B sends the first completed sentence;
- A sends the capped full response;
- mean TTS characters A/B: 204.10/17.74;
- exact fixed Chinese greeting outputs A/B: 0/50 vs 25/50;
- max-token stops A/B: 41/50 vs 43/50;
- 19/25 English-input B TTS strings contain no ASCII Latin letters;
- ASR and TTS share GPU0, Qwen2 uses GPU1;
- no client transport, device scheduling, or acoustic playback timestamp is included.

The 22.27-s versus 3.11-s median is not presented as an architecture-only or matched-TTS causal effect.

## 9. References

- 31 cited entries, 31 bibliography records;
- 21 dated 2023–2026 and 18 dated 2024–2026;
- no missing, orphan, duplicate-key, duplicate-DOI, or normalized-title duplicate;
- direct primary sources include Simul-Whisper, Prompt Cache, SGLang/RadixAttention, Conformer, PagedAttention, FlashAttention, StreamingLLM, Mini-Omni, Moshi, and LLaMA-Omni;
- official `IEEEtran.bst` remains unchanged; DOI database fields need not be visibly rendered.

## 10. Build and QA

```text
main.pdf: 14 pages, Letter
response_to_reviewers.pdf: 5 pages, Letter
```

Final checks:

- no fatal LaTeX error;
- no overfull box;
- no undefined citation/reference;
- no missing/uncited bibliography entry;
- all fonts embedded/subset;
- zero Type 3 fonts;
- no blank pages or clipping;
- source tables and equations remain editable.

MiKTeX's update reminder is an environment notice rather than a document error. Underfull box warnings remain non-blocking IEEE column-justification artifacts.

## 11. Source package

Final compilation inputs remain:

```text
main.tex
refs.bib
main.bbl
IEEEtran.cls
IEEEtran.bst
Fig1.pdf ... Fig6.pdf
```

Exclude internal reports, CPU-analysis products, response files, and `*.aux/*.log/*.blg/*.out` from the manuscript source ZIP unless separately requested.

## 12. Remaining workflow

1. Run similarity checking: ≤24% overall and ≤3% per source.
2. Run IEEE PDF eXpress on the final `main.pdf`.
3. If PDF eXpress alters pagination, verify any external submission metadata.
4. Build the clean LaTeX source ZIP.
5. Complete copyright transfer after acceptance.
