# SC focused revision execution record — 2026-09-08

## Authority, scope and status

Executed the author's repeated approval of the sole joint prefix-state repair mainline and migration of C1/C-E1/C-E2/A2 to an actual accompanying supplement. The historical proposal `sc_focused_revision_proposal_2026-09-06.md` is unchanged. Academic-paper revision guidance was loaded; this is a deterministic, traceable local revision, **not** a formally schema-validated revision-patch/authorization run. No invented schema PASS, independent panel, visual pass, journal quartile, or acceptance guarantee.

Initial Git HEAD: `d7af7abcdb4d0d610da806e40e33bcba038611bb`; initial `git status --short` was empty. No unrelated initial modifications existed. No commit, push, upload, model execution, or experimental rerun occurred. `src/`, all experimental scripts and result trees, GPU_RUN_NOTES, Chinese authoritative chapters, the historical SC Chinese translation, original figures, and the proposal are unchanged. R/B complete-tar verification from the prior turn was accepted and not repeated.

**Completed:** actual writing, two PDF builds, source ZIP extraction/recompilation, numerical/structural/citation checks, all-page PNG rendering, D-029, context milestone and handoff. **Awaiting:** one visual gate by the main orchestrator, independent scientific read-only review, author declarations/rights and final submission decision. No GPU task remains.

## Applied changes

- New title: “Joint Prefix-State Repair for Barge-In in Cascaded Spoken Dialogue: Integrity and Recovery Cost.” Updated abstract, six keywords and separate Highlights; removed duplicate in-PDF Highlights.
- Kept seven conventional main sections. Replaced five competing RQs with integrity, executable recovery cost and applicability questions. Kept invalidation as a method boundary case; moved trigger policy/timing detail out of the main method.
- Main evaluation now follows C2 + B → R → supporting A1/P1 → concise E3. The editable R table shows **all ten** case/event cells and actual 95% CIs; the supplement supplies all three endpoints, both arm means, and frozen bootstrap details.
- Standalone `supplementary.tex`/PDF provides S1 complete C1/C-E1/C-E2 policy, ten-condition grid and both favorable/unfavorable contrasts; S2 confounded A2; S3 all sixteen E3 estimand rows and detector agreement; S4 artifact/failure mapping; S5 all thirty R endpoint rows. No absent or merely promised supplement references.
- Preserved all four inconclusive E3 primary effects in the main table; exact-key/detector limitations remain explicit. C2 v1/v2 rejected and P1 v1 excluded remain transparent. A1/P1 are support, not central novelty claims.
- Consolidated scope limitations in discussion without dropping distinctions between software, device, acoustic, structural, numerical and downstream constructs. No clean-reprefill, cross-topology BF16/output-equivalence or broad functionality-no-loss claim.
- Retained four editable method figures and the original vector A1 figure. The historical E3 figure stays unchanged locally; the main table replaces its redundant display and it is excluded from the clean upload ZIP.
- Moved Acknowledgements directly before references. Guide/README/author checklist now cite the successful 2026-09-06 IAB LIVE verification, not stale 403 status. Data Option C is deposit/cite/link **or justified inability**, not compulsory DOI. No manuscript page/word reduction goal. No graphical abstract.
- `docs/decisions.md` adds reverse-chronological D-029; context timeline and handoff record R/B accepted, SC focus completed, current no-GPU status, and remaining gates. Historical decisions are retained rather than rewritten.

## Source-evidence checks

Formal source directory: `experiments/sci34_supplement/results/recovery_boundary/rb_formal_20260908T023934Z_f69e6478/`.

Read manifest, analysis, validation, boundary, all four session records/environment records, and protocol/runtime/campaign/boundary implementations. The frozen runtime is Qwen2-7B-Instruct (not Qwen2.5), BF16/SDPA, PyTorch 2.8.0+cu128, Transformers 4.57.1, one RTX 3090 per process, four process sessions, clean source `bfa7d7a37363d327cfa96b5448e151663ff578f8`, 118 snapshotted source files.

The CPU-only SC table extractor independently recomputed **320 interval identities, 160 matched structural-state pairs, and all 30 endpoint means and exact bootstrap intervals** from raw JSON. It uses no model or experiment code and writes only SC assets. The frozen bootstrap is four session means, 10,000 draws, seed 20260906, sorted indices 249/9749; fixtures/repeats/events are not 160 independent samples.

| Ordinary boundary | First-deliverable rebuild−crop mean (ms) | 95% session-cluster CI (ms) |
|---|---:|---|
| 512 tokens, case02/event0 | 113.621 | [111.555, 116.132] |
| 2048 tokens, case10/event0 | 465.377 | [464.181, 466.851] |
| 8192 tokens, case18/event0 | 1956.100 | [1940.446, 1968.986] |

8192 recovery-only: **1951.675 [1937.285, 1963.226] ms**. All 320 records/160 pairs delivered; nine cases produce ten cells, four pairs per cell/session. Cost sums synchronized intervals, includes first-generator KV commit, excludes preparation and CPU audit pauses, and is not continuous wall time or speech E2E. Second-event fixture reconciliation is disclosed; full rebuilding is not every serving baseline.

Stored diagnostics report top-1 160/160, max-absolute logit difference 0.59375, finite 320 float32 arrays of shape [1,152064]. **Continuation 144/160 was independently recomputed from raw token IDs**; all sixteen differences are case16/event1. Logit arrays were not reopened in this revision: the complete original tar was verified in the prior turn and the expanded repository copy intentionally lacks NPYs. Therefore this record does not claim a new full seal/NPY validation.

B stored evidence: 100 trajectories, 800 records, 1118 cursor checks, 27 C2 closure checks. This revision checked counts and read the independent software-reference logic, without claiming new acoustic/TTS/listening validation. C2 v3 remains 27 exact crops/60 matched steps; it is not a substitute for failed clean-reprefill comparisons.

Numeric provenance: `paper2/tougao/SC/latex/assets/numeric_checks.json`. Recorded worktree SHA-256 values are explicitly worktree identities, not substitutes for original LF seals. E1/E2 analysis_v2 and E3 weighting/dedup v2 supply the supplement tables; A2 descriptive values were checked against `experiments/results/paper2_reanalysis.json`.

## Build and technical validation

| Check | Actual result |
|---|---|
| Main PDF | **28 pages** (previous PDF 29); no reduction target |
| Supplement PDF | **9 pages**, standalone compilation |
| Main TeXcount | **6026 text words**; weighted sum 6416 |
| Supplement TeXcount | **2143 text words**; weighted sum 2311 |
| Abstract | **207 whitespace-delimited source words; 246 punctuation-split tokens**, both <250 |
| Keywords | 6 |
| Highlights | 5; character lengths **72, 76, 73, 72, 70** |
| References | 24 printed in main, 9 in supplement; all 26 database entries cited somewhere; no `nocite{*}` |
| Labels/citations | No duplicate labels, missing local reference targets, undefined citations/references, or citation/bibliography mismatches |
| Boxes | **0 overfull** in each final build; 9 main/2 supplement underfull diagnostics retained for visual judgment |
| Fonts | All embedded, no Type 3 |
| ZIP | 22 clean entries, both .tex entry points, no aux/log/build/Chinese source clutter |
| Independent ZIP build | Fresh extraction compiled both PDFs; PDF text extraction is byte-identical to local builds |
| Rendering | **28 + 9 = 37 PNGs**, all pages at 110 dpi; none inspected by editing agent |
| Git diff whitespace | `git diff --check` passed; standard LF→CRLF notices are not whitespace failures |

TeXcount text counts exclude the rendered bibliography and treat table/math material according to its defaults; the weighted sum additionally includes headers/captions/math units and is not a journal word-limit declaration. Custom generated table rows are not a prose word count. The first parser selected the main entry-file count rather than the final “Sum of files”; this was corrected to the actual 6026/6416 totals shown in the saved TeXcount output.

Build issues resolved locally: expandable table inputs avoid a booktabs/input alignment error; the out-of-tree MiKTeX bibliography name collision was removed by supplying the correct local `.bib`/`.bst`; fixed-size supplement tables avoid longtable infinite-glue warnings. Final logs contain none of these failures. MiKTeX's update-check notice does not prevent successful compilation. No installed packages or experiment dependencies were changed.

**Skipped/not claimed:** formal revision-schema/criteria-binding validation; new reference-existence/retraction or web-guide lookup; full original-tar/NPY/seal revalidation; new GPU or acoustic experiments; independent multi-agent scientific review; visual acceptance. The available toolset did not include scientific-review delegation, so that read-only review remains with the main orchestrator.

## Final artifacts and visual gate

All paths below are repository-relative within `D:/project/my/research/streamllm_p2/`.

- Main upload PDF: `paper2/tougao/SC/Speech_Communication_manuscript.pdf`.
- **Separate supplement upload PDF:** `paper2/tougao/SC/Speech_Communication_supplementary.pdf`.
- Editable package: `paper2/tougao/SC/Speech_Communication_LaTeX_source.zip`.
- Separate Highlights: `paper2/tougao/SC/latex/highlights.txt`.
- Main PNGs: `paper2/review/sc_revision_validation_2026-09-08/png_main/page-01.png` through `page-28.png`.
- Supplement PNGs: `paper2/review/sc_revision_validation_2026-09-08/png_supplementary/page-1.png` through `page-9.png`.
- Machine technical report, build/word/font logs and independent extracted ZIP: `paper2/review/sc_revision_validation_2026-09-08/`.
- Pre-edit backup: `paper2/review/sc_revision_validation_2026-09-08/before_revision.zip`.
- Deterministic source/document patch and before/after hashes: `paper2/review/sc_revision_validation_2026-09-08/revision.patch` and `change_manifest.json`.
- Git baseline/final logs: `git_before.txt`, `git_after.txt` in the same validation directory.

**Single visual gate owner: main orchestrator. Status: awaiting judgment.** Inspect all 37 PNGs, including method-figure labels/arrows, R/E3/A1 table widths, supplement artifact path wrapping, the complete R endpoint table, and declaration/reference transitions. Do not carry forward the previous 29-page visual approval. Author placeholders are intentional unresolved facts, not proof of submission readiness.
