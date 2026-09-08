# SC focused revision execution record — 2026-09-08

## Final closure — scientific and fixable-layout checks complete

**Status: scientifically revised; technical and visual-layout checks complete; author metadata/rights blocked.** Orchestrator-dispatched read-only scientific agent `6343...` completed the separate check: all 30 R intervals and E3 accuracy PASS. This is not an independent panel or a claim of statistically independent reviewer errors.

Final judge PASS for main 26/27 and supplement 8 closes all remaining fixable visual defects. Main 25 has only the known Acknowledgements placeholder. Remaining content blockers are author facts on main 1/23/24/25 and supplement 1; none is falsely resolved. No scientific or visual-layout check is pending. The editing agent inspected no images; closure records the orchestrator's final judge result.

Main remains **27 pages**, supplement **8 pages**, source ZIP **23 entries**. This closing pass changes status documentation only: all TeX, bibliography, figure and manuscript/supplement PDF bytes remain unchanged. README/AUTHOR_CONFIRM are synchronized into the ZIP, and fresh extraction independently compiles both to the same extracted PDF text. No further image gate is needed for unchanged TeX/PDF. Current package hashes, compile/check logs and protected-path audit are in `paper2/review/sc_revision_validation_2026-09-08/final_status/` (`checks.json`). Earlier pending statements, hashes and page counts below are explicitly superseded history.

Author identity, acknowledgements, CRediT, funding, conflicts, ethics/consent, AI disclosure, release rights/license and data availability still require author confirmation. Fresh reference/retraction checks and applicable search exports were not silently completed. No submission-ready, schema-certification, ranking or acceptance claim is made.

## Superseded final-layout addendum — bibliography closure and review provenance

This is the current status; all earlier page counts, hashes, review provenance shorthand and pending descriptions below are superseded historical records.

The scientific check was completed by **orchestrator-dispatched read-only agent `6343...`**: all 30 R intervals and E3 accuracy passed. It was not an author assertion or a new scientific check performed by this editing agent. A separate reviewer assignment does not establish statistically independent reviewer errors.

Targeted supplement pages 5–8 passed. The remaining page 9 contained only the final four-line Zou reference. A bibliography-local zero inter-entry skip and no-entry-split setting now fit all references on **page 8 at unchanged font size**, with no citations, URLs, or text removed. Source comparison confirms the only supplement change is the bibliography-layout wrapper. Earlier supplement pages 1–7 are text-extraction identical.

Main targeted pages 16–22 and 27 passed; pages 23–25 retain known author placeholders. The last identified repair was the Microsoft URL protrusion on page 26. Main now loads `xurl` rather than `url`, retaining the exact bibliography database and link. Earlier main pages 1–24 are text-extraction identical. The main remains **27 pages**; the supplement is **8 pages**; the clean source ZIP has **23 entries**.

Both final PDFs were rebuilt and freshly extracted ZIP sources independently compiled; extracted text matches final PDFs exactly. Zero overfull boxes/undefined citations/references, embedded fonts, no Type 3. All scientific material and author facts remain unchanged. No images were inspected.

**Current targeted visual gate:** main pages 25–27 and supplement page 8 await judge verification; no all-page pass is claimed. Current PNGs are in `paper2/review/sc_revision_validation_2026-09-08/supp_bib_repair/png_main/` (page-25.png through page-27.png) and `supp_bib_repair/png_supplementary/page-8.png`. Current hashes/checks, backup, patch, and independent extracted source build are in `supp_bib_repair/`; use its `checks.json`, not historical checks from earlier build stages.

## Superseded addendum — external audit and judge-directed layout repairs

This addendum supersedes the initial-build page counts/status below, which remain historical evidence.

- **Scientific review complete:** user relayed a separate scientific PASS confirming all 30 R intervals and E3 accuracy. No scientific changes were made in the subsequent layout repair.
- **Supplement judge:** page 1 author placeholder is known and retained; page 5 stranded the Rejected C2 heading/path; page 7 had excessive blank space before the full-page Table S4. Other original supplement pages passed. Added six-line keep-space before the rejected-C2 entry, changed Table S4 to a flowing longtable with repeated header and finite vertical spacing, and removed the forced reference-page break. Exact source comparison after removing these layout substitutions confirms all scientific text unchanged.
- **Main judge:** fixed double periods in three run-in paragraph headings; regenerated the SC A1 vector figure from unchanged accepted medians/quartiles/ratios with 11–13 pt labels and full text-width placement; tightened bibliography inter-entry spacing and prevented entry-internal page breaks. The complete final Zou reference now stays on page 27, eliminating the former almost-empty page 28. Remaining original pages 1/23/24/25 failures were only known author/declaration placeholders, left unchanged; other main pages passed.
- **Current outputs:** main **27 pages**, supplement **9 pages**, source ZIP **23 entries** including the new SC-only editable A1 figure generator. Both PDFs rebuilt; fresh ZIP extraction compiled both and produced identical extracted text to final outputs. Zero overfull/undefined-reference/citation errors; all fonts embedded, no Type 3. No experimental source/results/Chinese chapter changes or GPU runs.
- **Current visual gate:** judge-directed repairs implemented, **targeted recheck pending**. The editing agent inspected no images. Current affected-page PNGs are `paper2/review/sc_revision_validation_2026-09-08/main_layout_repair/png_main/page-16.png` through `page-27.png` (**12 PNGs**) and `main_layout_repair/png_supplementary/page-5.png` through `page-9.png` (**5 PNGs**). The earlier root PNG directories are historical and must not be used for repaired pages.
- **Current checks/hashes:** `paper2/review/sc_revision_validation_2026-09-08/main_layout_repair/checks.json`; both independent ZIP builds in its `zip_recompile/`. Supplement-only source normalization/checks in sibling `supp_layout_repair/checks.json`. Both repair directories contain pre-repair backups and layout patches. Initial `technical_checks.json`/`change_manifest.json` describe the earlier build, not these current artifact hashes.
- No formal revision-schema certification, new literature/retraction verification, or submission-ready assertion. Author identity, permissions and declarations remain unresolved.


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
