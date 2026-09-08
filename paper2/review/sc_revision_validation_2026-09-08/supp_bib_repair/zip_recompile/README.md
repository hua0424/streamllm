# Speech Communication manuscript and separate supplement

Focused revision: 2026-09-08. One method mainline: joint prefix-state repair. R/B formal evidence is complete and accepted for bounded structural/software and operational-cost claims. No new experiments were run for this revision. Author placeholders remain; this is not submission-ready and no acceptance or journal-quartile claim is made.

## Build

From this directory, using installed MiKTeX or TeX Live:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

Build in the source directory for an unambiguous local `references.bib`. If using a separate output directory on MiKTeX, copy `references.bib` and `elsarticle-harv.bst` into it first; otherwise an unrelated installed bibliography with the same name can be resolved. The main uses the included official `elsarticle` class; the independently compiled supplement uses the standard `article` class. Both use the same author–year bibliography and print only cited entries. Standard TeX packages must be installed.

## Files and uploads

- `main.tex` + `sections/`: seven-section main manuscript; submit its PDF as the manuscript, not as supplementary material.
- `supplementary.tex`: separate accompanying Supplementary Material. Submit `Speech_Communication_supplementary.pdf` separately as Supplementary Material with the description “Candidate-generation and history-naturalization characterizations, downstream sensitivities, artifact mapping, and complete recovery-cost endpoints.” It is cited explicitly by the main text. Do not omit this upload or assume a repository pointer substitutes for it.
- `highlights.txt`: five separate editable Highlights, not an extra page in the manuscript PDF.
- `references.bib`, `elsarticle.cls`, `elsarticle-harv.bst`: editable reference database and main journal style.
- `tables/`: editable JSON-derived table rows for main and supplement.
- `figures/Figure_5_cost_microbenchmark.pdf`: main-text A1 figure, regenerated from the same accepted medians/quartiles with larger labels and full-text-width placement; editable generator `assets/generate_a1_figure.py`. Four method figures are editable TikZ in the sections. `Figure_6_e3_intervals.pdf` is a preserved historical asset, not referenced or included in the clean upload ZIP; the main reports all E3 estimates in a table.
- `assets/generate_tables.py`: optional CPU-only extraction/check of accepted JSON. Requires the repository via `uv run --no-sync python assets/generate_tables.py --repo REPOSITORY_ROOT`. Tables are already included, so compilation does not require experimental data or this script. `assets/numeric_checks.json` records actual checks and their limits.
- `AUTHOR_CONFIRM.md`: unresolved author-controlled items.

`../Speech_Communication_LaTeX_source.zip` is a clean typesetting package containing both entry points, their required assets, Highlights, instructions, and numeric extraction provenance. Upload the editable source package in addition to the main PDF and separate supplement PDF according to the submission interface. It does not contain model weights or the complete experimental archive. Local `main_zh-CN.md` is an untouched historical translation, not an updated or submitted source.

## Main versus supplement

The main evaluates C2 integrity, B software replay, R same-policy executable recovery, supporting A1/P1 costs, and concise inconclusive E3 results. C1/C-E1/C-E2 and A2 are not main research questions or contribution evidence: S1/S2 preserve their favorable, unfavorable, and confounded results together. S3 gives all four E3 weighting estimands; S4 maps artifacts and rejected protocols; S5 reports every R endpoint and arm mean.

R has 320 records/160 event pairs, nine cases/ten cells, four process sessions. First-deliverable cost sums synchronized recovery, suffix/header, and first-generator-return intervals, includes first-token KV commit, and excludes preparation/CPU audit pauses. It is not continuous wall-clock or speech E2E. Full rebuild is not every serving baseline. 160/160 top-1 but 144/160 short continuations match; no numerical/output equivalence is claimed. B is software replay, not acoustic alignment.

## Guide source and remaining gates

The official guide was successfully read through the user's IAB/side-panel browser on 2026-09-06; the historical 2026-09-05 WebFetch 403 is not the current verification status. The evidence record is `paper2/review/sc_focused_revision_proposal_2026-09-06.md` §2 (repository-relative). No new live fetch was performed on 2026-09-08.

The verified page permits Original Full-length Research and Review, single-anonymized review, editable .tex, abstract at most 250 words, 1–7 keywords, 3–5 Highlights of at most 85 characters, and author–year references in a consistent format. No full-manuscript word/page cap was found on that page. No length-reduction target was applied. Graphical abstract is optional and was not generated. Acknowledgements now immediately precede references. Data Option C means deposit/cite/link, or a justified inability to share; a DOI is not an unconditional requirement.

Current abstract: 207 whitespace-delimited source words (also below 250 under punctuation-split counting); six keywords. Final page/word/font/citation checks are recorded in `paper2/review/sc_focused_revision_2026-09-08.md`. All final PDF pages are rendered for one visual gate assigned to the main orchestrator. The editing agent has not opened any PNG and does not claim visual acceptance. The orchestrator-dispatched read-only scientific agent (identifier supplied as `6343...`) completed its check: PASS for all 30 R intervals and E3 accuracy. This is a separate review assignment, not a claim of statistically independent reviewer errors. Supplement visual review found two pagination issues (pages 5 and 7); layout-only repairs are rebuilt and await targeted visual recheck. Other supplement pages passed apart from the known author placeholder. The main judge then identified doubled run-in punctuation, small A1 labels and a stranded final bibliography entry; these received layout-only fixes. All other main pages passed except known metadata/declaration placeholders. Targeted supplement pages 5–8 subsequently passed; the remaining isolated Zou reference on page 9 was repaired by adjusting bibliography spacing without changing font size or content. The supplement is now eight pages; only its revised page 8 awaits recheck. Main targeted pages 16–22 and 27 passed, while 23–25 retain known placeholders. A further page-26 Microsoft URL margin protrusion was repaired using breakable URL handling, preserving the link. The affected bibliography pages await targeted recheck; no all-page visual pass is claimed. Fresh citation/retraction verification was not performed here. Formal academic revision-schema authorization was not validated; the backup, patch, hashes, and deterministic numeric checks are ordinary traceability artifacts, not schema certification.
