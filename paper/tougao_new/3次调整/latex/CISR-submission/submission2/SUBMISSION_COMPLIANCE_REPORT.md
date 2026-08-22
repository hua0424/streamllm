# CISR 2026 LaTeX Submission Compliance — Final Local Status

- **Checked against**: https://www.icisr.com/submission/
- **Local instructions**: `templates/投稿须知 IEEE.docx`
- **Official template**: `templates/ieee-template-latex.zip`
- **Manuscript**: `submission2/main.tex`, `main.pdf`, `refs.bib`
- **Date**: 2026-08-22

## 1. Current verdict

The manuscript satisfies the locally checkable CISR LaTeX, structure, figure/table, bibliography, metadata, and font requirements. The regenerated Fig.6 removes the previous Type 3 font blocker. Remaining requirements are external process gates: similarity checking and IEEE PDF eXpress.

| Dimension | Result |
|---|---|
| Official CISR/IEEE LaTeX template | PASS |
| English manuscript and required structure | PASS |
| Letter, two-column layout, symmetric margins | PASS |
| Minimum body-page requirement | PASS |
| Editable tables and equations | PASS |
| All figures and tables cited in text | PASS |
| Figure/table numbering and caption placement | PASS |
| Reference count, recency, diversity | PASS |
| Citation-key and bibliography closure | PASS |
| Claim-to-source attribution | PASS after primary-source repair |
| PDF title/author/subject/keywords metadata | PASS |
| All PDF fonts embedded | PASS |
| No Type 3 fonts | PASS |
| Similarity requirement | PENDING external check |
| IEEE PDF eXpress | PENDING external validation |
| Current readiness | **LOCALLY READY; EXTERNAL GATES PENDING** |

## 2. CISR requirements applied

The conference website requires English, at least four full body pages excluding references, use of the template, at least eight diverse/authoritative references, high-resolution English-language figures, and editable formulas/tables. The downloaded local instructions are stricter and require at least ten references, recent literature, author sources from at least three countries, explicit citation of all figures/tables/references, IEEE PDF eXpress, and similarity below 24% overall / 3% per source.

The stricter local thresholds are used below.

## 3. Official-template identity

The manuscript copies are byte-identical to the downloaded template files:

- `IEEEtran.cls`: 288,304 bytes, SHA-256 `c972aca108fda004...e003f55`;
- `IEEEtran.bst`: 61,632 bytes, SHA-256 `d83aa3c9b47fc120...5ded24`.

The document uses:

```latex
\documentclass[conference]{IEEEtran}
```

The official class/style files were not edited.

## 4. Page and structure compliance

Current PDF properties:

- 13 pages;
- US Letter, 612 × 792 pt;
- IEEE conference two-column layout;
- symmetric content margins;
- no blank pages;
- no visible clipping;
- References begin on page 12, leaving substantially more than four body pages.

The manuscript contains title, ordered authors, affiliations, corresponding-author information, abstract, keywords, numbered sections, acknowledgment, and references.

PDF metadata contains:

- Title: `Latency Optimization of Cascaded Voice Dialogue Systems with a Pipeline-Parallel Streaming Architecture`;
- Author: `Haihua Mo; Zhengyou Liang`;
- Subject and keywords matching the manuscript.

## 5. Figure compliance and Fig.6 closure

### 5.1 General figure checks

- Six figures, numbered Fig.1–Fig.6.
- All six labels are explicitly referenced in the text.
- Captions and figures remain in the same LaTeX float.
- No Chinese characters detected in the PDF text layer or visual review.
- Fig.1–5 source rasters are approximately 384 ppi and render above 300 dpi at their placed sizes.
- Fig.6 is a vector plot with no embedded raster images.

### 5.2 Regenerated Fig.6

Pulled commit:

```text
5376b82 Fig6重绘消除Type3字体:pdf.fonttype=42+新增自包含SVG,submission2同步更新
```

Artifacts:

```text
Fig6.pdf SHA-256  a67be51b9670b64c0adfa8f36368d0b033d98d7c4b5a4ca3c2851a11d0535828
Fig6.svg SHA-256  b9fd3d8383e4c322ed48285f766b32ba7df2ed4aba54fdeb5ff5d7b51dfbc2df
```

Verification:

- experiment-result PDF and `submission2/Fig6.pdf` are byte-identical;
- PDF and SVG both use 446.4 × 259.2 pt / identical viewBox dimensions;
- all 24 CSV bin rows are field-identical to the previous figure version;
- the PDF contains 43 vector drawing objects and zero embedded images;
- the PDF embeds DejaVuSans as `CID TrueType`, subsetted, embedded, and Unicode mapped;
- no Type 3 font remains in Fig.6;
- the self-contained SVG converts text to paths and has no external font dependency;
- PDF/SVG side-by-side rendering is visually equivalent; mean absolute pixel difference is below 1/255 per RGB channel and is attributable to renderer anti-aliasing;
- Fig.6 appears on manuscript page 9, is readable, and is not clipped.

### 5.3 Final manuscript font check

`pdffonts main.pdf` shows:

- body/math fonts: embedded Type 1;
- Fig.6 font: embedded CID TrueType;
- Type 3 count: **0**.

This closes the local IEEE font blocker. IEEE PDF eXpress must still provide the official acceptance decision.

## 6. Tables and equations

### Tables

- Eight editable LaTeX tables, numbered Table I–VIII.
- All table captions remain above the tables.
- All eight tables are explicitly referenced in the text.
- Table IV is cited through `\ref{tab:ablation}`.
- Table VII is cited through `\ref{tab:la}`.
- No table overflow was reported by LaTeX.

### Equations

- Equations remain editable LaTeX, centered, consecutively numbered, and right-aligned by IEEEtran.
- All running-text equation references use the official-template `\eqref{...}` form.
- No legacy `\ref{eq:...}` occurrences remain.
- No equation value or mathematical conclusion was changed.

## 7. Methods/configuration correction

The manuscript no longer conflates the ASR trigger with VAD minimum speech duration.

- Legacy TTFT/ablation/context/human-read/matched-baseline experiments: 500-ms chunks, Silero threshold 0.5, 500-ms minimum speech, 300-ms minimum silence, and a separate 2.0-s ASR recognition trigger.
- R7 unified TTFA: 500-ms chunks, Silero threshold 0.5, 250-ms minimum speech, 100-ms minimum silence, 30-ms speech padding, and a 2.0-s ASR recognition trigger.

The text states that settings are fixed within each paired comparison and does not claim they are optimal because no VAD-parameter ablation was performed. No experiment or result was changed.

## 8. References

### Quantity and recency

- 28 references, above the stricter minimum of 10.
- 18 references dated 2023–2026.
- 15 references dated 2024–2026.
- Author/institutional origins cover substantially more than three countries.

### Closure

- 28 unique cited keys;
- 28 BibTeX entries;
- no missing key;
- no uncited entry;
- no undefined citation/reference;
- bibliography numbering remains in first-citation order.

### Primary-source repair

Primary sources directly support Conformer, vLLM/PagedAttention, FlashAttention, StreamingLLM, Mini-Omni, Moshi, LLaMA-Omni, and Whisper large-v3-turbo configuration. Mismatched rVAD, ASR-survey, compression-survey, efficient-Transformer-survey, and self-supervised-review entries were removed after their uses were replaced.

### Metadata repairs

- Corrected ACM Computing Surveys article 209 to pages 1–35 and six authors.
- Corrected MultiWOZ venue address to Brussels, Belgium.
- Protected Whisper, Mandarin, and M3-Embedding title capitalization.
- Clarified dynamic software/service identities and unavailable immutable revisions.

### DOI display

The official CISR `IEEEtran.bst` ignores `doi` fields. CISR does not require visible DOI strings, so the official style remains unchanged. DOI fields are retained as database metadata where available.

## 9. Build and QA results

The final manuscript and response letter compile successfully.

```text
main.pdf: 13 pages, Letter
response_to_reviewers.pdf: 5 pages, Letter
```

Checks:

- no fatal LaTeX errors;
- no overfull boxes;
- no undefined citations or references;
- no missing/uncited bibliography entries;
- all fonts embedded;
- zero Type 3 fonts;
- no blank pages;
- no content overflow;
- adequate page fill ratio;
- symmetric margins;
- table centering check passed;
- visual inspection of pages 8–13 passed.

Generic PDF-QA warnings about a non-full-bleed cover or left/right column table placement do not apply to the official IEEE two-column template. Punctuation warnings on the reference page arise from quoted reference titles at line breaks and are not LaTeX errors.

## 10. Source-package completeness

Required compilation inputs are present:

```text
main.tex
refs.bib
main.bbl
IEEEtran.cls
IEEEtran.bst
Fig1.pdf ... Fig6.pdf
```

For final submission, exclude internal reports and build products unless the submission system explicitly requests them:

```text
CITATION_AUDIT_REPORT.md
SUBMISSION_COMPLIANCE_REPORT.md
REVISION_TRACKING.md
response_to_reviewers.* (upload separately if requested)
*.aux *.log *.blg *.out
```

## 11. Remaining external workflow

1. Run similarity checking: ≤24% overall and ≤3% per source.
2. Run IEEE PDF eXpress and use the validated PDF.
3. If PDF eXpress changes pagination, recheck the response-letter page references.
4. Build the clean LaTeX source ZIP.
5. Complete copyright transfer after acceptance.

## 12. Scientific-result boundary

The manuscript/citation repair changed no Table I–VIII values, sample counts, exclusions, confidence intervals, p-values, WER/CER, TTFT/TTFA, or semantic scores. The GPU Fig.6 regeneration preserved all 24 bin-statistic rows exactly; only font embedding and SVG availability changed.
