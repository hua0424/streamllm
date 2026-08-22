# Citation Audit Report — Post-Repair Status

- **Target**: `paper/tougao_new/3次调整/latex/CISR-submission/submission2`
- **Files**: `main.tex`, `refs.bib`, `main.bbl`
- **Audit date**: 2026-08-22
- **Status**: citation repair completed; no manuscript experiment was rerun

## 1. Final verdict

The citation layer is now mechanically closed and substantively improved with primary sources. The previously reported high-priority citation mismatches have been repaired.

| Check | Result |
|---|---|
| Cited keys resolve | PASS |
| All BibTeX entries are cited | PASS |
| Duplicate keys, normalized titles, or DOI records | PASS |
| Bibliography compiles with official `IEEEtran.bst` | PASS |
| Citation numbering and first-use order | PASS |
| Primary-source attribution | PASS |
| Named systems have direct references | PASS |
| Dataset/model/software identity citations | PASS with disclosed revision boundaries |
| Current citation readiness | **READY; Fig.6 font repair verified, IEEE PDF eXpress still pending** |

## 2. Current bibliography inventory

- **28 unique bibliography entries**.
- **28 unique citation keys used in `main.tex`**.
- No missing or uncited entries.
- The reference set exceeds CISR's stricter local minimum of 10 references.
- 18/28 references are dated 2023–2026; 15/28 are dated 2024–2026.
- Author/institutional coverage spans substantially more than the required three countries.

## 3. Repairs applied

### 3.1 Removed mismatched or superseded entries

The following entries were removed after their only uses were replaced by primary sources:

- `ref2` — rVAD, previously attached to a Silero/modular-cascade statement;
- `ref5` — ASR survey, previously used for Conformer attribution;
- `ref8` — model-compression survey, previously used for vLLM/TGI/KV-cache claims;
- `ref9` — efficient-Transformer survey, previously used as the FlashAttention source;
- `ref12` — self-supervised speech review, previously used for Whisper-specific robustness.

### 3.2 Added primary sources

- `gulati2020conformer` — Conformer, Interspeech 2020;
- `kwon2023pagedattention` — vLLM/PagedAttention, SOSP 2023;
- `dao2022flashattention` — FlashAttention, NeurIPS 2022;
- `xiao2024streamingllm` — StreamingLLM, ICLR 2024;
- `xie2024miniomni` — Mini-Omni;
- `defossez2024moshi` — Moshi;
- `fang2025llamaomni` — LLaMA-Omni, ICLR 2025;
- `openai2024whisperlargev3turbo` — official pinned public model configuration for the checkpoint-family-specific 128-Mel-bin statement.

### 3.3 Corrected existing metadata

- `ref7`: expanded the full six-author list; replaced the incorrect `pages = {209}` with pages 1–35 and a rendered `note = {Art. no. 209}`.
- `ref14`: changed the EMNLP 2018 venue address from Stroudsburg to Brussels, Belgium and aligned the title with ACL Anthology metadata.
- `bu2017aishell1`: protected `{Mandarin}` capitalization and added the official O-COCOSDA acronym.
- `machacek2023turning`: protected `{Whisper}` capitalization.
- `chen2024m3embedding`: protected `{M3-Embedding}` capitalization.
- `commandcode2026deepseekv4flash`: clarified that the source is a Command Code service catalog, not an official DeepSeek technical report.

## 4. Claim-to-source corrections

- Removed the unsupported claim that most production systems use cascades; the Introduction now scopes the serial architecture to the system evaluated in this paper.
- Restricted the GPT-4o system-card citation to its reported 232/320-ms latency figures.
- Removed the rVAD/Silero association; Silero is cited at the implementation description.
- Replaced the Conformer survey attribution with the original paper.
- Replaced broad vLLM/FlashAttention/StreamingLLM survey citations with primary papers.
- Added direct references for Mini-Omni, Moshi, and LLaMA-Omni and changed the collective wording from “open-source systems” to “representative systems.”
- Replaced the incorrect Whisper review citation with the primary Whisper paper and narrowed the robustness statement to reported benchmarks.
- Added LocalAgreement/partial-hypothesis citations at the chunk-boundary instability claim.
- Recast the CosyVoice statement as a CosyVoice-based HTTP endpoint; the paper is cited as the model-family source, while the unavailable deployment revision is disclosed.
- Clarified that BGE-M3, Qwen2, Whisper, CosyVoice, and judge revisions that were not archived are unavailable rather than retrospectively inferred.

## 5. Official IEEEtran bibliography behavior

The CISR-provided `IEEEtran.bst` is unchanged and byte-identical to the downloaded template. It does not render `doi`, `articleno`, `eprint`, or `archivePrefix`. Therefore:

- DOI fields remain valid database metadata but do not appear in `main.bbl` or the PDF.
- CISR's website and local instructions do not require DOI display.
- The official `.bst` was not replaced or modified.
- ACM article identifiers that need visible output use the style-supported `note` field.

## 6. Dynamic software/service boundaries

- **Silero VAD**: official repository citation; no experiment-era immutable Git revision was archived.
- **Whisper large-v3-turbo**: official public configuration is cited for 128 Mel bins, but the experiment-era repository revision is explicitly unavailable.
- **Qwen2-7B-Instruct**: exact model identifier is reported; immutable repository revision was not archived.
- **CosyVoice**: model-family paper is cited; immutable endpoint/checkpoint revision was not archived.
- **BGE-M3**: ACL paper is cited; local artifact was not a versioned Hugging Face snapshot.
- **Judge**: exact Command Code service identifier and access period are reported; no immutable backend build ID was available.

These are disclosed reproducibility boundaries, not fabricated repairs.

## 7. Verification commands/results

The current manuscript successfully completed:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Post-build checks show:

- 28 cited keys / 28 BibTeX entries;
- no missing keys;
- no orphan entries;
- no undefined citations or references;
- no missing figure/table labels;
- all figures and Tables I–VIII explicitly referenced;
- no legacy `\ref{eq:...}` equation references; the manuscript uses `\eqref`.

## 8. Figure-font closure and remaining external gate

The GPU-regenerated Fig.6 has been pulled and verified:

- PDF SHA-256: `a67be51b9670b64c0adfa8f36368d0b033d98d7c4b5a4ca3c2851a11d0535828`;
- SVG SHA-256: `b9fd3d8383e4c322ed48285f766b32ba7df2ed4aba54fdeb5ff5d7b51dfbc2df`;
- PDF and SVG dimensions: 446.4 × 259.2 pt;
- all 24 bin-statistic rows are field-identical to the previous version;
- PDF contains no raster images and embeds DejaVuSans as CID TrueType with Unicode mapping;
- PDF and path-based SVG renderings are visually equivalent; mean absolute pixel difference is below 1/255 per channel and reflects renderer anti-aliasing;
- rebuilt `main.pdf` contains zero Type 3 fonts and all fonts are embedded.

The remaining non-citation gate is IEEE PDF eXpress, plus the separate CISR similarity check.
