# Citation Audit Report — Final CPU-Reanalysis Revision

- **Target**: `paper/tougao_new/3次调整/latex/CISR-submission/submission2`
- **Files**: `main.tex`, `refs.bib`, `main.bbl`
- **Audit date**: 2026-08-22
- **Boundary**: citation and claim-scope repair plus CPU-only reanalysis; no ASR/LLM/TTS experiment rerun

## 1. Verdict

The citation layer is mechanically closed and the claim-to-source boundaries are consistent with the final systems-paper framing. Three direct primary sources were added after the final pre-submission review.

| Check | Result |
|---|---|
| Cited keys resolve | PASS |
| All BibTeX entries are cited | PASS |
| Duplicate keys, normalized titles, or DOI records | PASS |
| Bibliography compiles with official `IEEEtran.bst` | PASS |
| Citation numbering and first-use order | PASS |
| Primary-source attribution | PASS |
| Dynamic software/service identity | PASS with disclosed revision boundaries |
| Scientific implementation facts | Bound to source code/run records, not external citations |

## 2. Bibliography inventory

- **31 unique bibliography entries**.
- **31 unique citation keys used in `main.tex`**.
- No missing or uncited entry.
- 21/31 references are dated 2023–2026; 18/31 are dated 2024–2026.
- The list exceeds the stricter CISR local minimum of ten and covers substantially more than three countries.

## 3. Primary-source repairs retained

The prior revision replaced mismatched sources with direct papers for Conformer, PagedAttention, FlashAttention, StreamingLLM, Mini-Omni, Moshi, LLaMA-Omni, and Whisper large-v3-turbo configuration. It also corrected metadata for ACM article 209, MultiWOZ venue location, and protected `Whisper`, `Mandarin`, and `M3-Embedding` capitalization.

The final review added:

- `wang2024simulwhisper` — Simul-Whisper, Interspeech 2024, DOI `10.21437/Interspeech.2024-1814`;
- `gim2024promptcache` — Prompt Cache, MLSys 2024, vol. 6, pp. 325–338;
- `zheng2024sglang` — SGLang/RadixAttention, NeurIPS 2024, DOI `10.52202/079017-2000`.

The manuscript uses them narrowly:

- Simul-Whisper supports attention-guided streaming Whisper with truncation detection;
- Prompt Cache supports reusable prompt-module attention/KV state;
- SGLang supports automatic shared-prefix KV reuse through RadixAttention.

It explicitly distinguishes these mechanisms from the within-request fragment appends evaluated here.

## 4. Evidence-scope corrections

The latest changes are not citation substitutions; they are source-code and archived-record corrections:

- the 2.0-s ASR condition is a historical cumulative-seen-duration startup gate that latches because removed segments do not decrement the counter;
- legacy values are post-feed residual TTFT, not physical-speech-end TTFT;
- the configured LA comparator matches weights, engine, segmenter, hardware, and subset, but not trigger policy;
- the 22.27-s versus 3.11-s result is a server-side B-first-sentence versus A-capped-full-response policy comparison;
- semantic scores do not establish quality, task, or usability equivalence;
- the locked 498-turn valid set is used after excluding seven run-log-confirmed externally contaminated executions.

These statements are grounded in implementation, Git history, and locked result records; no external citation is used to disguise an internal measurement fact.

## 5. Dynamic artifacts and reproducibility boundaries

- **Historical ASR implementation**: the latching behavior exists from commit `3ee6157` and in the R7 recorded commit `c9437c3`; no corrected-trigger performance is inferred.
- **Silero VAD**: artifact hash is archived for R7, but experiment-era Git revision is unavailable.
- **Whisper large-v3-turbo**: public configuration is cited; experiment-era repository revision is unavailable.
- **Qwen2-7B-Instruct**: exact model identifier is reported; immutable repository revision was not archived.
- **CosyVoice**: model-family paper is cited; endpoint/checkpoint revision was not archived.
- **BGE-M3**: ACL paper is cited; local artifact was not a versioned Hugging Face snapshot.
- **Judge**: exact Command Code service identifier/access period are reported; no immutable backend build ID exists.

## 6. Official IEEEtran behavior

The official CISR `IEEEtran.bst` remains byte-identical to the template and does not render `doi`, `articleno`, `eprint`, or `archivePrefix`. DOI fields remain database metadata. CISR does not require visible DOI strings, so the style was not replaced.

## 7. Build verification

The final source completes:

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Checks:

- 31 cited keys / 31 BibTeX entries;
- no missing or orphan keys;
- no undefined citations or references;
- no duplicate key, DOI, or normalized title;
- all figures and Tables I–VIII referenced;
- equation references use `\eqref`;
- no overfull box.

## 8. Fig.6 and remaining external gates

Fig.6 was relabeled from the unchanged 24-row bins file:

- PDF SHA-256: `d38927c9b25139268b2fb00003f573184a0250ede1d502638a1f57d25d573cbb`;
- SVG SHA-256: `ca9d3fd1efcb22a37cd23fc21a78e54bca55b4a752d6ce8edc7631fbcadb91b2`;
- manuscript copy is byte-identical to the experiment PDF;
- PDF contains no raster image and embeds CID TrueType;
- Type 3 count remains zero.

External gates remain similarity checking and IEEE PDF eXpress on the final PDF.
