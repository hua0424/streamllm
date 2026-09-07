# Speech Communication LaTeX submission package

This directory contains an English Original Full-length Research Article prepared using the local Speech Communication guide summary dated 2026-09-04. A fresh official-guide fetch on 2026-09-05 returned HTTP 403; current policy was not independently reverified. The 2026-09-05 revision clarifies manuscript claims but is not submission-ready: author placeholders, public artifact rights/availability and targeted visual re-verification remain unresolved.

## Build

Run from this directory with MiKTeX/TeX Live:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Or run the explicit sequence:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The package includes the official Elsevier `elsarticle` class and Harvard-style BibTeX style for portable compilation. The downloaded upstream bundle is preserved in `../template/`.

## Main files

- `main.tex`: manuscript entry point, abstract, title page, keywords, and Highlights.
- `sections/`: modular manuscript body.
- `references.bib`: author--year BibTeX database.
- `figures/`: separately named English figures.
- `highlights.txt`: mandatory separate Highlights file.
- `AUTHOR_CONFIRM.md`: unresolved author-controlled metadata and legal/administrative decisions.

## Guide-for-authors mapping

- Editable LaTeX source: provided.
- Concise factual abstract: 210 words by the repository check; no citations.
- English keywords: six.
- Highlights: five bullets, each no more than 85 characters including spaces.
- Numbered sections: provided; abstract is unnumbered.
- Author--year citations and alphabetic bibliography: `elsarticle-harv`.
- Editable tables and separately named figures: provided.
- Acknowledgements before references: provided as an author-confirmation placeholder.
- CRediT, funding, competing interests, data availability, ethics, and AI-use disclosure: structured placeholders provided and must be resolved by the authors.

## Scope and evidence boundary

The article is deliberately framed around spoken-dialogue barge-in and joint context-state repair. The accepted correctness claim is limited to direct crop integrity and within-run matched-arm recovery exactness under the frozen Qwen2-7B/Transformers/BF16/SDPA configuration. It does not claim clean-reprefill equivalence, device/acoustic playback truth, human-semantic effects, or production end-to-end latency.

Visual review update (2026-09-05): the user supplied a judge review of all 29 pages. Figure 2 label placement, Figure 3 connector routing, and Figure 6 main-analysis-only caption/prose were repaired. All other pages passed apart from author placeholders; repaired pages await targeted judge verification.
