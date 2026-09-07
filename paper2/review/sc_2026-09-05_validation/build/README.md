# Speech Communication LaTeX submission package

This directory contains an English Original Full-length Research Article prepared against the Speech Communication Guide for Authors accessed on 2026-09-04.

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
