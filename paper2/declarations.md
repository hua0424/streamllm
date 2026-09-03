# Paper 2 declarations draft

> Submission-support document only. This is not paper chapter text and is not ready to paste into a submission until every `AUTHOR CONFIRM` item has been resolved. Repository evidence is stated factually; identities, legal status, approvals, and relationships are not inferred.

## Data availability — draft

The accepted Paper 2 experiment records, manifests, analyses, validation outputs, available execution logs, and hardware/environment snapshots are archived in this repository under `experiments/sci34_supplement/results/`. The authoritative accepted/rejected campaign index, exact run IDs, commits, hashes, reproduction entrypoints, and claim boundaries are provided in [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

The accepted E3 processed input is preserved at `experiments/sci34_supplement/results/e3_exact_rescue/p2_turns.json` with SHA-256 `a2116b83b38509e45d641ade17ae1791282729836bdca251aa8aba8aa9248a0c`; its relation to E3 and portable verification commands are documented in that directory. It was derived deterministically from MultiWOZ 2.1 with the repository builder and seed 42. The raw MultiWOZ source is not redistributed. Users must obtain it from its authorized source and comply with its terms.

Model weights are not redistributed. Campaign manifests record the local snapshot identities for Qwen2-7B-Instruct, Mistral-7B-Instruct-v0.3, and TEN_Turn_Detection so separately obtained assets can be checked. Third-party datasets, model weights, tokenizers, and software remain subject to their own licenses and access terms.

**AUTHOR CONFIRM — public data URL/DOI:** `[insert immutable public archive URL/DOI, release tag, and access date]`

**AUTHOR CONFIRM — processed-data redistribution:** `[confirm that distributing the derived E3 p2_turns.json is permitted; add required MultiWOZ attribution/notice or remove it from the public package]`

**AUTHOR CONFIRM — retention/access exceptions:** `[state any repository files that cannot be public and the factual reason]`

## Code availability — draft

The research code and experiment harnesses are version controlled on the `paper2` branch. Campaign manifests bind formal runs to clean code commits; the accepted campaign matrix in [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) identifies those commits and result commits. Dependencies are recorded in `pyproject.toml` and `uv.lock`; formal reproduction must checkout the campaign-specific commit and use `uv sync --frozen`. Accepted result artifacts must not be overwritten during analysis-only verification.

The repository currently has no root `LICENSE`. No license should be claimed until the legal rights holder(s) and license choice are confirmed. This is a remaining submission/release blocker. Third-party components and assets are not relicensed by this repository.

**AUTHOR CONFIRM — public code URL/DOI:** `[insert immutable public repository/archive URL, release tag or commit, DOI if any, and access date]`

**AUTHOR CONFIRM — repository license:** `[identify rights holder(s), approved license, copyright notice, and whether institutional approval is required]`

**AUTHOR CONFIRM — third-party notices:** `[confirm all required notices and compatibility for bundled code/data; do not treat model or dataset licenses as the repository license]`

## Ethics and consent — draft placeholder

The repository documents experiments on MultiWOZ-derived text and local model inference; it does not contain a repository-grounded statement establishing whether institutional ethics review, exemption, informed consent, or participant recruitment applied. No status is inferred here.

**AUTHOR CONFIRM — ethics review:** `[state approving/exempting institution, protocol/reference number, decision, and date, or provide a justified “not applicable” statement consistent with venue policy]`

**AUTHOR CONFIRM — human participants and consent:** `[state whether any human participants or annotators were involved and, if so, the consent/compensation/privacy arrangements]`

**AUTHOR CONFIRM — automated judge disclosure:** `[confirm wording that E3 used a Mistral-7B single-model proxy and no human double annotation, unless additional evidence exists outside this repository]`

## Funding — draft placeholder

**AUTHOR CONFIRM — funding:** `[list funder names and grant numbers, and describe the funders’ role; if none, explicitly confirm “This research received no external funding.”]`

## Competing interests / conflict of interest — draft placeholder

**AUTHOR CONFIRM — COI:** `[disclose relevant financial/non-financial competing interests for every author, or explicitly confirm that the authors declare no competing interests]`

## Author contributions (CRediT) — draft placeholder

Do not infer author identities or roles from Git metadata or agent-generated commits.

**AUTHOR CONFIRM — author list and CRediT roles:**

- `[Author legal/publishing name]`: `[Conceptualization; Methodology; Software; Validation; Formal analysis; Investigation; Data curation; Writing – original draft; Writing – review & editing; Visualization; Supervision; Project administration; Funding acquisition — retain only confirmed roles]`
- `[Add one line per author]`

**AUTHOR CONFIRM — accountability:** `[identify the corresponding author and confirm that all authors approved the submitted version and accept accountability under venue policy]`

## Generative-AI / automated-tool disclosure — draft placeholder

The repository history and documentation indicate use of coding agents in experiment execution/documentation, but the repository alone cannot establish the complete scope required by a target venue’s disclosure policy.

**AUTHOR CONFIRM — disclosure:** `[describe any generative-AI or automated tools used for code, analysis, language editing, figures, or manuscript preparation; name tools/versions where required; state human verification and responsibility]`

## Final pre-submission checklist

- [ ] Public repository/archive URL and immutable version confirmed.
- [ ] Repository license and legal rights holder confirmed; root `LICENSE` added only after approval.
- [ ] Third-party data/model/software notices reviewed.
- [ ] E3 derived-input redistribution confirmed or package adjusted.
- [ ] Ethics/exemption and consent/participant statements confirmed.
- [ ] Funding and grant numbers confirmed.
- [ ] COI statement confirmed for every author.
- [ ] Author names, order, corresponding author, and CRediT roles confirmed.
- [ ] Generative-AI/tool disclosure matched to target venue policy.
- [ ] Data/code availability wording updated with final URLs and dates.
