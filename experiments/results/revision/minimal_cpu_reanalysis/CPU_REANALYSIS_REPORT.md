# Deterministic CPU Reanalysis

This is a deterministic reanalysis of locked numeric archives. It performs no ASR, LLM, TTS, CUDA, or audio processing and does not estimate the corrected trigger policy.

## Frozen method

- Metadata join: unique `sample_id -> (dataset, dialog_id)` over 1,133 processed JSON records.
- Bootstrap: dialogue-cluster resampling, 10,000 replicates, base seed 20260821, SHA-256-derived stable seed per comparison, percentile 95% CI.
- Point estimand: turn-weighted mean difference and ratio-of-means improvement; each sampled dialogue retains all its accumulated turns.
- Test: two-sided Wilcoxon on one mean difference per dialogue (`wilcox`, no continuity correction, `auto`), with Holm correction inside the named duration, ablation, and second-platform families.
- Difference direction is left minus right; positive values mean the right-hand system is faster.

## Sample ledger

| stage | samples | dialogues | change |
|---|---:|---:|---:|
| candidate | 505 | 99 | 0 |
| excluded_external_contamination | 7 | 7 | -7 |
| valid_three_arm | 498 | 99 | -7 |

Run-log review identified all seven excluded executions as contaminated by concurrent external programs. The locked 498-turn set is therefore the valid analysis cohort; contaminated values are retained only in the audit ledger and are not analyzed as system outcomes.

## Manuscript-ready results

- **Filtered primary A/B (498 turns, 99 dialogues):** A 4503.14 ms, B 1155.51 ms; difference 3347.63 ms [3075.50, 3589.85]; improvement 74.34% [72.62%, 75.76%]; dialogue-level Holm p=5.697221e-18.
- **Ablation arm contrasts (filtered):** baseline minus ASR-only 3332.13 ms [3064.38, 3572.61]; ASR-only minus full streaming 15.50 ms [10.95, 19.66]. These are order-confounded arm contrasts, not causal component effects.
- **Second platform:** A minus B 3736.83 ms [3438.69, 4000.96] (70.36% [68.50%, 71.90%]); configured LA-2-style minus B 541.11 ms [485.66, 599.14] (25.58% [23.47%, 27.73%]). The LA trigger policy is not matched to B.
- **R7 repeat-0 server-side policy comparison:** median TTFA is 22.27 s for A capped full response versus 3.11 s for B first sentence. Mean TTS text length is 204.10/17.74 characters; greeting-only outputs 0/50 versus 25/50; max-token caps 41/50 versus 43/50. For English inputs, 19/25 B outputs contain no ASCII Latin letters.

## Duration-group inference

| group | turns | dialogues | A-B mean [cluster 95% CI], ms | improvement [CI] | Holm p |
|---|---:|---:|---:|---:|---:|
| long | 108 | 89 | 603.12 [524.81, 680.94] | 35.67% [31.75%, 39.32%] | 1.783220e-15 |
| very_long | 150 | 93 | 2155.61 [2000.56, 2309.90] | 65.16% [62.79%, 67.33%] | 1.151233e-16 |
| extra_long | 240 | 95 | 5327.68 [5046.32, 5562.05] | 81.77% [80.61%, 82.75%] | 7.814927e-17 |

## Cluster structure

| cohort | samples | dialogues | cluster size min/median/mean/max |
|---|---:|---:|---:|
| ablation_candidate | 505 | 99 | 1/5.0/5.10/9 |
| ablation_excluded_external_contamination | 7 | 7 | 1/1.0/1.00/1 |
| ablation_valid_three_arm | 498 | 99 | 1/5.0/5.03/9 |
| second_platform_paired | 498 | 99 | 1/5.0/5.03/9 |
| ttfa_repeat0_unique_samples | 50 | 40 | 1/1.0/1.25/2 |

## Interpretation boundaries

- Results describe the historical latched implementation and locked execution order only; no corrected-trigger performance was measured.
- Accumulated turns from one source dialogue are dependent; cluster inference is primary here.
- Seven run-log-confirmed externally contaminated executions are excluded from inference; all reported effects use the locked 498-turn valid cohort.
- TTFA compares different response/TTS policies (B first sentence versus A capped full response) and excludes client playback; it is not an architecture-only effect.
- The ASCII-Latin output indicator is a deterministic descriptive proxy, not a language-identification or quality metric.

Software: Python 3.10.18; NumPy 1.26.4; SciPy 1.15.3.
