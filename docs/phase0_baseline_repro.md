# Phase 0 — Official baseline reproduction

Task Requirement #1: stand up the end-to-end pipeline and confirm it reaches the
official baseline's reported score.

Environment: Python 3.10.12, numpy 2.2.6, single CPU core, no GPU.
Dataset: KuaiRand-Pure (Zenodo record 10439422), unmodified.
Command: `python3 baseline.py --model {random,pop,fm}` (seed 0, all defaults).

## Split integrity

Row counts match the organizer's published splits exactly:

| split | dates | rows (ours) | rows (published) |
|---|---|---|---|
| train | 20220408–20220421 | 1,141,112 | 1,141,112 |
| valid | 20220422–20220428 |   124,909 |   124,909 |
| test  | 20220429–20220508 |   170,588 |   170,588 |

## Baseline ladder — reproduced vs. published

Published figures are means over seeds 0–4 (test primary std = 0.0008);
ours are single-seed (seed 0), so agreement is expected within ~1–2 sigma.

| model | split | GAUC | nDCG@5 | primary | published primary | delta |
|---|---|---|---|---|---|---|
| random | valid | 0.4990 | 0.4663 | 0.4827 | 0.4834 | -0.0007 |
| random | test  | 0.4999 | 0.4514 | 0.4757 | 0.4753 | +0.0004 |
| item popularity | valid | 0.6387 | 0.5227 | 0.5807 | 0.5807 | 0.0000 |
| item popularity | test  | 0.6308 | 0.5121 | 0.5715 | 0.5715 | 0.0000 |
| **FM (official baseline)** | valid | 0.6671 | 0.5358 | **0.6015** | 0.6016 | -0.0001 |
| **FM (official baseline)** | test  | 0.6621 | 0.5286 | **0.5953** | 0.5946 | +0.0007 |

Harness self-check passes: `--model random` gives test primary 0.4757, inside the
0.4753 +/- 0.001 tolerance the starter kit specifies. The evaluation code is sound.

Item popularity reproduces to four decimal places on both splits (it is
deterministic — no training, no seed).

## FM training trace (seed 0)

Early-stopped at epoch 11 on validation primary, patience 4; best epoch was 7.

| epoch | train logloss | valid primary |
|---|---|---|
| 1 | 0.6391 | 0.5869 |
| 3 | 0.5129 | 0.5993 |
| 5 | 0.4941 | 0.6010 |
| **7** | 0.4859 | **0.6015** (best) |
| 9 | 0.4784 | 0.6007 |
| 11 | 0.4705 | 0.5990 |

Note the divergence from epoch 7 onward: training logloss keeps falling while
validation primary falls too. The pointwise objective and the ranking metric come
apart once the model starts fitting the classification task well. This is direct
evidence for the loss-function hypothesis being the first thing worth testing.

## Targets to beat

Reference targets for the agent, taken from `baseline_scores.json`:

- Validation (agent-visible): GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016
- Hidden test (scored once):  GAUC 0.6610 / nDCG@5 0.5282 / primary 0.5946
- Oracle ceiling on test:     GAUC 1.0000 / nDCG@5 0.7289 / primary 0.8645

Convergence rule (fixed by organizers): eps = 0.002, N = 3.
Caps: 50 iterations, 6 h wall-clock.

## Wall-clock

A full FM run — data load plus 11 epochs plus evaluation on both splits — takes
about 18 s on one CPU core. Compute is not the binding constraint on this
benchmark; a 50-iteration run at baseline cost is well inside the 6 h ceiling.

## Note on test-set access

`baseline.py:run_fm` evaluates on the test split and prints the score, and the
test labels ship in the public dataset. The hidden test set is therefore hidden by
convention only. Phase 1 removes the test split from the agent's data loader
entirely; final test scoring is performed once, by a separate sealed script.
