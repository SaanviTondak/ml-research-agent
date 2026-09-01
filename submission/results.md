# Results

## Validation (agent-visible, used for selection)

| | GAUC | nDCG@5 | primary | vs baseline |
|---|---|---|---|---|
| FM baseline | 0.6674 | 0.5357 | 0.6016 | - |
| **agent, submitted (node #4)** | 0.6717 | 0.5370 | **0.6042** | +0.0026 |

Agent best verified over 3 seeds: 0=0.6043, 1=0.6045, 2=0.6038 (std 0.0004)

Selection rule: validation-best at stop, not the running peak.
Convergence rule: eps=0.002, N=3 (organizers').

## Held-out test (scored once, by seal/final_score.py)

| | primary |
|---|---|
| random | 0.4757 |
| item popularity | 0.5715 |
| FM baseline | 0.5946 |
| **this agent** | **0.5983** |
| oracle ceiling | 0.8645 |

**Delta vs baseline: +0.0037**  
Fraction of the headroom the baseline left on the table: +1.4%

Seeds: [0]. Candidate sha256 `80d5ff32a11c6ec3...`, evaluate.py sha256 `ecfde28392eb14fe...`.

## Every attempt

| # | stage | outcome | valid primary | hypothesis |
|---|---|---|---|---|
| 0 | draft | scored | 0.5986 | Within-user ranking performance (GAUC and nDCG@5) is degraded when training with pointwise binary cross-entropy because  |
| 1 | improve | scored | 0.5135 | Within-user ranking is significantly improved by incorporating recency-based user interaction history (last 3 positive v |
| 2 | improve | scored | 0.5984 | Within-user ranking is significantly improved by replacing single-pair BPR loss with a within-user InfoNCE (Sampled Soft |
| 3 | improve | scored | 0.5996 | Within-user ranking is significantly improved by replacing the pairwise BPR loss with a listwise Softmax Cross-Entropy l |
| 4 | improve | scored | 0.6042 | Within-user ranking is significantly improved by expanding the feature set with high-quality categorical video features  |
| 5 | improve | scored | 0.5908 | Within-user ranking is significantly improved by replacing the artificial 1-to-C listwise sampler with a principled List |
| 6 | improve | scored | 0.6038 | Within-user ranking is significantly improved by incorporating high-quality video statistic features (popularity, play r |
| 7 | improve | scored | 0.6046 | Within-user ranking is significantly improved by replacing the 1-pos-vs-C-neg listwise loss with an "all-impressions-per |
| 8 | improve | scored | 0.4665 | Within-user ranking is significantly improved by incorporating time-of-day (hour) and day-of-week features, as these var |
| 9 | draft | failed | - | Multi-task learning with is_click as an auxiliary task, sharing embeddings but using task-specific linear components, wi |
| 10 | improve | failed | - | Within-user ranking is significantly improved by incorporating "user favorite" features (most-interacted tag and author) |
| 11 | debug | failed | - | Within-user ranking is significantly improved by incorporating "user favorite" features (most-interacted tag and author) |
| 12 | debug | failed | - | The previous attempt was truncated. I will implement a compact Factorization Machine in NumPy, incorporating video-level |
| 13 | debug | failed | - | Within-user ranking is significantly improved by incorporating "user favorite" features (most-interacted tag and author) |
| 14 | debug | failed | - | Within-user ranking is significantly improved by incorporating personalized interaction features (is_fav_tag, is_fav_aut |
| 15 | debug | scored | 0.4791 | Within-user ranking is improved by adding video tags and user-favorite tags to the FM, allowing the model to capture spe |
| 16 | debug | failed | - | The previous attempt failed because the script was truncated. I will implement a complete, self-contained Factorization  |
| 17 | debug | failed | - | The previous script was cut off due to exceeding the token output limit and containing incomplete function definitions.  |
| 18 | debug | failed | - | The previous script was cut off due to exceeding the output token limit. I will write a concise, complete FM script with |
| 19 | debug | failed | - | The previous script was cut off due to exceeding the token output limit and getting too lengthy with excessive features. |
| 20 | debug | failed | - | The previous script was cut off due to exceeding the token output limit and containing too much redundant code. I will s |
| 21 | debug | scored | 0.5961 | The previous script was cut off due to exceeding the output token limit. I will compress the script to be concise and we |
| 22 | debug | failed | - | The previous script was cut off due to exceeding the token limit because of verbose code and redundant fields. I will wr |
| 23 | debug | failed | - | The previous script was cut off due to exceeding the output token limit. I will write a streamlined, self-contained vers |
| 24 | debug | failed | - | The previous script was cut off due to exceeding the output token limit. I will write a streamlined, self-contained vers |
| 25 | debug | failed | - | The previous script was cut off due to exceeding the token output limit and missing the main model training loop. I will |
| 26 | debug | failed | - | The previous script was cut off due to exceeding the output token limit. I will compress the script to be concise and we |
| 27 | debug | failed | - | The previous script was cut off due to exceeding the token limit. I will write a concise, complete Factorization Machine |
| 28 | debug | failed | - | The previous script was cut off due to exceeding the output token limit. I will write a streamlined, self-contained and  |
| 29 | debug | failed | - | The previous script was cut off due to exceeding the token limit. I will write a concise, complete script that adds vide |
| 30 | debug | failed | - | The previous script was cut off due to exceeding the output token limit. I will compress the code, use concise helper lo |
| 31 | debug | failed | - | The previous script was cut off due to exceeding the token limit because of verbose code and too many features. I will c |
| 32 | debug | failed | - | The previous script was cut off due to exceeding the output token limit. I will compress the implementation to ensure it |
| 33 | debug | scored | 0.4832 | The previous script was cut off due to exceeding the output token limit. I will write a streamlined, self-contained and  |
