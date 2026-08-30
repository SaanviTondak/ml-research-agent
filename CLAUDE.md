# TikTok Hackathon — Track 2: Autonomous ML Research Agent

Solo entry (Saanvi). Required benchmark: **KuaiRand-Pure**. Started 30 Aug 2026.

## What we are building

An autonomous ML research agent that runs the MLE loop on its own — read the
problem, EDA, engineer features, train + tune, evaluate, reflect, repeat — and
drives the validation score above the organizer's Factorization Machine baseline.

**The agent is the deliverable, not the model.** Grade weights:

| criterion | weight | what it actually measures |
|---|---|---|
| Technical Execution | 35% | hidden-test score delta vs. baseline + failure recovery |
| Innovation & Insight | 20% | what the agent chose to try, and its reasoning |
| Impact — Autonomy | 20% | **number of manual interventions** (fewer is better) |
| Feasibility | 15% | LLM tokens + agent wall-clock; only scored if we beat baseline |
| Presentation | 10% | final event only |

Only 35% rewards a better model. Build the loop first.

## Hard numbers (do not re-derive, these are verified)

Reproduced in Phase 0 — see `docs/phase0_baseline_repro.md`.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (test) | 0.4999 | 0.4514 | 0.4757 |
| item popularity (test) | 0.6308 | 0.5121 | 0.5715 |
| **FM baseline — valid** | 0.6674 | 0.5357 | **0.6016** |
| **FM baseline — test (beat this)** | 0.6610 | 0.5282 | **0.5946** |
| oracle ceiling (test) | 1.0000 | 0.7289 | 0.8645 |

Judge progress against **0.8645**, not 1.0. 27.1% of test users are all-negative
(nDCG always 0) and 9.2% all-positive. Baseline already captures ~31% of the
attainable range. **+0.02 is a real result; +0.05 is excellent.**

Seed std is **0.0008**. Any single-seed gain under 0.002 is noise — re-run
candidates on multiple seeds before promoting them.

Splits (date-based, fixed): train 20220408–0421 (1,141,112 rows) /
valid 20220422–0428 (124,909) / test 20220429–0508 (170,588).

Convergence rule (fixed): **eps = 0.002, N = 3**. Caps: **50 iterations, 6 h**.
A full FM run is ~18 s on one CPU core, so compute is not the binding constraint.

## Non-negotiable rules

1. **Test-split firewall.** The test labels ship in the public dataset and
   `baseline.py:run_fm` reads them. The agent's data loader must not be able to
   return the test split at all. Final test scoring happens once, from a separate
   sealed script. Document this in the README — it is a scoring asset, not just
   hygiene.
2. **Submit the validation-best checkpoint at convergence**, not the peak.
3. **Run log is emitted live by the loop**, never retrofitted. Per iteration:
   hypothesis, code diff, resulting GAUC/nDCG@5, error and recovery events.
   It is a graded deliverable.
4. **Do not rescue the agent by hand.** Every intervention costs autonomy points.
   Restarting a crashed process does not count as one (organizer Q&A).
5. **No external training data.** KuaiRand only. Papers, libraries and public
   solutions are all in scope.

## Where the headroom is

Organizers measured these as **dead ends** — don't spend iterations there:

- All 13 feature fields instead of 5: 0.5940 vs 0.5950. Noise.
- Embedding dim k = 8/16/32: 0.5895 / 0.5902 / 0.5887. Flat.
- Pure user-side first-order terms contribute **exactly zero** — ranking is
  within-user, so any within-user constant cancels out of the ordering. User
  features only act through crosses with the item side.

Open directions, organizer-untested, roughly in priority order:

1. **Loss function.** Baseline optimises pointwise logloss but is scored on
   ranking metrics. Pairwise BPR or within-user listwise softmax. Our own Phase 0
   trace is direct evidence: from epoch 7 to 11, train logloss fell 0.4859 →
   0.4705 while valid primary fell 0.6015 → 0.5990. The objectives come apart.
2. **User behaviour sequences.** Entirely unused. DIN / SIM interest modelling.
3. **Multi-task** on `is_click`, `is_like`, `play_time_ms` as auxiliary heads.
4. **Censored watch-time regression** (CWM) — watch time is truncated at video end.
5. **Unbiased validation** — `log_random_4_22_to_5_08_pure.csv` (1.18M randomized-
   exposure rows) as a second validation signal against biased-traffic overfitting.
6. Deeper models (DeepFM / DCN / xDeepFM) — after the above; capacity is not the limit.

## Plan

- **Phase 0 — DONE.** Baseline ladder reproduced, repo initialised, `docs/phase0_baseline_repro.md` committed.
- **Phase 1 — build the harness before any modelling.** `state.py` (solution journal),
  `llm.py` (engine + token counting), `executor.py` (subprocess, timeout, capture),
  `loop.py` (draft → improve → debug, AIDE-style greedy on best node), `journal.jsonl`
  + markdown renderer. Candidate contract: a solution script writes `scores_valid.csv`
  in submission schema; the untouched `evaluate.py` scores it. Keep that interface narrow.
- **Phase 2 — seed context, not answers.** Feed the agent the starter-kit README,
  `data.py`, `evaluate.py`, plus paper abstracts (BPR, LambdaRank/listwise softmax,
  DIN, SIM, ESMM, MMoE/PLE, DCN, CWM). Let it rank and choose; log the ranking —
  that ranking is the Innovation score. Give it an EDA step it runs itself.
- **Phase 3 — smoke test, then hands off.** Prove recovery on a deliberately injected
  syntax error and timeout, and keep those log entries. Cap per-iteration training time.
  Then start the real run and don't touch it.
- **Phase 4 — package.** submission.csv (validated with `submit.py --check`), results
  table with absolute deltas, resource report (tokens / wall-clock / iterations of 50 /
  GPU-hours 0), run log + intervention count, README, Devpost writeup. No video —
  not required, and a detailed report is the accepted alternative.

## Environment notes

- Dataset at `kuairand-starter-kit/KuaiRand-Pure/data/`. Gitignored — never commit it.
- Zenodo and kuairand.com are blocked from sandboxed environments; the data was
  downloaded manually. PyPI works.
- The folder name has a **trailing space** (`pingu `). Quote every path.
- `evaluate.py` is model-agnostic: `evaluate(user_ids, labels, scores)`. Never modify it.
