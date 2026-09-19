# Generalizing the agent, and what calibration found on the way

The agent ran on exactly one benchmark. The interesting findings it produced —
a repair chain with no depth limit absorbing 68% of a budget, a greedy argmax
unable to develop a direction that lost inside the noise floor — are claims
about autonomous search under measurement noise, not about recommendation. They
were demonstrated on n=1 task.

This is the work that makes them portable, and one measurement made while doing
it that matters on the original benchmark too.

## The structure

`agent/` is now task-agnostic. A task is a directory implementing
[`agent/task.py`](../agent/task.py); the interface is not invented, it is
exactly the set of calls `agent/loop.py` already made into `agent.prompts` and
`agent.scorer`:

```
prompts.SYSTEM              -> Task.system_prompt()
prompts.draft_prompt(...)   -> Task.draft_prompt(...)
scorer.score_file(...)      -> Task.validate_and_score(...)
scorer.assert_evaluate_untouched() -> Task.integrity_check()
```

Two things deliberately do not vary per task, because they are the seam that
keeps the loop simple: the **candidate contract** (a standalone script taking
`--data_dir --split --out --seed`, writing predictions aligned row-for-row with
the evaluation set) and the **scalar objective** (one comparable number per
node).

| task | metric | direction | groups | submission | noise floor |
|---|---|---|---|---|---|
| [`tasks/kuairand/`](../tasks/kuairand/) | mean(GAUC, nDCG@5) | maximise | per-user | 4 columns | a property |
| [`tasks/synthetic/`](../tasks/synthetic/) | RMSE | **minimise** | none | 2 columns | a **knob** |

The synthetic task is not a toy stand-in for a real one. It is chosen for what
it can prove: its noise floor is a construction parameter, so calibration can be
checked against ground truth instead of against another estimate. It also needs
no dataset and no API key, which means CI can now run the **whole loop** end to
end — previously every test exercised policy objects in isolation, and the only
end-to-end check needed the real dataset.

## Why calibration is the substance, not the plumbing

Every policy constant was denominated in one benchmark's noise floor.
`VERIFY_MARGIN = 0.0008` *was* KuaiRand's seed sigma; `EPS = 0.002` is a
multiple of it; `PROTECTION_WRITE_OFF` inherits it. Point that agent at a task
whose sigma is 0.05 and it verifies on every node and never converges.
Portability without calibration is portability on paper.

[`agent/calibrate.py`](../agent/calibrate.py) runs the task's reference
implementation on k seeds before the search starts, measures the noise, and
derives the thresholds from it.

### Two noise sources, two different questions

This is the part that was wrong on the original benchmark too. A measured score
is `S = mu(design) + T(design, seed) + E(design, eval set)`:

- **sigma_seed** — re-run the same design with a different seed; this is how
  much it moves.
- **sigma_eval** — the validation set is one draw; a different draw would rank
  designs slightly differently.

The policy asks two different questions and they key off different sources:

| question | site | scale |
|---|---|---|
| is this single-seed result worth more seeds? | `needs_verification` | **sigma_seed** — both numbers come from the same eval rows, so that draw is a common offset that largely cancels |
| is the search still progressing, or climbing this particular validation sample? | `has_converged` | **sigma_delta** = sqrt(2·sigma_seed² + sigma_eval_paired²) |

`agent/state.py` already argued that EPS and VERIFY_MARGIN answer different
questions. It just keyed both to the same variance.

### What it measured on KuaiRand

Ten reference fits (30.6 s each) plus a bootstrap that re-aggregates stored
predictions over the 22,377 validation users — no retraining, under a second:

```
sigma_seed         0.000627     (pooled from the recorded runs: 0.00076)
sigma_eval_paired  0.000700
sigma_eval_abs     0.002000
sigma_delta        0.001300
```

**Evaluation noise is the same size as seed noise, and was invisible to the
policy.** A validation primary is pinned only to about ±0.002 in absolute
terms, and a one-seed-vs-one-seed comparison between two designs carries
sigma_delta ≈ 0.0013 — roughly twice the quantity every threshold was keyed to.

### The derivation regenerates the hand-tuned constants

Fed only those measurements, with multipliers chosen from the argument in
`agent/calibrate.py` rather than from the answers:

| constant | hand-tuned | derived from measurement | agreement |
|---|---|---|---|
| `VERIFY_MARGIN` | 0.0008 | **0.000774** | 3.3% |
| `EPS` | 0.002 (organizers') | **0.00195** | 2.5% |
| `PROTECTION_WRITE_OFF` | 0.006 | **0.006000** | exact |

The organizers' mandated epsilon works out to **1.54 sigma_delta**, against a
multiplier of 1.5 chosen independently. That three constants tuned by hand
against two real runs fall out of a derivation that never saw them is the only
external check available short of many tasks, and it is the reason to believe
the scheme.

**A declared epsilon is honoured, never overruled.** Where a competition states
its stopping rule, calibration reports how many sigma it corresponds to and
leaves it alone — silently substituting a derived rule would invalidate the
result it produced.

### The degenerate cases, which are the ones that bite

| case | what happens |
|---|---|
| deterministic task (sigma_seed exactly 0) | thresholds stay finite via the floor; **seed verification is switched off entirely**, since re-running returns the identical number and buys nothing. Only a measured noise floor can know that. |
| discrete metric (accuracy over 500 rows) | no threshold is ever finer than one grid step |
| reference missing, broken, or too slow | calibration skips, the declared constants stay in force, and the run log says so |
| paired bootstrap unmeasurable | on a deterministic task it compares two identical models and reads zero — an artifact of what was compared. Flagged `eps_provisional` rather than trusted. |

Nothing divides by sigma anywhere, so sigma = 0 is a supported case rather than
a crash.

## What is verified

- **89 tests**, including the 9 guarding the original debug trap, which pass
  **unmodified** through all of this.
- Replaying both recorded runs through the refactored policy produces a
  **byte-identical** decision trace: 20 differing selections on `final_01`,
  0 on `final_02`, exactly as before the refactor.
- `harness_check.py` still 9/9, baseline still reproducing at 0.6015.
- The run log regenerates byte-identically from `final_02`'s recorded journal.

**Not verified:** whether the generalized agent is any *good* on a second real
task. That needs a Kaggle-style benchmark and a real run. The claim here is
portability and calibration, not performance.

## The honest limits

- **The reference is one point in design space.** sigma measured at the
  baseline is not sigma at whatever the agent writes on iteration 30. The right
  fix is to pool every verified node's spread online and recalibrate; that is
  not implemented, and each node would need stamping with the policy version it
  was judged under so replays stay reproducible.
- **sigma_split is unmeasured.** The train/valid partition is a constant offset
  shared by every candidate, so it cannot affect within-run ranking and no
  policy constant can reduce it. It is the term that makes all of this
  optimistic.
- **A bug found by the second task, not by review.** `agent/loop.py` printed
  `s.gauc` after scoring — invisible on KuaiRand, fatal anywhere else. The
  end-to-end loop test on the synthetic task is what caught it, which is a
  reasonable argument for having a second task at all.
