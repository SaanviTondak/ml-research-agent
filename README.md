# An autonomous ML research agent for KuaiRand-Pure

TikTok TechJam — **Track 2: Autonomous ML Research Agent**. Solo entry.

An agent that runs the machine-learning research loop by itself: read the
problem, explore the data, form a hypothesis, write the code, train, evaluate,
read its own failures, and try again — without a human in the loop. The target
is the organizer's Factorization Machine baseline on within-user ranking.

**The agent is the deliverable, not the model.** A model that scores well
because a human tuned it is worth less here than a loop that recovers from its
own crashes and reasons its way to a smaller gain.

---

## Status

| phase | state |
|---|---|
| **0 — reproduce the official baseline** | done — [`docs/phase0_baseline_repro.md`](docs/phase0_baseline_repro.md) |
| **1 — test-split firewall + candidate harness** | done — 9/9 checks pass |
| 2 — seed context, agent-run EDA | not started |
| 3 — recovery smoke test, then the autonomous run | not started |
| 4 — package and submit | not started |

The autonomous run **has not been started**. Everything below describes
infrastructure that is built and verified, not results that have been achieved.
Manual interventions to date: **0** — the count begins when the loop starts, and
is tracked honestly in [`docs/interventions.md`](docs/interventions.md).

---

## The task

For each user, order the videos they were actually shown. Relevance is the
native `long_view` column. Scoring is `mean(GAUC, nDCG@5)` over within-user
rankings, defined entirely by the organizer's [`evaluate.py`](kuairand-starter-kit/evaluate.py),
which is never modified.

Two properties dominate the design:

**The lists are short.** A typical user has 4–5 impressions in the evaluation
window; 55% of validation users have fewer than five. `nDCG@5` is therefore
close to full-list nDCG for most users, and a listwise objective over a user's
actual impression list is both metric-aligned and computationally free.

**A third of users are unrankable.** 27.1% of test users viewed nothing for
long and 9.2% viewed everything — no ordering changes their score, and GAUC
excludes them entirely. A model that cheats by reading the labels scores
**0.8645**, not 1.0. Progress should be judged against that ceiling.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| random (sanity floor) | 0.4999 | 0.4514 | 0.4757 |
| item popularity | 0.6308 | 0.5121 | 0.5715 |
| **FM baseline — the target** | **0.6610** | **0.5282** | **0.5946** |
| oracle ceiling | 1.0000 | 0.7289 | 0.8645 |

*Test split. The baseline has already captured ~31% of the reachable range, so
the remaining headroom is 0.27, not 0.41. Seed-to-seed std is 0.0008 — any
single-seed gain under 0.002 is noise.*

---

## The test-split firewall

**KuaiRand-Pure ships the test labels in the public download, and the
organizer's `baseline.py` reads them.** The hidden test set is hidden by
convention only.

For a human entrant that is a mild hazard. For an autonomous agent writing and
running its own code across 50 iterations it is fatal: one candidate that
selects on test — deliberately, or by copying the pattern out of `baseline.py` —
silently invalidates every number the project reports, and nothing about the run
would look wrong.

So the constraint is structural, not procedural.
[`agent/firewall.py`](agent/firewall.py) materialises `work/data_visible/`, a
data directory that **physically contains no impression dated after 20220428**,
the last day of the validation window:

```
log_standard_4_08_to_4_21_pure.csv   kept 1,141,112   dropped       0
log_standard_4_22_to_5_08_pure.csv   kept   124,909   dropped 170,588
log_random_4_22_to_5_08_pure.csv     kept   288,338   dropped 897,721
```

The row counts match the organizer's published split sizes exactly, which is
itself a check that the date filter is right.

The directory keeps the organizer's filenames and column layout, so the
untouched starter kit works against it verbatim — it simply reports `test = 0
rows`. The agent uses `data.load()` exactly as documented and still cannot reach
a single test label, because they are not on disk anywhere it can see.

[`agent/verify_firewall.py`](agent/verify_firewall.py) is a **second,
independent implementation** of the check. It does not trust the manifest; it
re-reads every written row and asserts the date bound directly, then confirms
the organizer's own loader returns an empty test split. It runs before any
candidate executes, so a breach halts the run.

Three more controls sit behind it: the scorer refuses `split="test"` unless
given an explicit seal token; `evaluate.py` is checksummed before every scoring
call, so a candidate that "improves" by rewriting the metric fails loudly; and
the sealed scorer refuses to overwrite an existing result without `--force`,
making any re-run visible in the audit trail.

Full rationale, including one residual risk that was accepted rather than
hidden, is in [`docs/firewall.md`](docs/firewall.md).

---

## Repository layout

```
agent/
  paths.py            canonical paths; the sealed directory named in one place
  firewall.py         builds work/data_visible/ — Step 1
  verify_firewall.py  independent re-verification of the above
  executor.py         runs untrusted candidate code under a hard timeout
  scorer.py           submission-contract validation + official metric
  journal.py          append-only JSONL run log, fsync per event
candidates/
  fm_baseline.py      candidate 0: the organizer's FM, in the contract
seal/
  final_score.py      SEALED — the only script permitted to read test
kuairand-starter-kit/ the organizer's code, unmodified
harness_check.py      the Phase 1 milestone: 9 checks, no LLM required
docs/
  phase0_baseline_repro.md
  firewall.md
  interventions.md
```

`work/` holds regenerable artifacts (the visible data copy, run journals) and is
gitignored. The dataset itself is never committed.

---

## Quickstart

Requires Python 3.9+ and numpy. Nothing else — no torch, no pandas, no sklearn.

```bash
# Place KuaiRand-Pure under kuairand-starter-kit/KuaiRand-Pure/data/
python3 -m agent.firewall           # build the agent's view of the data
python3 -m agent.verify_firewall    # prove it contains no test rows
python3 harness_check.py            # 9 checks end to end (~45 s)
```

`harness_check.py` is the Phase 1 milestone and needs no API key. It pushes the
organizer's own FM through the complete pipeline — executor, contract, scorer,
journal — against the firewalled directory and asserts it comes back out at the
published baseline:

```
[1. evaluate.py unmodified]                    PASS
[2. firewall built and clean]                  PASS   train=1,141,112 valid=124,909 test=0
[3. candidate 0 runs under the executor]       PASS   ok in 29.1s
[4. output satisfies the submission contract]  PASS   124,909 rows aligned
[5. reproduces the published baseline]         PASS   primary 0.6015, delta -0.0000
[6. executor survives a crashing candidate]    PASS   ValueError caught, loop continued
[7. executor kills a hanging candidate]        PASS   TIMEOUT after 5.1s
[8. scorer rejects a misaligned submission]    PASS
[9. scorer refuses the test split]             PASS
```

Checks 6 and 7 inject a crash and a hang deliberately. The hang test spawns a
*grandchild* process specifically because a plain `kill()` would leave it
running and hold the output pipe open forever — the most common way an
unattended loop dies silently. The executor kills the whole process group.

---

## The candidate contract

Every candidate — the reference one and every script the agent writes — is a
standalone program:

```bash
python3 <candidate>.py --data_dir DIR --split {train,valid,test} --out FILE
```

It reads only from `--data_dir`, trains on train, selects on valid, and writes
`row_id,user_id,video_id,score` for `--split` in exactly the row order of
`data.load(data_dir)[split]`.

Alignment is checked row by row before scoring. The submission format is
positional because `(user_id, video_id)` is **not** a key — the test split
contains 3.06% duplicate pairs. A candidate that sorted its output would
otherwise score like noise and be indistinguishable from a bad idea.

`--split` is a parameter from the very first candidate so that the winning
script needs **no edits** between validation-selection and the sealed test run.
An edit at that boundary is exactly where a leak would hide.

---

## Where the headroom is

The organizers measured two directions as dead ends: adding all 13 feature
fields (0.5940 vs 0.5950) and increasing embedding dimension (flat across
k = 8/16/32). Capacity and static features are not the bottleneck.

The loss function is. The baseline optimises pointwise logloss but is scored on
ranking. The Phase 0 training trace shows them come apart directly:

| epoch | train logloss | valid primary |
|---|---|---|
| 5 | 0.4941 | 0.6010 |
| **7** | 0.4859 | **0.6015** ← best |
| 9 | 0.4784 | 0.6007 |
| 11 | 0.4705 | 0.5990 |

After epoch 7 the model keeps improving at what it optimises and degrades at
what it is judged on. It is studying for the wrong exam.

**This is deliberately not encoded into the agent.** Ranking the open
directions — pairwise/listwise losses, behaviour-sequence modelling,
multi-task heads, censored watch-time regression, unbiased validation against
the randomised-exposure log — is the agent's job, and its reasoning is the
Innovation criterion. Handing it the answer would forfeit that.

---

## Design priorities

Grading weights technical execution at 35%, but autonomy at 20% — measured as
the **number of manual interventions**. An agent that stumbles, recovers by
itself, and gains +0.015 beats a hand-tuned model at +0.04 that was rescued six
times. That ordering drives every choice here: the executor never lets a
candidate take the loop down, the journal is fsynced per event so a hard kill
loses at most one record, and the run log is emitted live rather than
reconstructed afterwards.

---

## Attribution

`kuairand-starter-kit/` is the organizers' code, committed pristine and
unmodified; `evaluate.py` is checksummed to prove it stayed that way. The
dataset is [KuaiRand](https://kuairand.com) (Gao et al.), used under its
published licence and never redistributed here. No external training data is
used — KuaiRand only.
