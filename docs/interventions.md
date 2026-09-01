# Manual intervention log

The competition scores **Impact — Autonomy** (20%) on how often a human had to
step in. This file is the honest record. Every entry costs points; an
unexplained zero costs credibility.

Per the organizers' Q&A, **restarting a crashed process does not count** as an
intervention. Editing the agent's code, its prompts, or its candidates mid-run
does.

**Interventions during the autonomous run: 2.**

| # | time | what happened | why a human was needed | counted |
|---|---|---|---|---|
| 1 | 2026-09-01 08:47 | Run stopped itself after 9 min and 3 attempts: the Gemini free-tier daily quota for `gemini-3.6-flash` was exhausted. The loop saved state and parked, as designed. A human changed `agent/llm.py` so that quota exhaustion falls over to the next model instead of parking, then restarted the run. | The parking behaviour was a design error, not a research failure. Free-tier quota is metered **per model**; six other models still had quota at the moment the run parked. The agent could not fix its own client code. | **Yes** |

| 2 | 2026-09-01 08:56 | The resumed run stopped after 4 scored attempts, reporting convergence — on the same iteration it set a new best (0.5986 → 0.5996). A human added a floor of 12 scored attempts before the convergence rule may fire, then restarted. | The agent cannot change its own stopping rule. The rule as implemented ended the search while it was still improving. | **Yes** |

## Detail on intervention 1

The failure was ours, not the agent's. `agent/llm.py` deliberately treated
daily-quota exhaustion differently from transient rate limiting, on the
reasoning that waiting cannot fix a daily quota — which is true — and therefore
parked the run. The unexamined assumption was that quota is account-wide. It is
not; it is per model. A probe immediately after the stop confirmed seven other
models were serving normally.

The fix makes quota exhaustion fall over like any other model-level failure,
and park only when every model in the chain is spent — which is the only state
waiting could actually fix. Verified by pointing a request at the exhausted
model and watching it answer from `gemini-3.5-flash`.

The three attempts the agent completed before the stop are retained in the
journal; the run resumed into the same run directory rather than starting
clean, so the record is continuous.

This is recorded as an intervention even though the restart alone would not
have counted, because code was changed after seeing the failure. Reporting it
as zero would be false.

## Detail on intervention 2

The organizers' convergence rule is: three consecutive iterations improving the
validation score by no more than eps = 0.002. That was implemented literally
and applied from the first iteration.

Read literally it terminates almost any run with a slow start. The run's
best-so-far history was 0.5986, 0.5986, 0.5986, 0.5996: two attempts that did
not improve, then one that did, by 0.0010. All three gains are <= 0.002, so the
rule fired — stopping the search on the very iteration it found a better
solution, and immediately after the agent had switched from a pairwise to a
listwise objective.

That is the rule working as written and not as intended. eps and N are the
organizers' and are unchanged; what was added is a floor of 12 scored attempts
before the rule may fire at all. The floor is an interpretation of the rule's
scope, and it is disclosed here rather than folded silently into the code.

Counted as an intervention because a human changed the agent's stopping
behaviour after seeing it stop badly.

## Scope

Work done **before** the autonomous run starts — Phases 0–2, building the
harness, seeding context — is construction, not intervention. The count begins
when the loop is first started and ends when it converges or hits its cap.
