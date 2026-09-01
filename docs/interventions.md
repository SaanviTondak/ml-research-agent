# Manual intervention log

The competition scores **Impact — Autonomy** (20%) on how often a human had to
step in. This file is the honest record. Every entry costs points; an
unexplained zero costs credibility.

Per the organizers' Q&A, **restarting a crashed process does not count** as an
intervention. Editing the agent's code, its prompts, or its candidates mid-run
does.

**Interventions during the autonomous run: 1.**

| # | time | what happened | why a human was needed | counted |
|---|---|---|---|---|
| 1 | 2026-09-01 08:47 | Run stopped itself after 9 min and 3 attempts: the Gemini free-tier daily quota for `gemini-3.6-flash` was exhausted. The loop saved state and parked, as designed. A human changed `agent/llm.py` so that quota exhaustion falls over to the next model instead of parking, then restarted the run. | The parking behaviour was a design error, not a research failure. Free-tier quota is metered **per model**; six other models still had quota at the moment the run parked. The agent could not fix its own client code. | **Yes** |

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

## Scope

Work done **before** the autonomous run starts — Phases 0–2, building the
harness, seeding context — is construction, not intervention. The count begins
when the loop is first started and ends when it converges or hits its cap.
