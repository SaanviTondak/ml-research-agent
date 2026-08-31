# What the agent is told — and what is deliberately withheld

The agent's briefing lives in [`agent/prompts.py`](../agent/prompts.py). This
file records what was left out of it, and why, so the choice is auditable
rather than invisible.

## The principle

The agent is given **facts and materials**, never **conclusions**.

"Innovation & Insight" is 20% of the grade and is scored on what the agent
chose to investigate and why. Anything we hand it, it cannot be credited for
discovering. So where we already know an answer, we withhold the answer and
supply the evidence it would need to reach that answer itself.

## What it gets

- The task definition, the exact split dates, and the row counts.
- `evaluate.py` in full — the metric's conventions are subtle and it must be
  able to reason about them.
- `data.py` in full, and `candidates/fm_baseline.py` as a structural template
  for the submission contract.
- Complete column schemas for all six data files, including the columns the
  baseline ignores.
- The baseline ladder, the oracle ceiling of 0.8645, and the seed std of
  0.0008 — it needs these to judge its own results at the right scale.
- The organizers' **measured** dead ends (static features, model capacity) and
  the structural fact that within-user ranking cancels any per-user constant.
- Abstracts for nine relevant papers.
- Its own EDA output, from a script it writes and runs itself on the first
  iteration.

## What is withheld

**1. The organizers' ranking of the unexplored directions.**

The starter kit lists seven untested directions *in order of the organizers'
judged promise*, with the loss function first and an explicit note that they
consider it most likely to work.

The list is given to the agent, but **alphabetised**. Nothing is added,
removed, or altered — only the ordering, which is opinion rather than
measurement, is dropped.

**2. Our Phase 0 observation.**

Our own baseline reproduction shows training loss and validation ranking score
diverging after epoch 7: loss falls 0.4859 → 0.4705 while validation primary
falls 0.6015 → 0.5990. That is strong, specific evidence pointing at one
particular direction, and it is ours rather than the organizers'.

It is withheld entirely. The agent runs its own training and can observe the
same divergence in its own traces, because every candidate's stdout is captured
and fed back to it.

## Why this is the right trade

It costs iterations. The agent may spend several attempts on directions we
already believe are weaker, and it may never find the loss-function argument at
all.

That is the point. An agent that reproduces a conclusion handed to it in its
prompt demonstrates nothing about autonomous research. An agent that reaches it
from its own measurements demonstrates exactly what this track is scored on —
and if it finds something we did not consider, that is a better result than the
one we withheld.

The trade is only defensible because it is disclosed. It is disclosed here.

## What is enforced rather than requested

Two constraints are not left to the prompt:

- **The firewall.** The test rows are absent from the agent's data directory,
  so the instruction "do not use test" is backed by the data not existing. See
  [`firewall.md`](firewall.md).
- **The guard.** [`agent/guard.py`](../agent/guard.py) statically scans every
  generated script *before execution* and rejects any that hard-codes the
  sealed dataset path or references the test split. A rejection is fed back as
  an ordinary failure the agent can correct; it costs one iteration, not the
  run.

Prompt instructions are requests. These two are guarantees.
