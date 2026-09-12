# Post-mortem: 25 of 34 iterations produced nothing

Run `final_01`, 2026-09-01. The agent converged at validation 0.6042 and beat
the FM baseline by +0.0037 on held-out test. It also spent 68% of its iteration
budget going nowhere. This is the analysis of why.

Everything below is derived from `work/runs/final_01/state.json` and
`submission/journal.jsonl`, both written by the loop while it ran.

## What happened

The loop is a greedy tree search over a solution journal, in the style of AIDE.
Three stages:

| stage | picks a parent by |
|---|---|
| `draft` | none — opens a new root |
| `improve` | the best **scored** node so far |
| `debug` | the node that just **failed** |

Attempts 0–8 behaved exactly as intended. The agent drafted a pointwise FM,
recognised the loss/metric mismatch on its own, and walked BPR → sampled
softmax → listwise softmax, then added video-side categorical features:

| # | stage | parent | valid primary |
|---|---|---|---|
| 0 | draft | — | 0.5986 |
| 3 | improve | 0 | 0.5996 |
| 4 | improve | 3 | **0.6042** ← shipped |
| 7 | improve | 4 | 0.6046 (one seed) |
| 8 | improve | 7 | 0.4665 |

Then node 9 (`draft`) failed, node 10 (`improve` from 7) failed, and the loop
entered `debug`. **It never left.** Nodes 11 through 33 — twenty-three
consecutive iterations — all carry `parent_id = 10`:

```
$ python3 -c "import json; s=json.load(open('work/runs/final_01/state.json'));
  print({n['parent_id'] for n in s['nodes'] if n['id'] >= 11})"
{10}
```

Nodes 4 and 7 sat scored, healthy and unextended the entire time. The search had
a working frontier and could not reach it.

## Root cause

Two defects, one structural and one cosmetic-looking but worse.

### 1. `debug` had no depth limit and no escape hatch

[`agent/loop.py`](../agent/loop.py) selects `debug` whenever the previous node
failed, and `debug` re-parents to that failed node. Nothing caps how many times
this can chain, and nothing routes back to a scored node when repair is clearly
not converging. A single unrepairable node is therefore an absorbing state: the
loop will spend its entire remaining budget on it.

This is the whole bug. It is four lines of policy.

### 2. The agent misdiagnosed its own failures, and I amplified it

When a response hits `MAX_TOKENS`, the loop injects a fixed failure message:

> *"Your response was cut off at the output token limit, so the script is
> incomplete. Complete it, and keep it shorter — under ~250 lines."*

That message is correct 5 times. The actual failure distribution across the
22 failed attempts in nodes 9–33:

| cause | count | nodes |
|---|---|---|
| `SyntaxError` | 11 | 17, 18, 20, 23–26, 28, 30–32 |
| truncated at output limit | 5 | 10, 11, 13, 14, 16 |
| NaN/Inf in output scores | 3 | 19, 22, 29 |
| `IndexError` | 2 | 9, 27 |
| `ValueError` | 1 | 12 |

But the model carried the truncation story forward in its *hypothesis* text for
almost every subsequent attempt. Some variant of "the previous script was cut
off due to exceeding the output token limit" appears in the stated hypothesis of
**22 of the last 23 nodes** — 11 through 33, every one except node 14 — long
after truncation had stopped being the problem:

```
$ python3 -c "import json; s=json.load(open('work/runs/final_01/state.json'));
  print([n['id'] for n in s['nodes'] if 'cut off' in (n.get('hypothesis') or '').lower()])"
[11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
```

The stale explanation propagated through the debug chain's accumulated context
and crowded out the real stack trace, which was sitting right there in the same
prompt.

The 11 `SyntaxError`s are the tell. A syntax error in a *complete* generated
script is a different failure from truncation, but it looks similar from the
inside, and the agent never distinguished them because the loop never made it.

### 3. Compounding: truncation was handled by asking for brevity

Telling a model to "keep it shorter" when it ran out of output budget is the
wrong lever. `max_tokens` was 32768 ([`agent/llm.py:176`](../agent/llm.py#L176));
the correct responses are to raise it, to continue the generation from where it
stopped, or to have the model emit a diff against its parent rather than a full
rewrite. Instead the loop asked for a smaller model, and the agent complied by
dropping features — which is why the three scored attempts in the trap (nodes
15, 21, 33) came back at 0.4791, 0.5961 and 0.4832, all far below the 0.6042 it
had already achieved. The repair pressure was actively destroying the solution.

## Fixes

In priority order. None of these were applied to the submitted run; they are
the plan for `final_02`.

1. **Cap debug depth.** After 3 consecutive failed repairs on one lineage,
   abandon it: mark the subtree dead and select `improve` from the best scored
   node instead. This alone would have returned ~20 iterations to useful work.
2. **`compile()` the generated source before spawning a subprocess.** Eleven of
   22 failures were `SyntaxError`, each of which cost a full process launch and
   a data load to discover. A syntax check is free and the error message it
   produces is far more precise than "exited with code 1".
3. **Handle `MAX_TOKENS` by continuing, not by shrinking.** Raise the cap, and
   on truncation re-request with the partial script as a prefix rather than
   instructing the model to write less.
4. **Put the real stack trace in front of the stale narrative.** The failure
   message the loop injects should lead with the actual exception and explicitly
   tell the model to disregard its previous diagnosis if it conflicts.
5. **Never let a repair regress below the incumbent.** A debug attempt that
   scores worse than the best node should be discarded, not treated as progress.

## What this run got right

Worth recording alongside the failure, because the same log is evidence for
both:

- **Nothing crashed the loop.** All 22 failures were caught, journaled, and
  survived. The executor's process-group kill and the fsynced journal did their
  jobs; a hard stop would have lost at most one event.
- **The trap was expensive, not dangerous.** The convergence rule fired, the
  best node was still node 4, and the selection logic shipped the seed-verified
  candidate rather than the single-seed peak. The search wasted budget without
  corrupting the result.
- **The firewall held.** Zero candidates were rejected by the leak guard and
  zero test rows were reachable, across 34 scripts written by a model that was
  never told not to look.

## The honest summary

The model did the research well and the harness held. The **search policy** was
the weak component — one missing depth limit turned a recoverable failure into a
23-iteration sink, and a helpful-sounding error message taught the agent to
solve the wrong problem for twenty attempts. Both are cheap to fix, which is
mostly an argument for measuring where the budget actually went before tuning
anything else.
