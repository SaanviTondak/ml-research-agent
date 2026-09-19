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

All four are implemented. None were applied to the submitted run — that result
stands as scored, and the tag `hackathon-submission` marks the tree it came
from. They are the starting point for `final_02`.

**1. Cap the repair budget.** [`agent/state.py`](../agent/state.py) —
`MAX_DEBUG_ATTEMPTS = 3`. `select_parent()` now skips a broken child that has
already had three debug attempts, so the search falls back to improving the
best scored node. This is the whole bug; it is four lines of policy.

**2. `compile()` before spawning a subprocess.**
[`agent/guard.py`](../agent/guard.py) — `assert_parses()`. Eleven of 22
failures were `SyntaxError`, each discovered only after a process launch and a
1.1M-row load. `compile()` finds them in microseconds and reports the line,
column and message, which is a far better repair prompt than "exited with code
1". It executes nothing.

**3. Handle `MAX_TOKENS` by continuing, not shrinking.**
[`agent/loop.py`](../agent/loop.py) — the injected message now asks the model to
reproduce what it wrote and carry on from the cut, explicitly forbidding
redesign or dropping features. The old message asked for a *shorter* script,
and the model complied by deleting features: the three repairs that did run
came back at 0.4791 / 0.5961 / 0.4832 against an incumbent of 0.6042. The
repair pressure was destroying the solution.

**4. Put the error in front of the stale narrative.**
[`agent/prompts.py`](../agent/prompts.py) — `debug_prompt()` used to open with
the failed node's own hypothesis, which is exactly where the wrong
self-diagnosis lived. The traceback now comes first, the previous hypothesis is
demoted to "intent only" and explicitly marked unreliable, and the prompt states
how much repair budget is left so a final attempt reaches for the smallest fix.

### Verification

[`tests/test_search_policy.py`](../tests/test_search_policy.py) rebuilds this
run's tree up to node 10 — the moment the trap closed — replays 23 failed
repairs against it, and asserts node 10 is selected exactly
`MAX_DEBUG_ATTEMPTS` times and never again:

```
test_the_trap_cannot_recur                     PASSED
test_search_escapes_after_the_repair_budget    PASSED
test_a_successful_repair_clears_the_debug_state PASSED
```

### Since fixed

Both items previously listed here as not done are now implemented, along with
three further defects that `final_02` exposed. All are covered by
[`tests/test_exploration_policy.py`](../tests/test_exploration_policy.py) and
[`tests/test_budget.py`](../tests/test_budget.py); the eight tests guarding the
original trap pass unmodified.

**5. The cap and the candidate timeout now read the same clock.** `elapsed_h()`
measured `time.time()` while `communicate(timeout=...)` measures
`time.monotonic()`, so `final_02` billed ~2 h of suspended host against the 6 h
cap and reported a candidate at 7141 s against a 600 s timeout that correctly
never fired. The cap is now charged in monotonic time, `wall_h` and
`suspended_h` are journalled beside it, and `ExecResult.wall_clock_s` keeps the
wall-clock figure rather than erasing it. Awake time also accumulates across
resumes now, via `budget.json` — `final_01` was resumed three times and each
segment had been getting a fresh six hours.

**6. Verification anchors on the best *verified* score.** The trigger compared
against the running best regardless of how many seeds confirmed it, so
`final_02`'s nodes 6–11 — six gains of +0.0005, −0.0001, +0.0005, −0.0014,
−0.0002, +0.0001 — compounded six steps past solid ground without one of them
tripping a check. The anchor is now `best_verified_score()`, the **max** of
confirmed scores rather than the latest, and the threshold is its own constant
(`VERIFY_MARGIN`, one seed σ) instead of borrowing the organizers' convergence
`EPS`: "is the search finished?" and "is this gain worth two more seed runs?"
are different questions. `finalise()` also verifies the shipped node if it was
never confirmed, which is the manual step `docs/final_02_results.md` records
someone doing by hand.

**7. Greed is scoped to a lineage, not the whole journal.** This is the defect
under the first one. `select_parent()` returned the global argmax, so `improve`
only ever extended the single best node and a new lineage survived only if it
won on its first scored attempt. `final_02` node 5 — multi-task supervision plus
censored watch-time — cleared `EPS` on seed 0 by 0.0025, took the incumbency,
and became the whole rest of the run; had it scored a little lower it would have
been orphaned permanently, and the node-0 lineage was in fact abandoned the
moment it lost. A new root now gets `PROTECTED_SCORED_ATTEMPTS = 3` scored
attempts on its own branch before competing globally, bounded by
`PROTECTION_MAX_NODES` and written off immediately if it is more than
`PROTECTION_WRITE_OFF` behind. Convergence is blocked while an exploration is in
budget, and its floor counts exploit attempts only — so buying an exploration
never brings the stopping rule closer. `MIN_SCORED_BEFORE_CONVERGENCE` stays 12.

**8. The draft schedule was preempted, and restarted on every resume.**
`iteration % 7 == 6` sat below the debug branch, so a due draft was silently
swallowed by a repair chain: `final_01`'s iterations 13, 20 and 27 all came due
and all three were eaten, leaving one alternative root opened in thirty
iterations. The counter was also the loop's, which restarts at 1 on resume. The
cadence is now keyed to the journal (`nodes_since_last_root()`), checked before
the debug branch, and gated on having budget left to develop what it opens.

**9. `N_DRAFTS = 3` had never taken effect.** The gate was
`len(self.state.good()) == 0`, which ends the phase the moment the first draft
scores — so both recorded runs opened exactly one initial root while the loop's
module docstring documented three. It now gates on `len(roots())`.

### Deliberately not done

**A broken draft root still gets no repair budget.** `final_01`'s node 9 was a
draft that crashed, and it received zero repairs and zero children in the 24
iterations that followed, because `select_parent()` only searched children of
the best node and a root is never one. The obvious fix is to let protection
cover it — but a lineage with no score has produced no measurement to protect,
and giving it priority over a fresh breakage on the incumbent branch inverts
the two. Protection is therefore earned by evidence: a lineage must have scored
at least once. A crashed draft is replaced by the draft schedule (defect 8),
not repaired. `test_a_broken_draft_root_does_not_preempt_an_in_flight_repair`
pins the decision.

**Protection is root-scoped, so a losing `improve` node is still orphaned.**
This is the honest limit of the fix. `final_02`'s two DIN-style attempts —
node 2, which lost by 0.0004, and node 9, which lost by 0.0014, both inside 2σ
of a *single-seed* incumbent — were `improve` nodes inside an existing lineage,
not new roots, so nothing above would have rescued them. What the fix does is
make the *root* the unit that survives losing narrowly, and route structural
exploration toward roots: drafts now actually happen, they carry a rollup of
which directions each lineage has already spent, and a root that lands inside
the noise gets three attempts to prove itself. Extending a second chance to
near-miss `improve` nodes is the natural next step and is not attempted here.

**Never let a repair regress below the incumbent.** Unchanged from before: a
debug attempt scoring worse than the best node is still recorded normally. It
cannot corrupt selection, since `best()` takes the max, so this stays budget
hygiene rather than a correctness problem.

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
