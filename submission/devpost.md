## Inspiration

Track 2 asks for an autonomous ML research agent, and the rubric says something
easy to miss: only 35% rewards a better model. Autonomy — measured as the number
of times a human had to step in — is worth 20%, and the agent's own reasoning is
worth another 20%.

So we built the loop, not the model. An agent that recovers from its own crashes
and gains a little beats a hand-tuned model that had to be rescued six times.

The second thing that shaped the project came from reading the starter kit
closely. KuaiRand-Pure ships the **test labels in the public download**, and the
organizer's own `baseline.py` reads them and prints the test score. The hidden
test set is hidden by convention only.

For a human entrant that is a mild hazard. For an agent writing and running its
own code across dozens of iterations it is fatal: one candidate that selects on
test — deliberately, or by copying a pattern out of the reference implementation
— silently invalidates every number the project reports, and nothing about the
run would look wrong.

That became the first thing we built.

## What it does

An agent that runs the ML research loop unattended. It writes its own
exploratory analysis, forms a hypothesis, writes a complete training script,
runs it, scores it, reads its own failures, and tries again — with no human in
the loop. Every hypothesis, code diff, score and error is written to disk as it
happens, not reconstructed afterwards.

The benchmark is within-user ranking on KuaiRand-Pure: for each user, order the
videos they were actually shown, scored as `mean(GAUC, nDCG@5)`.

Crucially, the agent physically cannot see the held-out test data. That is
enforced by the filesystem, not by instructions in a prompt.

## How we built it

**The test-split firewall.** Rather than promising not to look, we made looking
impossible. `agent/firewall.py` materialises a data directory containing no
impression dated after the end of the validation window — 170,588 test rows
physically removed. It keeps the organizer's exact filenames and column layout,
so their untouched `data.py` works against it verbatim and simply reports
`test = 0 rows`. A second, independent verifier re-reads every written row and
confirms the date bound rather than trusting the manifest.

Three controls sit behind it: a static guard that scans generated code before
execution and rejects any hard-coded path to the sealed dataset; a checksum on
`evaluate.py` before every scoring call, so a candidate cannot improve its score
by rewriting the metric; and a sealed scoring script — the only code permitted
to read test — which refuses to overwrite an existing result.

**The harness, proven before any LLM touched it.** An executor that runs
untrusted code under a hard timeout, killing the whole process group so
grandchildren cannot survive and hold the output pipe open. A scorer that
validates submission alignment row by row, because `(user_id, video_id)` is not
a unique key — the test split has 3.06% duplicate pairs, so a candidate that
sorted its output would score like noise and be indistinguishable from a bad
idea. An append-only journal, fsynced per event so a hard kill loses at most one
record.

We validated all of it by pushing the organizer's own FM through the pipeline
and asserting it came back out at the published baseline: **9/9 checks pass,
valid primary 0.6015, delta −0.0000**, with test physically unreachable. Two of
those checks inject a crash and a hang deliberately.

**The agent.** Greedy tree search: draft three independent attempts, then
improve the best, repair its broken children first, and branch out periodically
so the search does not collapse onto one lineage. Promotion requires agreement
across seeds — the benchmark's seed-to-seed std is 0.0008, so a single-seed gain
below 0.002 is luck, not evidence.

**What we deliberately withheld.** We already knew the likely answer: the
baseline optimises pointwise log-loss but is scored on ranking, and our own
Phase 0 trace shows the two diverging after epoch 7. We kept that from the agent
entirely, and gave it the organizers' list of untested directions *alphabetised*
rather than in their order of judged promise — nothing added or removed, only
the opinion dropped.

## Challenges we ran into

**Responses truncated mid-script.** The first smoke run produced no working
candidates at all. The model was hitting an 8,192-token output cap partway
through the file, so there was no closing code fence to parse. Fixed by raising
the budget, teaching the parser to handle a cut-off fence, and — the part that
mattered — making truncation its *own named failure*, so the repair step
completes the script instead of blindly re-deriving it from a meaningless
`SyntaxError`.

**A guard that would have rejected the reference implementation.** Our leak
guard rejected a good candidate for the line `"test": (20220429, 20220508),` —
copied verbatim from the organizer's `data.py`. The reasoning error was treating
a *name* as a *capability*: with no test rows in the directory, `splits['test']`
is an empty list and every mention of it is inert. The only construct that can
reach real data is a path. A false rejection is not free — it burns an iteration
and teaches the agent to avoid something that was never dangerous.

**Quota exhaustion that stopped a healthy run.** Nine minutes into the live run,
the model hit its free-tier daily quota. Our client parked, by design: waiting
cannot fix a daily quota, so retrying is pointless. The unexamined assumption was
that quota is account-wide. It is metered **per model** — a probe taken seconds
after the stop found six other models serving normally. The run had parked with
plenty of capacity left.

**A convergence rule that fired while the agent was still improving.** The
organizers' rule is three consecutive iterations gaining ≤ 0.002. Implemented
literally and applied from iteration one, it stopped our run after four attempts
— on the very iteration that set a new best, immediately after the agent
switched from a pairwise to a listwise objective. We added a floor of scored
attempts before the rule may fire, and left eps and N untouched.

The last two required human fixes, and both are recorded as manual
interventions. **The count is {{INTERVENTIONS}}, not zero.** Neither was the
agent failing — both were our infrastructure failing in ways the agent could not
repair itself. We report it that way because an unexplained zero costs more
credibility than an explained {{INTERVENTIONS}}.

## Accomplishments that we're proud of

**The agent found the answer we hid from it.** Its first draft was a pairwise
BPR loss; by its fourth attempt it had moved to a listwise softmax
cross-entropy. We had deliberately withheld both our Phase 0 evidence and the
organizers' ranking of the untested directions. It reasoned its way there from
the metric definition and its own runs. That reasoning is in the run log with
timestamps, and it is the agent's, not ours.

**A firewall that is a guarantee rather than a promise.** Most entries in a
competition like this will say they did not use the test set. We can show that
the test set was not present on disk in any directory the agent could read, that
an independent verifier re-checked every row, and that the organizer's own
loader reports zero test rows against our data directory.

**A harness verified before it was trusted.** 9/9 checks, including deliberate
crash and hang injection, and a byte-exact reproduction of the published
baseline through the full pipeline.

**An honest intervention log.** Every human touch is recorded with its cause and
reasoning, including the two that were our own bugs.

### Results

| | valid | test |
|---|---|---|
| random | 0.4834 | 0.4757 |
| item popularity | 0.5807 | 0.5715 |
| FM baseline | 0.6016 | 0.5946 |
| **ML Research Agent** | **{{FINAL_VALID}}** | **{{FINAL_TEST}}** |
| oracle ceiling | 0.8484 | 0.8645 |

{{RESULTS_NOTE}}

**Resources:** {{ITERATIONS}} iterations of 50 · {{WALLCLOCK}} wall-clock of a
6 h cap · {{TOKENS}} LLM tokens · **0 GPU-hours** (numpy on one CPU core) ·
{{INTERVENTIONS}} manual interventions.

## What we learned

**The metric's ceiling is not 1.0.** 27.1% of test users viewed nothing for long
and 9.2% viewed everything; no ordering changes their contribution, and GAUC
excludes them entirely. A model that cheats by reading the labels scores 0.8645.
The baseline at 0.5946 has already captured ~31% of the reachable range.
Measuring progress against 1.0 would have made every real gain look like a
rounding error.

**Guarantees beat instructions.** "Do not use the test split" in a prompt is a
request. A directory that does not contain the test split is a guarantee. Every
control we are proud of is of the second kind.

**Failure handling is the product.** Most of our engineering went into what
happens when things break — timeouts, truncated responses, exhausted quota,
malformed submissions — and every one of those paths fired in anger during the
live run.

**Rules need scope, not just parameters.** The convergence rule was implemented
exactly as specified and still did the wrong thing, because we applied it from
iteration one. A correct formula in the wrong scope is still a bug.

## What's next for ML Research Agent

The run was cut short by a submission deadline rather than by convergence. The
agent was still improving when it was stopped, and the direction it reached
under its own steam — listwise objectives — is the one with the most headroom
left.

Entirely unexplored: behaviour-sequence modelling (each user has a median of ~31
interactions the baseline ignores completely), multi-task heads on the unused
`is_click` / `is_like` / `play_time_ms` columns, censored regression on watch
time, and the 1.18M-row randomised-exposure log as an unbiased validation signal
against biased-traffic overfitting.

Beyond this benchmark, the parts worth reusing are not the model at all: the
firewall pattern, the guard, and the failure taxonomy that turns every crash
into context for the next attempt rather than an end to the run.
