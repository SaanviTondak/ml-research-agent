"""Prompt templates - the shape of what the agent is asked, not the content.

Every task fills these with its own briefing. What stays here is the structure
that turned out to matter, and two pieces of it were paid for in live
iterations:

  * debug_template() leads with the traceback, not with the failed attempt's
    own hypothesis. In run final_01 that hypothesis was where a stale
    self-diagnosis lived - "the previous script was cut off at the output token
    limit" appears in 22 of the last 23 nodes, long after truncation had
    stopped being the cause - and the model kept solving for it while the real
    error sat further down the prompt.
  * draft_template() carries a per-lineage rollup of directions already spent,
    because the journal summary truncates each hypothesis to 88 characters -
    enough to tell attempts apart, not enough to tell directions apart.

See docs/postmortem.md for both.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def explore_prompt(briefing):
    return briefing + """

# Your task right now: exploratory data analysis

Before modelling, write a script that measures whatever you think you need to
know about this data in order to choose a direction well.

This script is NOT scored and does not need to follow the submission contract.
It takes --data_dir and should simply print findings to stdout. Everything it
prints is kept and shown to you on every future iteration, so measure what will
actually inform your decisions - and keep the output compact enough to read.

Think about what the metric's definition implies about which properties of the
data matter. Some of what determines your ceiling here is a property of the
label distribution, not of any model.

Output one fenced ```python block. Prefix it with a single line:
HYPOTHESIS: <what you are trying to find out>
"""


def draft_prompt(briefing, journal_summary, eda="", n_existing=0,
                 lineages=""):
    intro = ("# Your task right now: write your first candidate model\n"
             if n_existing == 0 else
             "# Your task right now: draft a NEW approach\n\n"
             "Previous attempts are below. Draft something genuinely different "
             "from what is already there - a variation on the current best "
             "belongs in an improvement step, not a draft.\n")
    return briefing + _eda_block(eda) + f"""

{intro}
## Directions tried so far, by lineage
```
{lineages or "(none yet)"}
```

## Attempts so far
```
{journal_summary}
```

Choose a direction, justify it from what you know about the data and the
metric, and implement it as a complete script satisfying the contract.

Start simple enough that it runs. A working mediocre model you can improve
beats an ambitious one that crashes.

Output one fenced ```python block. Prefix it with a single line:
HYPOTHESIS: <the idea you are testing and why you expect it to help>
"""


def improve_prompt(briefing, node, journal_summary, eda=""):
    return briefing + _eda_block(eda) + f"""

# Your task right now: improve the best solution so far

## Attempts so far
```
{journal_summary}
```

## The solution to improve (attempt #{node.id}, valid primary {node.score:.4f})
Its hypothesis was:
{node.hypothesis}

```python
{node.code}
```

Its output ended with:
```
{node.stdout_tail[-1500:]}
```

Make ONE substantive change you can attribute. Explain what you are changing
and why you expect it to raise the validation score. Output the COMPLETE
modified script, not a diff.

Remember: seed noise is ~0.0008, so aim for a change that could plausibly move
the score by more than 0.002. Re-tuning a hyperparameter by a little is not
worth an iteration.

Output one fenced ```python block. Prefix it with a single line:
HYPOTHESIS: <the change and why>
"""


def debug_prompt(briefing, node, journal_summary, attempt=1,
                 max_attempts=3):
    """Prompt to repair a broken node.

    Ordering matters here. The first version led with the failed attempt's own
    hypothesis, and in run final_01 that hypothesis was where a stale
    self-diagnosis lived: "the previous script was cut off at the output token
    limit" appears in 22 of the last 23 nodes, long after truncation had
    stopped being the cause. The model read its own wrong explanation first and
    kept solving for it while the real traceback sat further down the prompt.

    The error now comes first, and the previous diagnosis is explicitly marked
    unreliable. See docs/postmortem.md.
    """
    budget = ""
    if attempt >= max_attempts:
        budget = (f"\n**This is repair attempt {attempt} of {max_attempts}.** "
                  "If it fails again this line of work is abandoned and the "
                  "search returns to the best working solution. Prefer the "
                  "smallest change that makes the script run over a better "
                  "version of it that might not.\n")
    elif attempt > 1:
        budget = f"\nThis is repair attempt {attempt} of {max_attempts}.\n"

    return briefing + f"""

# Your task right now: fix a broken script

Attempt #{node.id} failed. Diagnose it **from the error below**, not from what
you believed was wrong last time.
{budget}
## What went wrong
{node.failure_reason}

## stderr
```
{node.stderr_tail[-2500:]}
```

## stdout before it failed
```
{node.stdout_tail[-1200:]}
```

## The script that failed
```python
{node.code}
```

## What this attempt was trying to do
{node.hypothesis}

Treat that last line as context for the *intent* only. If it contains a claim
about why the previous attempt failed, ignore it - it is the previous
attempt's guess, it is frequently wrong, and the error above is the evidence.

Fix the actual cause. Do not abandon the idea and submit something unrelated,
and do not simply retry the same code. If the approach cannot work as written,
simplify it until it runs.

Output the COMPLETE fixed script in one fenced ```python block. Prefix it with:
HYPOTHESIS: <what was wrong and how you fixed it>
"""


def _eda_block(eda):
    if not eda:
        return ""
    return f"""

# Your own exploratory analysis
You wrote and ran this analysis earlier. It is your own measurement of the data.
```
{eda[:6000]}
```
"""


def extract_hypothesis(text):
    """Pull the HYPOTHESIS line out of a response; fall back to the first prose."""
    for line in (text or "").splitlines():
        s = line.strip()
        if s.upper().startswith("HYPOTHESIS:"):
            return s.partition(":")[2].strip()
    # Fall back to prose that appears BEFORE the code block. Anything after
    # the opening fence is source, and a line of source recorded as a
    # hypothesis makes the run log useless.
    preamble = (text or "").split("```", 1)[0]
    for line in preamble.splitlines():
        s = line.strip().lstrip("#*- ")
        if len(s.split()) >= 5 and "=" not in s:
            return s[:200]
    return "(no hypothesis given)"
