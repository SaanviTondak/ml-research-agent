"""Step 3c - what the agent is told.

A deliberate line runs through this file: it is given *facts and materials*,
not *conclusions*.

The competition scores "Innovation & Insight" on what the agent chose to
investigate and why. Handing it the answer converts that into a transcription
exercise, so two things are withheld even though we know them:

  * the organizers' own ranking of the unexplored directions - the direction
    list below is present, but alphabetised rather than ordered by promise;
  * our Phase 0 observation that training loss and validation ranking score
    diverge after epoch 7, which points hard at one particular direction.

Everything factual is included: the task definition, the metric's exact
conventions, the file schemas, the measured dead ends, and paper abstracts.
The agent has to work out where the headroom is by itself. If it re-derives
the loss-function argument from its own runs, that is a real result. If it
finds something else, that is more interesting still.

Withheld material is recorded in docs/agent_briefing.md so the choice is
auditable rather than invisible.
"""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.paths import STARTER, CANDIDATES

SYSTEM = """\
You are an autonomous machine-learning researcher. You work alone, without a \
human to consult, on a recommender-system ranking benchmark.

You operate in a loop: form a hypothesis, write a single self-contained Python \
script that tests it, and read the result. You will see the outcome of every \
previous attempt, including your own failures. Learn from them.

Rules you must follow:
- Output ONE fenced ```python block containing the complete script. No prose \
outside it beyond a short HYPOTHESIS line. Never output a partial file or a diff.
- The script must run standalone with `python3 script.py --data_dir ... \
--split valid --out ...`.
- numpy is available. torch, pandas, sklearn and lightgbm are NOT installed \
and cannot be installed. Write the model yourself in numpy.
- Read data ONLY from the --data_dir argument. Never hard-code a dataset \
path, and do not give --data_dir a default value - make it required.
- Evaluate ONLY on the 'valid' split. A held-out test split exists but is not \
present in your data directory and must never be referenced.
- Never modify or reimplement evaluate.py. It defines the score.
- Keep the script under ~250 lines. Long scripts get truncated at the \
output limit and are wasted.\n- Keep runtime under 10 minutes. The reference model trains in ~30 seconds, so \
you have room, but an unbounded loop wastes an iteration.

Be concrete and empirical. Prefer a clean test of one idea over a bundle of \
changes you cannot attribute."""


def _read(p):
    return Path(p).read_text()


TASK = """\
# The task

KuaiRand-Pure, a short-video recommendation log. For each user, rank the \
videos that user was actually shown. This is WITHIN-USER ranking over logged \
impressions - there is no retrieval over a catalogue.

Relevance label: `long_view` (binary, 0/1), a native column in the logs.
Metric: mean(GAUC, nDCG@5), called the "primary" score. Higher is better.

The exact scoring conventions are fixed by evaluate.py, reproduced below. \
Read it carefully - several properties of the metric are not obvious.

# Splits

train 20220408-20220421 (1,141,112 rows) - fit on this
valid 20220422-20220428 (  124,909 rows) - select on this
test  20220429-20220508 - EXISTS BUT IS NOT AVAILABLE TO YOU. It is absent \
from your data directory. Do not reference it.

# What you must beat

                     GAUC     nDCG@5   primary
random               0.4999   0.4514   0.4757
item popularity      0.6308   0.5121   0.5715
FM baseline          0.6610   0.5282   0.5946   <- the target
perfect ordering     1.0000   0.7289   0.8645   <- the ceiling

Note the ceiling is 0.8645, not 1.0. A large share of users have all-positive \
or all-negative impressions; no ordering changes their contribution. Judge \
your progress against 0.8645.

Seed-to-seed standard deviation of the baseline is 0.0008. A single-seed gain \
below ~0.002 is not evidence of anything.

# Data files in --data_dir

log_standard_4_08_to_4_21_pure.csv   training-window impressions
log_standard_4_22_to_5_08_pure.csv   validation-window impressions
log_random_4_22_to_5_08_pure.csv     RANDOMISED-exposure impressions, same
                                     window as valid. Videos here were shown
                                     to users at random rather than chosen by
                                     the production recommender.
user_features_pure.csv               per-user static attributes
video_features_basic_pure.csv        per-video attributes
video_features_statistic_pure.csv    per-video aggregate engagement counts

Impression log columns:
  user_id, video_id, date, hourmin, time_ms, is_click, is_like, is_follow,
  is_comment, is_forward, is_hate, long_view, play_time_ms, duration_ms,
  profile_stay_time, comment_stay_time, is_profile_enter, is_rand, tab

video_features_basic columns:
  video_id, author_id, video_type, upload_dt, upload_type, visible_status,
  video_duration, server_width, server_height, music_id, music_type, tag

user_features_pure columns:
  user_id, user_active_degree, is_lowactive_period, is_live_streamer,
  is_video_author, follow_user_num, follow_user_num_range, fans_user_num,
  fans_user_num_range, friend_user_num, friend_user_num_range, register_days,
  register_days_range, onehot_feat0 ... onehot_feat17

video_features_statistic has 52 columns of aggregate counts (show_cnt,
play_cnt, complete_play_cnt, like_cnt, comment_cnt, play_progress, ...).
"""

MEASURED_DEAD_ENDS = """\
# Measured dead ends - do not spend iterations re-testing these

The organizers ran these ablations. The numbers are theirs, on test:

- Adding static features. Expanding from the baseline's 5 fields to 13
  (adding music_id, video_type, upload_type and six coarse user-side buckets)
  scored 0.5940 vs 0.5950 for 5 fields. No difference beyond noise.
- Adding capacity. Embedding dimension k = 8 / 16 / 32 scored
  0.5895 / 0.5902 / 0.5887. Essentially flat.

A structural fact that follows from the metric, and is worth thinking through
before you design anything:

- Ranking is done WITHIN a user. Any term in your score that is constant
  across a given user's impressions cannot change that user's ordering, and
  therefore cannot change the metric. Pure user-side first-order terms
  contribute exactly zero. This was verified empirically: `item_pop x user
  bias` and plain `item_pop` produce identical scores.
"""

DIRECTIONS = """\
# Unexplored directions

Neither the organizers nor anyone else has tested these on this benchmark.
They are listed ALPHABETICALLY, not in any order of expected value - working
out which are promising, and why, is your job.

- Behaviour-sequence modelling. Each user has a median of ~31 interactions in
  the training window. The baseline uses none of this history.
- Censored watch-time modelling. play_time_ms is truncated when a video ends,
  so watch time is a censored observation rather than a clean regression target.
- Deeper architectures. DeepFM, DCN, xDeepFM over the same features.
- Loss function. The baseline optimises pointwise binary cross-entropy.
- Multi-task learning. is_click, is_like, is_follow, is_comment, is_forward and
  play_time_ms are all present and unused.
- Time and distribution shift. hourmin and date are unused, and the training
  window precedes the evaluation window.
- Unbiased evaluation. log_random_*.csv contains randomised-exposure
  impressions, free of the production recommender's selection bias.

You are not restricted to this list. If your own analysis of the data suggests
something else, do that instead and say why.
"""

PAPERS = """\
# Reference material - abstracts only, unordered

BPR: Bayesian Personalized Ranking from Implicit Feedback (Rendle et al., 2009)
  Optimises a pairwise ranking criterion derived from a Bayesian analysis of
  the personalised ranking problem, rather than fitting each observation
  independently. Training maximises the posterior probability that an observed
  item is ranked above an unobserved one for the same user.

ListNet / Softmax cross-entropy for ranking (Cao et al., 2007)
  Treats a whole list as one training instance. Scores over the list are turned
  into a probability distribution via softmax and compared against the
  distribution implied by the labels, giving a listwise loss rather than a
  pointwise or pairwise one.

LambdaRank / LambdaMART (Burges et al., 2006/2010)
  Optimises non-smooth ranking metrics indirectly by weighting each pairwise
  gradient by the change in the target metric that swapping the pair would
  cause, avoiding the need to differentiate the metric itself.

DIN: Deep Interest Network (Zhou et al., 2018)
  Represents a user by attending over their historical behaviour sequence with
  respect to the candidate item, so the user representation varies by candidate
  rather than being a single fixed vector.

SIM: Search-based Interest Model (Pi et al., 2020)
  Extends behaviour-sequence modelling to very long histories by first
  retrieving a relevant subset of past behaviours for the candidate, then
  applying attention only to that subset.

ESMM: Entire Space Multi-Task Model (Ma et al., 2018)
  Models a downstream conversion jointly with the upstream click it is
  conditioned on, over the full impression space, to address sample selection
  bias and data sparsity in the downstream task.

MMoE / PLE (Ma et al., 2018; Tang et al., 2020)
  Multi-task architectures with shared and task-specific expert subnetworks
  combined by per-task gates, intended to let related tasks share
  representation without negative transfer between conflicting objectives.

DCN: Deep & Cross Network (Wang et al., 2017)
  Applies an explicit, bounded-degree feature-crossing layer in parallel with a
  standard deep network, learning high-order interactions without exhaustive
  manual feature engineering.

CWM: Counterfactual Watch Model (Zhan et al.)
  Treats watch time as censored by video duration: a video watched to the end
  yields a lower bound on true interest rather than an exact value. Uses a
  one-sided loss appropriate to censored observations instead of squared error.
"""


def contract():
    return f"""\
# The contract your script must satisfy

Invoked as:
    python3 script.py --data_dir DIR --split valid --out FILE --seed N

`--seed` is REQUIRED and must actually control every source of randomness in
your script (initialisation, shuffling, any sampling). Promising results are
re-run on several seeds before being accepted, because the seed-to-seed spread
on this benchmark is large relative to the improvements worth chasing. A script
that ignores --seed will appear to produce identical results on every seed and
its apparent gain cannot be distinguished from luck.

It must write a CSV with header `row_id,user_id,video_id,score`, one line per
row of `data.load(data_dir)[split]`, IN THAT EXACT ORDER. row_id starts at 0
and increments by 1. Do not sort, shuffle or deduplicate - (user_id, video_id)
is not a unique key. Any real number is a valid score; only relative order
matters. NaN and Inf are rejected.

The organizer's `data.py` and `evaluate.py` are importable (they are on
PYTHONPATH). You may use them, or parse the CSVs yourself.

## evaluate.py - the definition of the score. Do not modify or reimplement it.
```python
{_read(STARTER / 'evaluate.py')}
```

## data.py - the organizer's loader and feature encoder.
```python
{_read(STARTER / 'data.py')}
```

## A working reference implementation satisfying the contract.
This is the FM baseline you must beat. Use it as a structural template.
```python
{_read(CANDIDATES / 'fm_baseline.py')}
```
"""


def _briefing():
    return "\n\n".join([TASK, MEASURED_DEAD_ENDS, DIRECTIONS, PAPERS, contract()])


# --------------------------------------------------------------- the prompts
def explore_prompt():
    return _briefing() + """

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


def draft_prompt(journal_summary, eda="", n_existing=0):
    intro = ("# Your task right now: write your first candidate model\n"
             if n_existing == 0 else
             "# Your task right now: draft a NEW approach\n\n"
             "Previous attempts are below. Draft something genuinely different "
             "from what is already there - a variation on the current best "
             "belongs in an improvement step, not a draft.\n")
    return _briefing() + _eda_block(eda) + f"""

{intro}
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


def improve_prompt(node, journal_summary, eda=""):
    return _briefing() + _eda_block(eda) + f"""

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


def debug_prompt(node, journal_summary):
    return _briefing() + f"""

# Your task right now: fix a broken script

Attempt #{node.id} failed. Diagnose it from the error and fix it.

Its hypothesis was:
{node.hypothesis}

```python
{node.code}
```

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
