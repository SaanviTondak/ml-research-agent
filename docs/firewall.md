# The test-split firewall

## Why this exists

KuaiRand-Pure ships the test-window labels in the public download. The
organizer's `baseline.py:run_fm` reads them and prints the test score. The
"hidden" test set is hidden by convention only.

For a human entrant that is a mild hazard. For an autonomous agent that writes
and runs its own code across 50 iterations it is fatal: a single candidate that
selects on test — deliberately, or by copying a pattern out of `baseline.py` —
silently invalidates every number the project reports, and nothing in the run
would look wrong.

So the constraint is enforced structurally rather than by policy.

## How it works

`agent/firewall.py` materialises `work/data_visible/`, a data directory that

- uses the organizer's exact filenames and column layout, so the untouched
  starter kit (`data.py`, `encode`, `evaluate.py`) works against it verbatim, and
- physically contains no impression dated after **20220428**, the last day of
  the validation window.

Against that directory `data.load()` returns train and valid as normal and an
**empty list** for test. The agent cannot exfiltrate test labels because they
are not on disk anywhere it can reach. It is not masking, and it is not an
honour system — the rows are absent.

| file | treatment | rows kept | rows dropped |
|---|---|---|---|
| `log_standard_4_08_to_4_21_pure.csv` | date filter ≤ 20220428 | 1,141,112 | 0 |
| `log_standard_4_22_to_5_08_pure.csv` | date filter ≤ 20220428 | 124,909 | **170,588** |
| `log_random_4_22_to_5_08_pure.csv` | truncated to the valid window | 288,338 | 897,721 |
| `user_features_pure.csv` | copied verbatim | — | — |
| `video_features_basic_pure.csv` | copied verbatim | — | — |
| `video_features_statistic_pure.csv` | copied verbatim | — | — |

The 1,141,112 / 124,909 / 170,588 figures match the organizer's published split
sizes exactly, which is itself a check that the date filter is correct.

Row order within each file is preserved, so submission `row_id` alignment
against `data.load()[split]` still holds.

### The randomised-exposure log

`log_random_4_22_to_5_08_pure.csv` is the one judgement call. It spans
20220422–20220508, straddling both the validation and test windows, and it
carries `long_view` labels. It is truncated to the validation window
(20220422–20220428).

This keeps the unbiased-validation direction open — the organizers list it as
an untested avenue — while leaking no test-window label. It is also the
temporally consistent choice: validation is the latest evidence the agent is
allowed to condition on, and randomised rows from the test window are future
information by the same argument that seals the test split itself.

### Known residual: static video statistics

`video_features_statistic_pure.csv` holds platform-level aggregate counts per
video (`long_time_play_cnt`, `play_progress`, and 50 others). These are static
side features shipped by the organizers and used by their own reference work,
but they were computed over the full logging period, which includes the test
window. They are therefore a weak, indirect channel from test-period behaviour
into training.

This is accepted rather than fixed: excluding them would depart from the
organizer's own feature set and rule out a legitimate direction. It is recorded
here so the choice is visible rather than hidden. Nothing in the file is a
per-impression label, and no row of it is attributable to a test impression.

## Verification

`agent/verify_firewall.py` is a **second, independent implementation** of the
check. It does not trust `manifest.json`; it re-reads every row that was
written and asserts the date bound directly, then confirms the organizer's own
loader reports an empty test split.

```
$ python3 -m agent.verify_firewall
  log_random_4_22_to_5_08_pure.csv       288,338 rows, max date 20220428  OK
  log_standard_4_08_to_4_21_pure.csv   1,141,112 rows, max date 20220421  OK
  log_standard_4_22_to_5_08_pure.csv     124,909 rows, max date 20220428  OK
  data.load(data_visible) -> train=1,141,112, valid=124,909, test=0
  test split is empty                                                     OK

firewall intact: no test-window row is reachable by the agent.
```

It runs as check 2 of `harness_check.py`, so a breach halts the run before any
candidate executes.

## Defence in depth

The materialised directory is the primary control. Three more sit behind it:

1. **`agent/scorer.py` refuses `split="test"`** unless explicitly passed
   `allow_test=True`, which only the sealed script does.
2. **`evaluate.py` is checksummed** (`ecfde283…`) before every scoring call. A
   candidate that tries to improve its score by rewriting the metric fails
   loudly instead of silently.
3. **`seal/final_score.py` refuses to overwrite an existing result** without
   `--force`. Scoring test repeatedly to pick a winner is the leak the firewall
   exists to prevent, so the audit trail makes re-runs visible.

## The sealed run

`seal/final_score.py` is the only script permitted to read the full dataset.
Nothing under `agent/` imports it. It takes the candidate already selected on
validation, re-runs it against the full data with `--split test`, scores it
once, and writes an audit trail recording the candidate's own sha256, the
`evaluate.py` sha256, the seeds, and the timestamp.

Because the candidate contract takes `--data_dir` and `--split` as parameters
from the very first candidate, the winning script needs no edits to be scored
on test. That is deliberate: an edit between selection and scoring is exactly
where a leak would hide.
