"""The synthetic task: a second benchmark, chosen for what it can prove.

It differs from KuaiRand in every dimension the Task interface is supposed to
absorb, which is the point - a second task that looked like the first would
demonstrate nothing:

  * the metric is RMSE, so **lower is better**. Every comparison in
    agent/state.py is written in the maximize direction, so a minimize task
    exercises the orientation boundary on every single decision rather than in
    some special case.
  * the metric is unbounded above and near 1, not bounded in [0,1] near 0.6.
  * there are no groups - the score is over rows, not within-user.
  * the submission is `row_id,score`, two columns instead of four.
  * the noise floor is a knob, not a property, so calibration can be checked
    against ground truth instead of against itself.
  * it needs no download and fits in under a second, so CI can run the whole
    loop end to end - which it has never been able to do.

The test targets are written to a separate file that the visible directory
does not contain, the same structural firewall KuaiRand uses: absent, not
masked.
"""
import csv
import hashlib
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from agent import prompts
from agent.paths import WORK
from agent.task import ContractError, IntegrityError, Score, Task

HERE = Path(__file__).resolve().parent
HEADER = ["row_id", "score"]


class SyntheticTask(Task):
    name = "synthetic-regression"
    metrics = ("RMSE", "MAE")
    higher_is_better = False          # the whole reason this task exists
    group_label = "rows"

    # No organizer, so nothing is mandated: EPS is derived from measurement.
    declared_eps = None

    def __init__(self, data_dir=None, jitter=0.0, quantum=0.0, n_eval=None):
        self.data_dir = Path(data_dir or WORK / "data_synthetic")
        self.jitter = jitter
        self.quantum = quantum         # simulate a discrete metric
        # Declared, not inferred: when the spread is smaller than one grid
        # step every seed lands on the same value and there is nothing for
        # detect_quantum() to see. A task that knows its granularity says so.
        self.metric_granularity = quantum or None
        self.n_eval = n_eval

    # ------------------------------------------------------------------ data
    def prepare(self, force=False, verbose=True):
        from tasks.synthetic.generate import build
        if self.data_dir.exists() and not force:
            import json
            return json.loads((self.data_dir / "manifest.json").read_text())
        m = build(self.data_dir, seed=0, n_eval=self.n_eval)
        if verbose:
            print(f"  built {self.data_dir}: "
                  + ", ".join(f"{k} {v['rows']}" for k, v in m["splits"].items()))
        return m

    def visible_data_dir(self):
        return self.data_dir

    def verify_isolation(self, verbose=False):
        """The test split exists but carries no target column."""
        counts = {}
        for split in ("train", "valid", "test"):
            path = self.data_dir / f"{split}.csv"
            if not path.exists():
                raise IntegrityError(f"{path} missing - run prepare()")
            with open(path, newline="") as fh:
                head = next(csv.reader(fh))
            counts[split] = 0 if head[-1] != "target" else sum(
                1 for _ in open(path)) - 1
        if counts["test"] != 0:
            raise IntegrityError("the test split exposes its target column")
        if verbose:
            print(f"  visible: {counts}")
        return counts

    def extra_sys_path(self):
        return [HERE]

    # --------------------------------------------------------------- prompts
    def briefing(self):
        return f"""\
# The task

Predict a continuous target from 12 numeric features. This is plain tabular
regression over independent rows - there is no grouping, no ranking, and no
user structure.

Metric: RMSE, called the "primary" score. **LOWER IS BETTER.** A run that
drives the score down is improving.

# Splits

train  - fit on this, carries a `target` column
valid  - select on this, carries a `target` column
test   - EXISTS BUT HAS NO TARGET COLUMN. It is not scoreable by you.

# Data files in --data_dir

train.csv   f0..f11, target
valid.csv   f0..f11, target
test.csv    f0..f11

All columns are floats. The target was generated from a linear combination of
some of the features plus at least one term a linear model on the raw columns
cannot represent, plus irreducible noise. Not all 12 features are informative.

{self.contract()}
"""

    def contract(self):
        ref = (HERE / "reference.py").read_text()
        ev = (HERE / "evaluate.py").read_text()
        return f"""\
# The contract your script must satisfy

Invoked as:
    python3 script.py --data_dir DIR --split valid --out FILE --seed N

`--seed` is REQUIRED and must control every source of randomness you use.
Promising results are re-run on several seeds before being accepted.

It must write a CSV with header `row_id,score`, one line per row of the chosen
split's CSV, IN THAT EXACT ORDER. row_id starts at 0 and increments by 1.
`score` is your predicted target. NaN and Inf are rejected.

numpy is available. Write the model yourself.

## evaluate.py - the definition of the score. Do not modify or reimplement it.
```python
{ev}
```

## A working reference implementation satisfying the contract.
It is deliberately mediocre and you are expected to beat it.
```python
{ref}
```
"""

    def system_prompt(self):
        return """\
You are an autonomous machine-learning researcher. You work alone, without a \
human to consult, on a tabular regression benchmark.

You operate in a loop: form a hypothesis, write a single self-contained Python \
script that tests it, and read the result. You will see the outcome of every \
previous attempt, including your own failures. Learn from them.

Rules you must follow:
- Output ONE fenced ```python block containing the complete script. No prose \
outside it beyond a short HYPOTHESIS line.
- The script must run standalone with `python3 script.py --data_dir ... \
--split valid --out ...`.
- numpy is available. Do not rely on pandas, sklearn or torch.
- Read data ONLY from the --data_dir argument. Never hard-code a dataset path.
- Evaluate ONLY on the 'valid' split.
- Never modify or reimplement evaluate.py. It defines the score.
- The metric is RMSE and LOWER IS BETTER.

Be concrete and empirical. Prefer a clean test of one idea over a bundle of \
changes you cannot attribute."""

    def explore_prompt(self):
        return prompts.explore_prompt(self.briefing())

    def draft_prompt(self, journal_summary, eda="", n_existing=0, lineages=""):
        return prompts.draft_prompt(self.briefing(), journal_summary, eda=eda,
                                    n_existing=n_existing, lineages=lineages)

    def improve_prompt(self, node, journal_summary, eda=""):
        return prompts.improve_prompt(self.briefing(), node, journal_summary,
                                      eda=eda)

    def debug_prompt(self, node, journal_summary, attempt=1, max_attempts=3):
        return prompts.debug_prompt(self.briefing(), node, journal_summary,
                                    attempt=attempt, max_attempts=max_attempts)

    # --------------------------------------------------------------- running
    def reference_implementation(self):
        return HERE / "reference.py"

    def reference_argv(self, data_dir, split, out, seed):
        argv = self.candidate_argv(data_dir, split, out, seed)
        if self.jitter:
            argv += ["--jitter", str(self.jitter)]
        return argv

    # --------------------------------------------------------------- scoring
    def integrity_check(self):
        return hashlib.sha256((HERE / "evaluate.py").read_bytes()).hexdigest()

    def _targets(self, split):
        path = self.data_dir / f"{split}.csv"
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd)
            if head[-1] != "target":
                raise IntegrityError(
                    f"split '{split}' has no target column and cannot be "
                    f"scored here. That is the firewall working, not a bug.")
            return [float(r[-1]) for r in rd]

    def validate_and_score(self, out_path, split="valid", data_dir=None,
                           allow_holdout=False):
        if split == "test" and not allow_holdout:
            raise IntegrityError("refusing to score the held-out split")
        y = self._targets(split)
        preds = self._read_scores(out_path, len(y))
        sys.path.insert(0, str(HERE))
        from evaluate import evaluate
        r = evaluate(y, preds, quantum=self.quantum)
        return Score(primary=r["primary"],
                     metrics={"RMSE": r["RMSE"], "MAE": r["MAE"]},
                     split=split, rows=r["rows"], groups=r["groups"],
                     group_label="rows")

    def _read_scores(self, path, n_expected):
        """Same positional contract as KuaiRand, two columns instead of four."""
        path = Path(path)
        if not path.exists():
            raise ContractError(f"candidate produced no output file at {path}")
        out = []
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd, None)
            if head != HEADER:
                raise ContractError(
                    f"header must be exactly {','.join(HEADER)} - got {head}")
            for lineno, rec in enumerate(rd, start=2):
                if len(rec) != 2:
                    raise ContractError(
                        f"line {lineno}: {len(rec)} fields, expected 2")
                if len(out) >= n_expected:
                    raise ContractError(
                        f"line {lineno}: more rows than the eval set "
                        f"({n_expected:,d} rows)")
                try:
                    if int(rec[0]) != len(out):
                        raise ContractError(
                            f"line {lineno}: row_id={rec[0]}, expected "
                            f"{len(out)} (must start at 0, increment by 1)")
                except ValueError:
                    raise ContractError(
                        f"line {lineno}: row_id={rec[0]!r} is not an integer")
                try:
                    v = float(rec[1])
                except ValueError:
                    raise ContractError(
                        f"line {lineno}: score {rec[1]!r} is not a number")
                if v != v or v in (float("inf"), float("-inf")):
                    raise ContractError(f"line {lineno}: score is NaN or Inf")
                out.append(v)
        if len(out) != n_expected:
            raise ContractError(
                f"submission has {len(out):,d} rows, eval set has {n_expected:,d}")
        return out

    # ------------------------------------------------- evaluation-set noise
    def eval_contributions(self, out_path, split="valid"):
        y = np.asarray(self._targets(split), dtype=float)
        p = np.asarray(self._read_scores(out_path, len(y)), dtype=float)
        return {"se": (y - p) ** 2}

    def eval_n_units(self, contributions):
        return len(contributions["se"])

    def eval_aggregate(self, contributions, idx):
        """RMSE is the root of a mean, not a mean - hence the aggregator."""
        return float(np.sqrt(contributions["se"][idx].mean()))

    # ----------------------------------------------------------------- guard
    def guard_patterns(self):
        forbidden = [
            (re.compile(re.escape(str(self.data_dir / "_test_target"))),
             "hard-codes the held-out target file"),
            (re.compile(r"_test_target"), "references the held-out targets"),
        ]
        suspicious = [
            (re.compile(r"""split\s*=\s*['"]test['"]"""), "selects split='test'"),
        ]
        return forbidden, suspicious


TASK = SyntheticTask()
