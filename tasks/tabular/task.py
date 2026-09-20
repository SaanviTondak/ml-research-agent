"""Bring your own CSV: a task that needs no Python from the user.

The Task interface is small, but "small" still meant 150-300 lines before
anyone could point the agent at their own problem. Most real problems are not
within-user ranking over logged impressions; they are "here is train.csv and
valid.csv, here is the target column, score me on AUC". This covers that case
with a command line and no code:

    python3 -m agent.loop --task tabular \\
        --data ./mydata --target churn --metric auc

What the user supplies is a directory containing `train.csv` and `valid.csv`
(and optionally `test.csv`), each with a header row and a target column.

What this supplies for them:

  * **the firewall.** prepare() copies the data into the run's visible
    directory with the *held-out target column removed from the file on disk*.
    Not masked, absent - the same structural guarantee the KuaiRand task gives,
    which is the single thing a newcomer is most likely to get wrong and least
    likely to notice.
  * a briefing generated from their own column names and metric,
  * a reference implementation to beat and to calibrate the noise floor with,
  * the submission contract and its validation.

Direction is taken from the metric, not from the user: RMSE minimises, AUC
maximises, and agent/state.py compares an oriented score so neither needs
special-casing.
"""
import csv
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from agent.paths import WORK
from agent.task import (ContractError, IntegrityError, Score, Task,
                        load_metric_module)

HERE = Path(__file__).resolve().parent
HEADER = ["row_id", "score"]
SPLITS = ("train", "valid", "test")


class TabularTask(Task):
    """A CSV regression or classification problem, configured not coded."""

    name = "tabular"
    group_label = "rows"
    declared_eps = None               # no organizer: derive it from measurement

    def __init__(self, source=None, target=None, metric="rmse",
                 data_dir=None, holdout="test", name=None):
        from tasks.tabular import evaluate as ev
        if metric not in ev.DIRECTION:
            raise ValueError(
                f"unknown metric {metric!r}; have {sorted(ev.DIRECTION)}")
        self.source = Path(source) if source else None
        self.target = target
        self.metric = metric
        self.holdout = holdout
        self.higher_is_better, self.classification = ev.DIRECTION[metric]
        self.metrics = (metric,)
        self.data_dir = Path(data_dir or WORK / "data_tabular")
        if name:
            self.name = name
        self._columns = None

    # ------------------------------------------------------------------ data
    def prepare(self, force=False, verbose=True):
        """Copy the user's CSVs in, with the held-out target removed on disk.

        This is the firewall. The held-out split keeps every feature column and
        loses its target column entirely, so a candidate cannot read the labels
        it is going to be scored against - there is nothing on disk to read.
        """
        if self.source is None:
            raise IntegrityError("no --data directory given")
        if self.data_dir.exists():
            if not force and (self.data_dir / "manifest.json").exists():
                return json.loads((self.data_dir / "manifest.json").read_text())
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True)

        manifest = {"source": str(self.source), "target": self.target,
                    "metric": self.metric, "holdout": self.holdout,
                    "splits": {}}
        for split in SPLITS:
            src = self.source / f"{split}.csv"
            if not src.exists():
                if split == "train" or split == "valid":
                    raise IntegrityError(f"{src} is required and missing")
                continue
            dst = self.data_dir / f"{split}.csv"
            rows, cols = self._copy(src, dst, drop_target=(split == self.holdout))
            manifest["splits"][split] = {"rows": rows, "columns": cols,
                                         "has_target": self.target in cols}
            if verbose:
                print(f"  {split:<6} {rows:>8,d} rows, {len(cols)} columns"
                      + ("  (target removed)" if self.target not in cols else ""))

        if self.target not in manifest["splits"]["train"]["columns"]:
            raise IntegrityError(
                f"target column {self.target!r} not found in train.csv; "
                f"columns are {manifest['splits']['train']['columns']}")
        # Keep the held-out labels somewhere the agent's data directory is not.
        self._stash_holdout()
        (self.data_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n")
        return manifest

    def _copy(self, src, dst, drop_target):
        with open(src, newline="") as fin, open(dst, "w", newline="") as fout:
            rd, wr = csv.reader(fin), csv.writer(fout)
            head = next(rd)
            keep = [i for i, c in enumerate(head) if not (drop_target
                                                          and c == self.target)]
            wr.writerow([head[i] for i in keep])
            n = 0
            for r in rd:
                wr.writerow([r[i] for i in keep])
                n += 1
        return n, [head[i] for i in keep]

    def _stash_holdout(self):
        """Held-out labels live outside the visible directory, for the sealed
        scorer only - the same separation seal/final_score.py relies on."""
        src = self.source / f"{self.holdout}.csv"
        if not src.exists():
            return
        with open(src, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd)
            if self.target not in head:
                return
            ti = head.index(self.target)
            y = [r[ti] for r in rd]
        out = self.data_dir.parent / f"_holdout_{self.data_dir.name}.json"
        out.write_text(json.dumps({"target": self.target, "values": y}))

    def visible_data_dir(self):
        return self.data_dir

    def verify_isolation(self, verbose=False):
        counts = {}
        for split in SPLITS:
            path = self.data_dir / f"{split}.csv"
            if not path.exists():
                counts[split] = 0
                continue
            with open(path, newline="") as fh:
                head = next(csv.reader(fh))
            if split == self.holdout and self.target in head:
                raise IntegrityError(
                    f"{path} still carries the target column {self.target!r}")
            counts[split] = sum(1 for _ in open(path)) - 1
        if self.holdout in counts and counts.get(self.holdout):
            counts[self.holdout] = 0      # present but unscoreable
        if verbose:
            print(f"  visible: {counts}")
        return counts

    def extra_sys_path(self):
        return [HERE]

    # --------------------------------------------------------------- prompts
    def columns(self):
        if self._columns is None:
            with open(self.data_dir / "train.csv", newline="") as fh:
                self._columns = next(csv.reader(fh))
        return self._columns

    def system_prompt(self):
        direction = "HIGHER IS BETTER" if self.higher_is_better else "LOWER IS BETTER"
        return f"""\
You are an autonomous machine-learning researcher. You work alone, without a \
human to consult, on a tabular prediction benchmark.

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
- The metric is {self.metric} and {direction}.

Be concrete and empirical. Prefer a clean test of one idea over a bundle of \
changes you cannot attribute."""

    def briefing(self):
        cols = [c for c in self.columns() if c != self.target]
        direction = ("higher is better" if self.higher_is_better
                     else "lower is better")
        kind = "classification" if self.classification else "regression"
        preview = self._preview()
        return f"""\
# The task

Tabular {kind} over independent rows. Predict `{self.target}` from the other
columns. There is no grouping and no ranking structure.

Metric: {self.metric}, called the "primary" score. **{direction.upper()}.**

# Splits

train  - fit on this, carries `{self.target}`
valid  - select on this, carries `{self.target}`
{self.holdout:<6} - EXISTS BUT HAS NO `{self.target}` COLUMN. Not scoreable by you.

# Columns in --data_dir/train.csv

target:   {self.target}
features: {', '.join(cols)}

{preview}

Not every column is necessarily informative, and some may need encoding before
a model can use them.

{self.contract()}
"""

    def _preview(self):
        """A few real rows beat any description of the schema."""
        try:
            with open(self.data_dir / "train.csv", newline="") as fh:
                rd = csv.reader(fh)
                head = next(rd)
                rows = [next(rd) for _ in range(3)]
        except (OSError, StopIteration):
            return ""
        w = [max(len(h), max((len(r[i]) for r in rows), default=0))
             for i, h in enumerate(head)]
        fmt = lambda r: "  ".join(str(v)[:w[i]].ljust(w[i])
                                  for i, v in enumerate(r))
        return ("# First rows of train.csv\n```\n"
                + fmt(head) + "\n" + "\n".join(fmt(r) for r in rows) + "\n```")

    def contract(self):
        ev = (HERE / "evaluate.py").read_text()
        ref = (HERE / "reference.py").read_text()
        return f"""\
# The contract your script must satisfy

Invoked as:
    python3 script.py --data_dir DIR --split valid --out FILE --seed N

`--seed` is REQUIRED and must control every source of randomness you use.
Promising results are re-run on several seeds before being accepted.

It must write a CSV with header `row_id,score`, one line per row of the chosen
split's CSV, IN THAT EXACT ORDER. row_id starts at 0 and increments by 1.
`score` is your prediction for `{self.target}`. NaN and Inf are rejected.
{"For a classification metric, emit a probability, not a hard 0/1 label."
 if self.classification else ""}

## evaluate.py - the definition of the score. Do not modify or reimplement it.
```python
{ev}
```

## A working reference implementation satisfying the contract.
It is a plain linear baseline and you are expected to beat it.
```python
{ref}
```
"""

    # --------------------------------------------------------------- running
    def reference_implementation(self):
        return HERE / "reference.py"

    def reference_argv(self, data_dir, split, out, seed):
        argv = self.candidate_argv(data_dir, split, out, seed)
        argv += ["--target", self.target]
        if self.classification:
            argv += ["--classification"]
        return argv

    # --------------------------------------------------------------- scoring
    def integrity_check(self):
        return hashlib.sha256((HERE / "evaluate.py").read_bytes()).hexdigest()

    def _targets(self, split):
        path = self.data_dir / f"{split}.csv"
        if not path.exists():
            raise IntegrityError(f"{path} does not exist")
        with open(path, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd)
            if self.target not in head:
                raise IntegrityError(
                    f"split '{split}' has no target column and cannot be "
                    f"scored here. That is the firewall working, not a bug.")
            ti = head.index(self.target)
            return [float(r[ti]) for r in rd]

    def validate_and_score(self, out_path, split="valid", data_dir=None,
                           allow_holdout=False):
        if split == self.holdout and not allow_holdout:
            raise IntegrityError("refusing to score the held-out split")
        y = self._targets(split)
        preds = self._read_scores(out_path, len(y))
        ev = load_metric_module(HERE / "evaluate.py")
        r = ev.evaluate(y, preds, metric=self.metric)
        metrics = {k: v for k, v in r.items()
                   if k not in ("primary", "rows", "groups")}
        return Score(primary=r["primary"], metrics=metrics, split=split,
                     rows=r["rows"], groups=r["groups"], group_label="rows")

    def _read_scores(self, path, n_expected):
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
                f"submission has {len(out):,d} rows, eval set has "
                f"{n_expected:,d}")
        return out

    # ------------------------------------------------- evaluation-set noise
    def eval_contributions(self, out_path, split="valid"):
        """Per-row quantities the metric aggregates, for the noise bootstrap.

        Every metric here is recomputed from scratch on the resampled rows
        rather than re-averaged, so a rank statistic like AUC bootstraps just
        as correctly as a mean - it is only slower, and at a few thousand rows
        that is not worth optimising.
        """
        y = np.asarray(self._targets(split), dtype=float)
        p = np.asarray(self._read_scores(out_path, len(y)), dtype=float)
        return {"y": y, "p": p}

    def eval_aggregate(self, contributions, idx):
        ev = load_metric_module(HERE / "evaluate.py")
        y, p = contributions["y"][idx], contributions["p"][idx]
        return float(ev.FUNCS[self.metric](list(y), list(p)))

    # ----------------------------------------------------------------- guard
    def guard_patterns(self):
        holdout = re.escape(str(self.data_dir.parent / f"_holdout_"))
        forbidden = [
            (re.compile(holdout), "reaches for the held-out labels"),
            (re.compile(r"_holdout_"), "references the held-out label file"),
        ]
        if self.source:
            forbidden.append(
                (re.compile(re.escape(str(self.source))),
                 "hard-codes the original data directory instead of --data_dir"))
        suspicious = [
            (re.compile(rf"""split\s*=\s*['"]{self.holdout}['"]"""),
             f"selects split='{self.holdout}'"),
        ]
        return forbidden, suspicious
