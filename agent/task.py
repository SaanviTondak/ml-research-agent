"""What a task is, from the agent's point of view.

The loop does not know what a recommender system is. It knows how to ask a
model for a script, run it, find out whether the result was better than the
last one, and decide what to try next. Everything that makes a *particular*
benchmark particular - the metric, the data layout, the submission contract,
the briefing the model is given - lives behind this interface.

The interface is not invented. It is exactly the set of calls `agent/loop.py`
already made into `agent.prompts` and `agent.scorer`, with the benchmark
knowledge lifted out from underneath them:

    prompts.SYSTEM               -> Task.system_prompt()
    prompts.explore_prompt()     -> Task.explore_prompt()
    prompts.draft_prompt(...)    -> Task.draft_prompt(...)
    prompts.improve_prompt(...)  -> Task.improve_prompt(...)
    prompts.debug_prompt(...)    -> Task.debug_prompt(...)
    scorer.score_file(...)       -> Task.validate_and_score(...)
    scorer.assert_evaluate_untouched() -> Task.integrity_check()

Two things deliberately do NOT vary per task, because they are the seam that
keeps the loop simple:

  * The candidate contract. A solution is a standalone script invoked as
    `python3 script.py --data_dir DIR --split S --out FILE --seed N` which
    writes a CSV of scores aligned row-for-row with the evaluation set. Every
    task gets the same contract; only the columns differ.
  * The scalar objective. A node has one comparable number. Multi-objective
    search is a different design and is not this one.

Score
-----
`Score` carries an arbitrary metric dict rather than named fields, because
`gauc`/`ndcg5` were the one genuinely benchmark-shaped thing in the old
scorer. `to_dict()` is what gets written into the journal and into
`submission/final_result.json`, so for a task that has already been scored its
keys are a frozen wire format: KuaiRand must keep emitting exactly
GAUC / nDCG@5 / primary / users / rows / split, which it does.
"""
import importlib.util
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_metric_module(path, alias=None):
    """Import a task's evaluate.py by *path*, not by bare module name.

    Three tasks each ship a file called evaluate.py. Loading them with
    `sys.path.insert(...)` plus `from evaluate import ...` means the first one
    imported wins sys.modules and every later task silently scores with
    somebody else's metric. That is invisible in a single-task process and
    wrong the moment two tasks meet - which is exactly what the test suite is.

    Candidates still do a bare `import evaluate`; they run in their own
    subprocess with the task's directory on PYTHONPATH, so nothing collides
    there and the contract shown to the model is unchanged.
    """
    path = Path(path)
    name = alias or f"_metric_{path.parent.name}_{path.stem}"
    cached = sys.modules.get(name)
    if cached is not None and getattr(cached, "__file__", None) == str(path):
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load metric module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Score:
    """One evaluation of one candidate on one split.

    `metrics` holds every number the task measured; `primary` is the single
    scalar the search compares. Keeping the others is not decoration - the run
    log shows them, and a metric that moves while the primary does not is
    exactly the kind of thing a postmortem needs.
    """
    primary: float
    metrics: dict = field(default_factory=dict)
    split: str = "valid"
    rows: int = 0
    groups: int = 0
    group_label: str = "groups"

    @property
    def _has_groups(self):
        """False for an ungrouped task, where the group count IS the row count
        and emitting it twice would collide in to_dict()."""
        return bool(self.group_label) and self.group_label != "rows"

    def to_dict(self):
        d = dict(self.metrics)
        d["primary"] = self.primary
        if self._has_groups:
            d[self.group_label] = self.groups
        d["rows"] = self.rows
        d["split"] = self.split
        return d

    def __str__(self):
        parts = " | ".join(f"{k} {v:.4f}" for k, v in self.metrics.items())
        where = f"{self.rows:,d} rows"
        if self._has_groups:
            where += f", {self.groups:,d} {self.group_label}"
        return (f"{parts} | primary {self.primary:.4f}  "
                f"({where}, split={self.split})")


class ContractError(Exception):
    """The candidate's output does not satisfy the submission contract."""


class IntegrityError(Exception):
    """The task's own metric implementation has been modified."""


class Task:
    """A benchmark the agent can be pointed at.

    Subclasses supply the benchmark; the loop supplies the search. Anything a
    subclass has to override is abstract here rather than defaulted, so a
    half-implemented task fails loudly at construction rather than quietly
    mid-run.
    """

    name = "unnamed"
    metrics = ()                  # metric names, in display order
    higher_is_better = True
    group_label = "groups"

    # Convergence epsilon mandated from outside (a competition's stated rule).
    # None means "derive it from the measured noise floor". When it is set,
    # calibration reports how many sigma it corresponds to but does not
    # overrule it - silently replacing a competition's stopping rule with one
    # of our own would invalidate the result it produced.
    declared_eps = None

    # Smallest difference the metric can express, if it is discrete. Accuracy
    # over n rows moves in steps of 1/n, and a "gain" smaller than one row is
    # not a gain. None means continuous.
    metric_granularity = None

    @property
    def sign(self):
        """+1 when larger is better, -1 when smaller is. The policy layer
        compares `sign * score` so that RMSE and AUC behave identically."""
        return 1.0 if self.higher_is_better else -1.0

    # ------------------------------------------------------------------ data
    def prepare(self, force=False):
        """Materialise the agent's visible data directory. Must not expose
        held-out labels. Returns a manifest dict for the journal."""
        raise NotImplementedError

    def visible_data_dir(self):
        raise NotImplementedError

    def verify_isolation(self, verbose=False):
        """Assert the held-out split is unreachable. Returns per-split counts."""
        raise NotImplementedError

    def extra_sys_path(self):
        """Directories to put on a candidate's PYTHONPATH."""
        return []

    # --------------------------------------------------------------- prompts
    def system_prompt(self):
        raise NotImplementedError

    def briefing(self):
        """Everything factual the agent is given about the task."""
        raise NotImplementedError

    # The four prompt builders below are the same for every task: take the
    # task's briefing and pour it into the shared template. They were abstract
    # at first, and both tasks then implemented them byte-identically - which
    # is the signal that they belong here, not in each subclass. A task only
    # has to supply briefing() and system_prompt(); override one of these only
    # if the *structure* of the ask differs, not the content.
    def explore_prompt(self):
        from agent import prompts
        return prompts.explore_prompt(self.briefing())

    def draft_prompt(self, journal_summary, eda="", n_existing=0, lineages=""):
        from agent import prompts
        return prompts.draft_prompt(self.briefing(), journal_summary, eda=eda,
                                    n_existing=n_existing, lineages=lineages)

    def improve_prompt(self, node, journal_summary, eda=""):
        from agent import prompts
        return prompts.improve_prompt(self.briefing(), node, journal_summary,
                                      eda=eda)

    def debug_prompt(self, node, journal_summary, attempt=1, max_attempts=3):
        from agent import prompts
        return prompts.debug_prompt(self.briefing(), node, journal_summary,
                                    attempt=attempt, max_attempts=max_attempts)

    def extract_hypothesis(self, text):
        from agent import prompts
        return prompts.extract_hypothesis(text)

    # --------------------------------------------------------------- running
    def candidate_argv(self, data_dir, split, out, seed):
        """Arguments after the script path. The contract is fixed; this exists
        so a task can add its own flags, not so it can change the contract."""
        return ["--data_dir", str(data_dir), "--split", split,
                "--out", str(out), "--seed", str(seed)]

    def reference_implementation(self):
        """A working solution used to measure the task's noise floor.

        None disables calibration, which falls back to declared constants.
        """
        return None

    # ------------------------------------------------- evaluation-set noise
    def eval_contributions(self, out_path):
        """Per-evaluation-unit quantities that the metric aggregates.

        Supplying this (plus eval_aggregate) lets calibration measure
        evaluation-set noise by resampling stored predictions, with no
        retraining. Returning None disables that measurement; the policy then
        keys off training variance alone, which underestimates how much a
        validation gain can move.
        """
        return None

    def eval_n_units(self, contributions):
        """How many evaluation units there are. Generic for a dict of arrays."""
        if isinstance(contributions, dict) and contributions:
            return len(next(iter(contributions.values())))
        return len(contributions)

    def eval_aggregate(self, contributions, idx):
        """Recompute the primary metric over a resampled index array.

        Only reached by a task that overrides eval_contributions; a metric is
        rarely a plain mean of per-row numbers (RMSE is the root of one, a
        grouped AUC is a ratio of weighted sums), so it cannot be defaulted.
        """
        raise NotImplementedError(
            f"{type(self).__name__} returns eval_contributions but does not "
            f"implement eval_aggregate, so evaluation noise cannot be "
            f"measured. Implement it, or return None from eval_contributions.")

    # --------------------------------------------------------------- scoring
    def integrity_check(self):
        """Checksum the metric implementation. Returns the digest."""
        raise NotImplementedError

    def validate_and_score(self, out_path, split="valid", data_dir=None,
                           allow_holdout=False):
        """Check the submission contract, then score it. Raises ContractError
        or IntegrityError; never returns a Score for invalid output."""
        raise NotImplementedError

    # ----------------------------------------------------------------- guard
    def guard_patterns(self):
        """(forbidden, suspicious) - lists of (compiled_regex, reason).

        Forbidden constructs are rejected before execution; suspicious ones are
        reported in the run log only. The distinction cost a live iteration to
        learn: see agent/guard.py.
        """
        return [], []

    # -------------------------------------------------------------- display
    def metric_table(self, score_dict):
        """One markdown table of this task's metrics, for the run log."""
        names = [m for m in self.metrics if m in score_dict]
        if not names:
            return None
        head = "| " + " | ".join(names + ["primary"]) + " |"
        rule = "|" + "---|" * (len(names) + 1)
        cells = [f"{score_dict[n]:.4f}" for n in names]
        cells.append(f"**{score_dict.get('primary', float('nan')):.4f}**")
        return f"\n{head}\n{rule}\n| " + " | ".join(cells) + " |"


def spread(values):
    """Robust scale estimate: the larger of the std and the IQR-implied sigma.

    A heavily skewed metric distribution makes the plain standard deviation
    understate the tail, which would hand the policy layer an implausibly
    tight threshold. 1.349 is the IQR of a standard normal, so for Gaussian
    data the two agree and this is a no-op.
    """
    v = sorted(values)
    if len(v) < 2:
        return 0.0
    sd = statistics.stdev(v)
    try:
        q = statistics.quantiles(v, n=4)
        iqr_sigma = (q[2] - q[0]) / 1.349
    except (statistics.StatisticsError, IndexError):
        iqr_sigma = 0.0
    return max(sd, iqr_sigma)
