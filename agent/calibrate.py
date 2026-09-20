"""Measure the task's noise floor, then derive the policy constants from it.

Why this exists
---------------
Every threshold in agent/state.py was a literal tuned against one benchmark:
VERIFY_MARGIN = 0.0008 *is* that benchmark's seed sigma, EPS = 0.002 is a
multiple of it, PROTECTION_WRITE_OFF inherits it. Point the same agent at a
task whose sigma is 0.05 and the policy is nonsense - it would verify on every
node and never converge. Constants in absolute metric units cannot travel;
constants in units of the task's own noise can.

Two noise sources, two different questions
------------------------------------------
This is the part that was wrong even on the original benchmark, so it is worth
stating precisely. A measured score is

    S = mu(design) + T(design, seed) + E(design, eval set)

  * **sigma_seed** - training variance. Re-run the same design with a
    different seed and this is how much the score moves.
  * **sigma_eval** - evaluation-set sampling. The validation set is one draw
    from a population; a different draw would rank designs slightly
    differently.

The policy asks two different questions and they key off different sources:

  * *"Is this single-seed result worth spending more seeds on?"* compares two
    numbers measured on the **same** evaluation rows, so the eval draw is a
    common offset that largely cancels. Scale: **sigma_seed**.
  * *"Is the search still making real progress, or climbing this particular
    validation sample?"* is asking whether a gain will survive a different
    draw. Scale: **sigma_delta = sqrt(2*sigma_seed^2 + sigma_eval_paired^2)**.

Measured on this project's own recorded runs, sigma_seed = 0.00076 (pooled,
12 dof, six multi-seed nodes) while the paired eval bootstrap gives ~0.0010 -
so sigma_delta is about 0.0015, roughly double the quantity every constant was
keyed to. The eval term was the larger of the two and was invisible to the
policy. Nothing divides by sigma anywhere in here, so a deterministic task
(sigma_seed exactly 0) is a supported case rather than a crash.

What it costs
-------------
k reference fits, bounded by a share of the run budget. On a ~20 s reference
that is about five minutes against a six-hour cap. The bootstrap re-aggregates
stored predictions and costs under a second.
"""
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.executor import run_script
from agent.task import spread

# Multipliers. Chosen so that, fed this project's measured noise, they
# regenerate the constants a human tuned against two real runs - which is the
# only external check available short of many tasks. See tests.
VERIFY_Z = 1.0          # VERIFY_MARGIN = 1 sigma_seed
EPS_Z = 1.5             # EPS = 1.5 sigma_delta
"""Why 1.5 and not 2.5.

The stopping rule already requires N=3 *consecutive* sub-eps steps, so the
confidence comes from the repetition, not from the per-step threshold. The
probability of stopping while real progress of delta is still available is
about Phi((eps-delta)/sigma)^N:

    eps      delta = 2 sigma still available
    1.0 s    0.4%
    1.5 s    3%
    2.5 s    33%

At 2.5 sigma you have a one-in-three chance of ending a search that is still
genuinely improving, and that is the failure you cannot see in the log. Bias
it small. Note also that has_converged() reads a *monotone running best*, so
record increments shrink as a run lengthens: the fails-to-stop mode
self-corrects and the stops-too-early mode does not."""

MIN_SEEDS = 2
MAX_SEEDS = 10
DEFAULT_CALIB_FRACTION = 0.05     # share of the run budget calibration may use
CALIB_CAP_S = 600.0
BOOTSTRAP_B = 400
RELATIVE_EPS_FLOOR = 1e-3   # last-resort eps as a fraction of |score|

# sd of a k-sample estimate is itself noisy: relative SE is 1/sqrt(2(k-1)).
# Inflate to a one-sided upper bound so the thresholds err wide. Understating
# sigma promotes noise, which corrupts the result invisibly; overstating it
# spends some verification budget, which is merely wasteful. Prefer waste.
_CHI2_025_BY_DF = {1: 0.1015, 2: 0.5754, 3: 1.2125, 4: 1.9226, 5: 2.6746,
                   6: 3.4546, 7: 4.2549, 8: 5.0706, 9: 5.8988}


def _ucl_factor(k):
    """Multiply a k-sample sd by this for a 75% upper confidence bound.

    sigma_ucl = sigma_hat * sqrt(df / chi2_{0.25, df}), df = k-1. The price of
    a small k, made explicit:  k=2 -> 3.14x,  k=3 -> 1.86x,  k=5 -> 1.44x,
    k=10 -> 1.24x.
    """
    df = max(1, k - 1)
    chi2 = _CHI2_025_BY_DF.get(df)
    if chi2 is None:                      # df > 9: Wilson-Hilferty approximation
        chi2 = df * (1 - 2 / (9 * df) - 0.6745 * math.sqrt(2 / (9 * df))) ** 3
    return math.sqrt(df / chi2) if chi2 > 0 else 3.14


def detect_quantum(values, rel_tol=1e-9):
    """Smallest step a discrete metric moves in, or 0.0 if it is continuous.

    Accuracy over 500 rows moves in steps of 0.002, and a "gain" finer than
    one row is not a gain. Detected rather than declared, so a task does not
    have to know.
    """
    v = sorted(set(values))
    if len(v) < 3:
        return 0.0
    scale = max(abs(x) for x in v) or 1.0
    diffs = [b - a for a, b in zip(v, v[1:]) if (b - a) > rel_tol * scale]
    if len(diffs) < 2:
        return 0.0
    q = min(diffs)
    if q <= 0:
        return 0.0
    if all(abs(d / q - round(d / q)) < 1e-6 for d in diffs):
        return q
    return 0.0


@dataclass
class Calibration:
    """What was measured and what was derived from it. Journalled verbatim."""
    task: str = ""
    status: str = "ok"              # ok | skipped | partial
    reason: str = ""
    k: int = 0
    seed_scores: list = field(default_factory=list)
    reference_wall_s: float = 0.0

    sigma_seed: float = 0.0
    sigma_seed_ucl: float = 0.0
    sigma_eval_paired: float = 0.0
    sigma_eval_abs: float = 0.0
    quantum: float = 0.0
    sigma_policy: float = 0.0       # the seed-scale figure, floored
    sigma_delta: float = 0.0        # the replication-scale figure

    deterministic: bool = False
    verification_useful: bool = True
    paired_measurable: bool = True
    eps_provisional: bool = False

    eps: float = 0.0
    eps_source: str = "derived"     # derived | declared
    eps_derived: float = 0.0
    declared_eps_in_sigma: float = 0.0
    verify_margin: float = 0.0
    protection_write_off: float = 0.0

    def to_dict(self):
        return asdict(self)

    def summary(self):
        if self.status != "ok":
            return f"calibration {self.status}: {self.reason}"
        det = " (deterministic)" if self.deterministic else ""
        return (f"sigma_seed {self.sigma_seed:.5f}{det}, "
                f"sigma_eval_paired {self.sigma_eval_paired:.5f}, "
                f"sigma_delta {self.sigma_delta:.5f} "
                f"-> eps {self.eps:.5f} ({self.eps_source}), "
                f"verify_margin {self.verify_margin:.5f}")


def seed_budget(t_ref, max_hours, fraction=DEFAULT_CALIB_FRACTION):
    """How many seeds the budget affords.

    More seeds is not perfectionism: every threshold is linear in sigma-hat,
    whose relative SE is 1/sqrt(2(k-1)), so k=2 inflates every threshold by
    3.1x and k=10 by only 1.24x. Buying seeds buys tighter thresholds.
    """
    budget = min(CALIB_CAP_S, fraction * max_hours * 3600.0)
    if t_ref <= 0:
        return MAX_SEEDS
    k = int(budget // t_ref)
    return max(0, min(MAX_SEEDS, k))


def eval_bootstrap(task, contrib_a, contrib_b=None, b=BOOTSTRAP_B, seed=0):
    """Resample evaluation units and re-aggregate. Costs no retraining.

    The task supplies per-unit contributions and an aggregator, because a
    metric is rarely a plain mean of per-row numbers: RMSE is the root of a
    mean, and a grouped AUC is a ratio of weighted sums. Re-aggregating stored
    predictions is what makes this cost a second instead of k more fits.

    With two sets of contributions the resample indices are **shared**, which
    measures the noise of the *difference* between two designs - the quantity
    that says whether a validation gain would survive a different evaluation
    draw. That paired figure, not the absolute one, is what the stopping rule
    should be scaled against.
    """
    try:
        import numpy as np
    except ImportError:
        return 0.0
    n = task.eval_n_units(contrib_a)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(b):
        idx = rng.integers(0, n, n)
        v = task.eval_aggregate(contrib_a, idx)
        if contrib_b is not None:
            v -= task.eval_aggregate(contrib_b, idx)
        vals.append(v)
    return float(statistics.stdev(vals)) if len(vals) > 1 else 0.0


def calibrate(task, run_dir, max_hours=6.0, timeout_s=600, log=None,
              fraction=DEFAULT_CALIB_FRACTION):
    """Run the reference on k seeds, measure the noise, derive the constants.

    Never raises for a task-side problem: a missing or broken reference
    downgrades to `status="skipped"` and the caller keeps the task's declared
    constants. The one thing worth failing on - a reference that cannot run at
    all - is the caller's decision, not this function's.
    """
    say = log or (lambda *a, **k: None)
    cal = Calibration(task=getattr(task, "name", "unnamed"))

    ref = task.reference_implementation()
    if ref is None or not Path(ref).exists():
        cal.status = "skipped"
        cal.reason = ("no reference implementation, so the noise floor cannot "
                      "be measured before the search starts")
        return cal

    work = Path(run_dir) / "calibration"
    work.mkdir(parents=True, exist_ok=True)
    argv = getattr(task, "reference_argv", task.candidate_argv)

    # --- one fit, to find out what a fit costs -----------------------------
    t0 = time.monotonic()
    out0 = work / "ref_seed0.csv"
    r = run_script(ref, argv(task.visible_data_dir(), "valid", out0, 0),
                   timeout_s=timeout_s, extra_path=task.extra_sys_path())
    t_ref = time.monotonic() - t0
    cal.reference_wall_s = round(t_ref, 2)
    if not r.ok:
        cal.status = "skipped"
        cal.reason = f"reference implementation failed: {r.summary()}"
        return cal

    k = seed_budget(t_ref, max_hours, fraction)
    if k < MIN_SEEDS:
        cal.status = "skipped"
        cal.reason = (f"reference takes {t_ref:.0f}s; the calibration budget "
                      f"affords {k} seeds, fewer than the {MIN_SEEDS} needed")
        return cal
    say(f"  calibrating: reference {t_ref:.1f}s, running {k} seeds")

    # --- the seeds ---------------------------------------------------------
    scores, outs, failures = [], [], 0
    for seed in range(k):
        out = work / f"ref_seed{seed}.csv"
        if seed > 0:
            rr = run_script(ref, argv(task.visible_data_dir(), "valid", out, seed),
                            timeout_s=timeout_s, extra_path=task.extra_sys_path())
            if not rr.ok:
                failures += 1
                continue
        try:
            s = task.validate_and_score(out, split="valid",
                                        data_dir=task.visible_data_dir())
        except Exception as e:                       # task-side, never fatal
            failures += 1
            say(f"    seed {seed}: scoring failed ({type(e).__name__})")
            continue
        scores.append(s.primary)
        outs.append(out)

    if len(scores) < MIN_SEEDS:
        cal.status = "skipped"
        cal.reason = (f"only {len(scores)} of {k} reference runs scored; "
                      f"cannot estimate a spread")
        return cal
    if failures:
        cal.status = "partial"
        cal.reason = f"{failures} of {k} reference runs failed"

    cal.k = len(scores)
    cal.seed_scores = [round(v, 6) for v in scores]

    # --- the two sigmas ----------------------------------------------------
    cal.sigma_seed = spread(scores)
    cal.sigma_seed_ucl = cal.sigma_seed * _ucl_factor(cal.k)
    cal.deterministic = (cal.sigma_seed == 0.0)
    cal.quantum = (task.metric_granularity
                   or detect_quantum(scores) or 0.0)

    contribs = getattr(task, "eval_contributions", None)
    if contribs is not None and len(outs) >= 2:
        try:
            ca, cb = contribs(outs[0]), contribs(outs[1])
            cal.sigma_eval_abs = eval_bootstrap(task, ca)
            cal.sigma_eval_paired = eval_bootstrap(task, ca, cb)
        except Exception as e:
            say(f"    eval bootstrap unavailable ({type(e).__name__}: {e})")

    # The paired bootstrap compares two runs of the *same* design. On a
    # deterministic task those runs are identical, so it measures exactly zero
    # - an artifact of what was compared, not a claim that the task has no
    # evaluation noise. Two genuinely different designs on the same finite
    # eval set would still differ; we simply have only one design here.
    cal.paired_measurable = not cal.deterministic
    eval_term = cal.sigma_eval_paired

    floor = max(cal.quantum / math.sqrt(12) if cal.quantum else 0.0,
                eval_term,
                64 * abs(scores[0]) * sys.float_info.epsilon)
    cal.sigma_policy = max(cal.sigma_seed_ucl, floor)
    cal.sigma_delta = math.sqrt(2 * cal.sigma_policy ** 2 + eval_term ** 2)

    # --- the constants -----------------------------------------------------
    if cal.deterministic:
        # Re-running a seed returns the identical number, so seed verification
        # buys no information and should not be paid for. That is the single
        # largest behavioural win here, and it is only available because the
        # noise floor was measured rather than assumed.
        cal.verification_useful = False
        cal.verify_margin = max(cal.quantum, 0.0)
        # The only thing that can make a gain fail to replicate is the eval
        # draw, and its *paired* magnitude cannot be measured from one design.
        # The absolute bootstrap is an upper bound on it (common variation no
        # longer cancels), and for eps an upper bound is the dangerous
        # direction - too wide stops a search that is still improving. So take
        # it as provisional, scale it down to the smallest defensible figure
        # rather than the largest, and mark it for online refinement once two
        # genuinely different designs have been scored.
        cal.eps_provisional = True
        cal.sigma_delta = cal.sigma_eval_abs / math.sqrt(2)
        cal.eps_derived = max(EPS_Z * cal.sigma_delta, cal.quantum)
        if cal.eps_derived <= 0.0:
            # Deterministic *and* no eval bootstrap: there is genuinely no
            # measurement to work from. A zero epsilon is not a safe answer -
            # it makes convergence demand exactly no improvement - so fall
            # back to a small fraction of the metric's own magnitude and say
            # so loudly. A task that lands here should implement
            # eval_contributions.
            cal.eps_derived = max(RELATIVE_EPS_FLOOR * abs(scores[0]),
                                  sys.float_info.epsilon)
            cal.reason = (cal.reason + "; " if cal.reason else "") + (
                "no noise could be measured (deterministic reference and no "
                "eval bootstrap); eps is a relative-magnitude fallback")
    else:
        cal.verify_margin = max(VERIFY_Z * cal.sigma_policy, cal.quantum)
        cal.eps_derived = max(EPS_Z * cal.sigma_delta, cal.quantum)
    if getattr(task, "declared_eps", None):
        cal.eps = float(task.declared_eps)
        cal.eps_source = "declared"
        if cal.sigma_delta > 0:
            cal.declared_eps_in_sigma = cal.eps / cal.sigma_delta
    else:
        cal.eps = cal.eps_derived
        cal.eps_source = "derived"
    # "Could this lineage still reach the incumbent?" is a step-size question,
    # floored so it never writes off a lineage that is inside the noise.
    from agent.state import PROTECTED_SCORED_ATTEMPTS
    cal.protection_write_off = max(PROTECTED_SCORED_ATTEMPTS * cal.eps,
                                   2 * cal.sigma_policy * math.sqrt(2))
    return cal


def write(cal, run_dir):
    p = Path(run_dir) / "calibration.json"
    p.write_text(json.dumps(cal.to_dict(), indent=2) + "\n")
    return p
