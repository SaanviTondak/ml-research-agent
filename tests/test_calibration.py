"""Calibration: measuring the noise floor instead of declaring it.

The constants in agent/state.py were tuned against one benchmark, in that
benchmark's units. These tests check the two properties that have to hold for
the agent to survive being pointed anywhere else:

  * the derived thresholds track the task's actual noise across orders of
    magnitude, and
  * the degenerate cases - a deterministic task, a discrete metric, a broken
    or missing reference - produce something finite and sensible rather than
    zeros, NaNs or a crash.

The synthetic task exists so these can be checked against a knob rather than
against another estimate.
"""
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.calibrate import (Calibration, EPS_Z, VERIFY_Z, _ucl_factor,
                             calibrate, detect_quantum, seed_budget)
from tasks.synthetic.task import SyntheticTask


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cal_data")
    SyntheticTask(data_dir=d).prepare(force=True, verbose=False)
    return d


def run_calibration(data_dir, run_dir, jitter=0.0, quantum=0.0):
    t = SyntheticTask(data_dir=data_dir, jitter=jitter, quantum=quantum)
    return calibrate(t, run_dir, max_hours=6.0, timeout_s=120)


# ------------------------------------------------------------- the estimators
def test_the_upper_bound_prices_a_small_k():
    """sigma-hat from k samples is itself noisy; the factor makes that visible."""
    assert _ucl_factor(2) == pytest.approx(3.14, abs=0.02)
    assert _ucl_factor(3) == pytest.approx(1.86, abs=0.02)
    assert _ucl_factor(10) == pytest.approx(1.24, abs=0.02)
    assert _ucl_factor(10) < _ucl_factor(5) < _ucl_factor(3) < _ucl_factor(2)


def test_seed_budget_tracks_the_reference_cost():
    assert seed_budget(25, 6.0) == 10          # cheap reference: buy the max
    assert seed_budget(600, 6.0) < 2           # 10-min fit: cannot afford a spread
    assert seed_budget(120, 6.0) == 5


def test_detect_quantum_finds_a_grid_and_ignores_continuous_values():
    assert detect_quantum([0.842, 0.844, 0.846, 0.850]) == pytest.approx(0.002)
    assert detect_quantum([0.5961, 0.59937, 0.60101, 0.5988]) == 0.0
    assert detect_quantum([1.0]) == 0.0         # too few to tell


# ----------------------------------------------------- tracking the noise floor
@pytest.mark.parametrize("jitter", [0.001, 0.01, 0.05])
def test_the_measured_sigma_matches_the_task(data_dir, tmp_path, jitter):
    """Ground truth here is the spread the task actually produces, not the
    knob - RMSE is quadratic in the error, so the two differ by design."""
    t = SyntheticTask(data_dir=data_dir, jitter=jitter)
    cal = calibrate(t, tmp_path / "run", max_hours=6.0, timeout_s=120)
    assert cal.status == "ok"
    # seed_scores is journalled rounded, so reconstructing the stdev from it
    # differs from the unrounded computation in the last few digits.
    truth = statistics.stdev(cal.seed_scores)
    # spread() is max(stdev, IQR/1.349) - robust to a skewed metric, so never
    # meaningfully below the plain stdev and not expected to equal it.
    assert cal.sigma_seed >= truth * (1 - 1e-5)
    assert cal.sigma_seed <= 2.5 * truth
    assert cal.sigma_seed_ucl >= cal.sigma_seed


def test_thresholds_scale_with_the_noise(data_dir, tmp_path):
    """The property that makes the agent portable: a task 50x noisier gets
    thresholds roughly 50x wider, with no code change."""
    cals = {j: run_calibration(data_dir, tmp_path / f"r{j}", jitter=j)
            for j in (0.001, 0.01, 0.05)}
    eps = [cals[j].eps for j in (0.001, 0.01, 0.05)]
    ver = [cals[j].verify_margin for j in (0.001, 0.01, 0.05)]
    assert eps == sorted(eps) and ver == sorted(ver)
    ratio = cals[0.05].eps / cals[0.001].eps
    assert 10 < ratio < 1000, f"eps spanned only {ratio:.0f}x"


def test_eps_keys_off_the_replication_scale_not_the_seed_scale(data_dir, tmp_path):
    """EPS asks 'will this survive a different eval draw', so it must be
    larger than the seed-only figure VERIFY_MARGIN keys off."""
    cal = run_calibration(data_dir, tmp_path / "r", jitter=0.01)
    assert cal.sigma_delta > cal.sigma_policy
    assert cal.eps > cal.verify_margin
    assert cal.eps == pytest.approx(EPS_Z * cal.sigma_delta, rel=1e-6)
    assert cal.verify_margin == pytest.approx(VERIFY_Z * cal.sigma_policy, rel=1e-6)


# ------------------------------------------------------------ degenerate cases
def test_a_deterministic_task_does_not_produce_zero_thresholds(data_dir, tmp_path):
    """The failure this exists to prevent: sigma=0 making every gain look
    significant, the agent verifying forever and never converging."""
    cal = run_calibration(data_dir, tmp_path / "det", jitter=0.0)
    assert cal.deterministic and cal.sigma_seed == 0.0
    assert cal.eps > 0.0, "a zero epsilon would never let the search converge"
    assert cal.eps_provisional


def test_a_deterministic_task_stops_paying_for_seed_verification(data_dir, tmp_path):
    """Re-running a seed returns the identical number. Buying more is waste."""
    cal = run_calibration(data_dir, tmp_path / "det2", jitter=0.0)
    assert cal.verification_useful is False


def test_nothing_divides_by_sigma(data_dir, tmp_path):
    import math
    cal = run_calibration(data_dir, tmp_path / "det3", jitter=0.0)
    for name, v in cal.to_dict().items():
        if isinstance(v, float):
            assert math.isfinite(v), f"{name} is {v}"


def test_a_discrete_metric_never_gets_a_threshold_finer_than_one_step(
        data_dir, tmp_path):
    q = 0.01
    cal = run_calibration(data_dir, tmp_path / "q", jitter=0.001, quantum=q)
    assert cal.quantum == pytest.approx(q, rel=1e-6)
    assert cal.eps >= q
    assert cal.verify_margin >= q


# ------------------------------------------------------------------- guards
def test_a_missing_reference_skips_rather_than_crashes(data_dir, tmp_path):
    t = SyntheticTask(data_dir=data_dir)
    t.reference_implementation = lambda: None
    cal = calibrate(t, tmp_path / "none", max_hours=6.0)
    assert cal.status == "skipped" and "reference" in cal.reason


def test_a_broken_reference_skips_rather_than_crashes(data_dir, tmp_path):
    bad = tmp_path / "broken.py"
    bad.write_text("import sys\nsys.exit(3)\n")
    t = SyntheticTask(data_dir=data_dir)
    t.reference_implementation = lambda: bad
    cal = calibrate(t, tmp_path / "broke", max_hours=6.0, timeout_s=60)
    assert cal.status == "skipped" and "failed" in cal.reason


def test_a_slow_reference_skips_rather_than_guessing(data_dir, tmp_path):
    """Two seeds is the minimum that can express a spread at all."""
    t = SyntheticTask(data_dir=data_dir)
    cal = calibrate(t, tmp_path / "slow", max_hours=0.0005, timeout_s=60)
    assert cal.status == "skipped"


def test_a_declared_epsilon_is_honoured_not_overruled(data_dir, tmp_path):
    """A competition's stated stopping rule must survive calibration.
    Replacing it silently would invalidate the result it produced."""
    t = SyntheticTask(data_dir=data_dir, jitter=0.01)
    t.declared_eps = 0.002
    cal = calibrate(t, tmp_path / "declared", max_hours=6.0, timeout_s=120)
    assert cal.eps == 0.002
    assert cal.eps_source == "declared"
    assert cal.eps_derived != 0.002, "the derived value is still reported"
    assert cal.declared_eps_in_sigma > 0
