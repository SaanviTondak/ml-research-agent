"""The second task, which is what makes the Task interface a claim rather than
a shape.

A generalization demonstrated on one benchmark is not demonstrated. This task
differs from KuaiRand in every dimension the interface is supposed to absorb -
it minimizes instead of maximizes, its metric is unbounded, it has no groups,
its submission has two columns instead of four, and its noise floor is a knob.
It also needs no dataset and no API key, so unlike KuaiRand it can run in CI.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.executor import run_script
from agent.journal import _metric_table
from agent.task import ContractError, IntegrityError, Score
from tasks.synthetic.task import SyntheticTask


@pytest.fixture(scope="module")
def task(tmp_path_factory):
    t = SyntheticTask(data_dir=tmp_path_factory.mktemp("syn"))
    t.prepare(force=True, verbose=False)
    return t


def score_reference(task, out, seed=0, split="valid"):
    r = run_script(task.reference_implementation(),
                   task.reference_argv(task.data_dir, split, out, seed),
                   timeout_s=120, extra_path=task.extra_sys_path())
    assert r.ok, r.summary()
    return task.validate_and_score(out, split=split, data_dir=task.data_dir)


# -------------------------------------------------------------- the interface
def test_the_reference_runs_and_scores(task, tmp_path):
    s = score_reference(task, tmp_path / "p.csv")
    assert isinstance(s, Score)
    assert s.primary > 0 and s.rows == 2000


def test_lower_is_better(task):
    assert task.higher_is_better is False
    assert task.sign == -1.0


def test_an_ungrouped_score_does_not_double_count_rows(task, tmp_path):
    """group_label 'rows' must not collide with the row count in to_dict()."""
    d = score_reference(task, tmp_path / "p.csv").to_dict()
    assert sorted(d) == ["MAE", "RMSE", "primary", "rows", "split"]
    assert d["rows"] == 2000


def test_the_metric_table_renders_without_kuairand_metrics(task, tmp_path):
    table = _metric_table(score_reference(task, tmp_path / "p.csv").to_dict())
    assert "RMSE" in table and "primary" in table
    assert "GAUC" not in table


# ----------------------------------------------------------------- the firewall
def test_the_test_split_carries_no_target(task):
    assert task.verify_isolation()["test"] == 0


def test_scoring_the_test_split_is_refused(task, tmp_path):
    out = tmp_path / "p.csv"
    score_reference(task, out)
    with pytest.raises(IntegrityError):
        task.validate_and_score(out, split="test", data_dir=task.data_dir)


def test_the_guard_rejects_reaching_for_held_out_targets(task):
    from agent.guard import assert_clean, GuardRejection
    with pytest.raises(GuardRejection):
        assert_clean("import numpy as np\ny = np.load('_test_target.npy')\n",
                     patterns=task.guard_patterns())
    assert assert_clean("x = 1\n", patterns=task.guard_patterns()) == []


# ------------------------------------------------------------ the contract
@pytest.mark.parametrize("body,why", [
    ("row_id,score\n0,1.0\n", "too few rows"),
    ("wrong,header\n", "bad header"),
    ("row_id,score\n1,1.0\n", "row_id does not start at 0"),
    ("row_id,score\n0,nan\n", "NaN score"),
    ("row_id,score\n0,abc\n", "non-numeric score"),
])
def test_the_submission_contract_is_enforced(task, tmp_path, body, why):
    p = tmp_path / "bad.csv"
    p.write_text(body)
    with pytest.raises(ContractError):
        task.validate_and_score(p, split="valid", data_dir=task.data_dir)


def test_a_missing_output_file_is_a_contract_error(task, tmp_path):
    with pytest.raises(ContractError):
        task.validate_and_score(tmp_path / "nope.csv", split="valid",
                                data_dir=task.data_dir)


# ------------------------------------------------------- the noise-floor knob
def test_jitter_zero_is_exactly_deterministic(task, tmp_path):
    """A deterministic task is a real case the policy layer has to survive."""
    a = score_reference(task, tmp_path / "a.csv", seed=0).primary
    b = score_reference(task, tmp_path / "b.csv", seed=7).primary
    assert a == b, "jitter=0 must not vary with seed"


def test_jitter_controls_the_seed_spread(tmp_path_factory, tmp_path):
    """Monotone in the knob, spanning orders of magnitude. Not equal to it -
    RMSE is quadratic in the error, which is why the flag is not --sigma."""
    import statistics
    data = tmp_path_factory.mktemp("syn_j")
    base = SyntheticTask(data_dir=data)
    base.prepare(force=True, verbose=False)

    spreads = []
    for jitter in (0.0, 0.001, 0.01, 0.05):
        t = SyntheticTask(data_dir=data, jitter=jitter)
        vals = [score_reference(t, tmp_path / f"j{jitter}_{s}.csv", seed=s).primary
                for s in range(5)]
        spreads.append(statistics.stdev(vals))

    assert spreads[0] == 0.0
    assert spreads == sorted(spreads), f"not monotone in jitter: {spreads}"
    assert spreads[-1] > 100 * spreads[1], "knob does not span a useful range"


def test_a_discrete_metric_moves_in_steps(tmp_path_factory, tmp_path):
    data = tmp_path_factory.mktemp("syn_q")
    SyntheticTask(data_dir=data).prepare(force=True, verbose=False)
    t = SyntheticTask(data_dir=data, jitter=0.01, quantum=0.01)
    vals = {score_reference(t, tmp_path / f"q{s}.csv", seed=s).primary
            for s in range(6)}
    for v in vals:
        assert abs(v / 0.01 - round(v / 0.01)) < 1e-9, f"{v} is off the grid"
