"""The bring-your-own-CSV task: the one that needs no Python from the user.

The Task interface being small is not the same as the project being usable by
someone else. These tests cover the path a newcomer actually takes - point it
at a directory of CSVs, name a column, pick a metric - and in particular the
part they are most likely to get wrong and least likely to notice: the
held-out labels must be absent from disk, not merely ignored.
"""
import csv
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent import registry
from agent.calibrate import calibrate
from agent.executor import run_script
from agent.task import ContractError, IntegrityError
from tasks.tabular.task import TabularTask

HEAD = ["tenure", "plan", "spend", "tickets", "region", "churn"]


def write_csvs(d, n_train=800, n_eval=200):
    """Mixed numeric and categorical, with missing values - like real data."""
    rng = random.Random(7)
    d.mkdir(parents=True, exist_ok=True)

    def row():
        tenure, plan = rng.randint(1, 72), rng.choice(["basic", "pro", "ent"])
        spend, tickets = round(rng.uniform(5, 200), 2), rng.randint(0, 9)
        z = (-2.5 + 0.045 * tickets - 0.02 * tenure + 0.008 * spend
             + {"basic": 0.8, "pro": 0.0, "ent": -0.6}[plan])
        churn = 1 if rng.random() < 1 / (1 + 2.718 ** -z) else 0
        return [tenure, plan, spend if rng.random() > 0.05 else "",
                tickets, rng.choice(["na", "emea"]), churn]

    for split, n in (("train", n_train), ("valid", n_eval), ("test", n_eval)):
        with open(d / f"{split}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(HEAD)
            for _ in range(n):
                w.writerow(row())
    return d


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    return write_csvs(tmp_path_factory.mktemp("csvs"))


def make(source, tmp_path, metric="auc", **kw):
    t = TabularTask(source=source, target="churn", metric=metric,
                    data_dir=tmp_path / f"visible_{metric}", **kw)
    t.prepare(force=True, verbose=False)
    return t


def predict(task, out, seed=0, split="valid"):
    r = run_script(task.reference_implementation(),
                   task.reference_argv(task.visible_data_dir(), split, out, seed),
                   timeout_s=120, extra_path=task.extra_sys_path())
    assert r.ok, r.stderr[-600:]
    return out


# --------------------------------------------------------------- the firewall
def test_the_holdout_target_is_absent_from_disk(source, tmp_path):
    """Absent, not masked. Nothing on disk to read."""
    t = make(source, tmp_path)
    with open(t.visible_data_dir() / "test.csv", newline="") as fh:
        head = next(csv.reader(fh))
    assert "churn" not in head
    assert set(HEAD) - {"churn"} == set(head), "features must survive"
    with open(t.visible_data_dir() / "valid.csv", newline="") as fh:
        assert "churn" in next(csv.reader(fh)), "valid keeps its target"


def test_isolation_reports_the_holdout_as_unscoreable(source, tmp_path):
    assert make(source, tmp_path).verify_isolation()["test"] == 0


def test_scoring_the_holdout_is_refused(source, tmp_path):
    t = make(source, tmp_path)
    out = predict(t, tmp_path / "p.csv")
    with pytest.raises(IntegrityError):
        t.validate_and_score(out, split="test", data_dir=t.visible_data_dir())


def test_the_guard_rejects_reaching_for_the_stashed_labels(source, tmp_path):
    from agent.guard import assert_clean, GuardRejection
    t = make(source, tmp_path)
    with pytest.raises(GuardRejection):
        assert_clean("d = open('_holdout_visible_auc.json').read()\n",
                     patterns=t.guard_patterns())


def test_a_missing_target_column_is_caught_at_prepare(source, tmp_path):
    t = TabularTask(source=source, target="not_a_column", metric="auc",
                    data_dir=tmp_path / "bad")
    with pytest.raises(IntegrityError):
        t.prepare(force=True, verbose=False)


# ------------------------------------------------------------- the metrics
@pytest.mark.parametrize("metric,higher", [
    ("rmse", False), ("mae", False), ("logloss", False),
    ("auc", True), ("accuracy", True)])
def test_every_metric_runs_and_orients_itself(source, tmp_path, metric, higher):
    """Direction comes from the metric, not from the user."""
    t = make(source, tmp_path, metric=metric)
    assert t.higher_is_better is higher
    assert t.sign == (1.0 if higher else -1.0)
    s = t.validate_and_score(predict(t, tmp_path / f"{metric}.csv"),
                             split="valid", data_dir=t.visible_data_dir())
    assert s.primary == s.primary                     # not NaN
    assert metric in s.to_dict()


def test_the_reference_is_better_than_chance(source, tmp_path):
    t = make(source, tmp_path, metric="auc")
    s = t.validate_and_score(predict(t, tmp_path / "p.csv"), split="valid",
                             data_dir=t.visible_data_dir())
    assert s.primary > 0.6, "a linear baseline should find this signal"


# ------------------------------------------------------------- the contract
@pytest.mark.parametrize("body", [
    "row_id,score\n0,1.0\n",            # too few rows
    "wrong,header\n",                   # bad header
    "row_id,score\n1,1.0\n",            # row_id does not start at 0
    "row_id,score\n0,nan\n",            # NaN
])
def test_the_submission_contract_is_enforced(source, tmp_path, body):
    t = make(source, tmp_path)
    p = tmp_path / "bad.csv"
    p.write_text(body)
    with pytest.raises(ContractError):
        t.validate_and_score(p, split="valid", data_dir=t.visible_data_dir())


# ------------------------------------------------------------- calibration
@pytest.mark.parametrize("metric", ["auc", "rmse", "accuracy"])
def test_calibration_never_yields_a_zero_epsilon(source, tmp_path, metric):
    """The reference here is deterministic, so sigma_seed is exactly 0. An
    eps of 0 would make convergence demand exactly no improvement - the
    evaluation bootstrap is what keeps it meaningful."""
    t = make(source, tmp_path, metric=metric)
    cal = calibrate(t, tmp_path / f"cal_{metric}", max_hours=6.0, timeout_s=120)
    assert cal.status == "ok"
    assert cal.deterministic, "a ridge/logistic baseline has no randomness"
    assert cal.eps > 0.0
    assert cal.sigma_eval_abs > 0.0, "eval noise must still be measured"
    assert cal.verification_useful is False


def test_auc_is_bootstrapped_like_any_other_metric(source, tmp_path):
    """A rank statistic is recomputed on the resample, not re-averaged."""
    t = make(source, tmp_path, metric="auc")
    c = t.eval_contributions(predict(t, tmp_path / "p.csv"))
    assert c is not None
    import numpy as np
    n = t.eval_n_units(c)
    whole = t.eval_aggregate(c, np.arange(n))
    half = t.eval_aggregate(c, np.arange(n // 2))
    assert 0.0 <= whole <= 1.0 and 0.0 <= half <= 1.0
    assert whole != half, "a subsample should move a rank statistic"


# ----------------------------------------------------------------- the CLI
class Args:
    def __init__(self, **kw):
        self.task = "tabular"
        self.data = self.target = None
        self.metric, self.holdout = "rmse", "test"
        self.__dict__.update(kw)


def test_the_cli_builds_and_prepares_a_task(source, tmp_path, monkeypatch):
    # TabularTask binds WORK at import time, so redirect it on the module the
    # registry will actually construct from - otherwise this test writes into
    # the real work/ directory.
    import tasks.tabular.task as tt
    monkeypatch.setattr(tt, "WORK", tmp_path)
    t = registry.build(Args(data=str(source), target="churn", metric="auc"))
    assert tmp_path in t.visible_data_dir().parents
    assert t.target == "churn" and t.metric == "auc"
    assert t.verify_isolation()["test"] == 0


@pytest.mark.parametrize("kw,expect", [
    ({}, "needs --data and --target"),
    ({"data": "/definitely/not/here", "target": "churn"}, "not a directory"),
])
def test_the_cli_explains_mistakes_instead_of_raising(kw, expect):
    with pytest.raises(SystemExit) as e:
        registry.build(Args(**kw))
    assert expect in str(e.value)


def test_an_unknown_metric_is_a_message_not_a_traceback(source):
    with pytest.raises(SystemExit) as e:
        registry.build(Args(data=str(source), target="churn", metric="f1"))
    assert "unknown metric" in str(e.value)


def test_an_unknown_target_lists_the_real_columns(source):
    with pytest.raises(SystemExit) as e:
        registry.build(Args(data=str(source), target="nope"))
    assert "tenure" in str(e.value) and "churn" in str(e.value)
