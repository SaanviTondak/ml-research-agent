"""The run's time budget, and the clock it is charged in.

Run final_02 reported a candidate at 7141s against a 600s timeout that
correctly never fired. Both figures were right: `communicate(timeout=...)`
measures `time.monotonic()`, which does not advance while macOS sleeps, while
`elapsed_h()` measured `time.time()`, which does. So the cap was billed ~2h of
suspended host that the timeout never saw.

These tests pin the two properties that follow: the cap and the timeout now
read the same clock, and awake time accumulates across resumes instead of
each resumed segment getting a fresh six hours.
See docs/postmortem.md and docs/interventions.md.
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.llm import Backend
from agent.loop import AgentLoop


class NoBackend(Backend):
    """Never called. Injected so constructing a loop needs no API key."""
    name = "none"
    model = "none"

    def complete(self, system, user, **kw):
        raise AssertionError("these tests must not call the model")


def make_loop(run_dir):
    return AgentLoop(run_dir=run_dir, skip_eda=True, backend=NoBackend(),
                     calibrate_policy=False)


@pytest.fixture
def loop(tmp_path):
    """A loop object with no LLM, no data and no journal activity."""
    return make_loop(tmp_path / "run")


def test_elapsed_uses_the_same_clock_as_the_timeout(loop, monkeypatch):
    """Reproduces final_02's sleep event: two hours of wall, none of them awake."""
    t_wall, t_mono = time.time(), time.monotonic()
    monkeypatch.setattr(time, "time", lambda: t_wall + 7200)
    monkeypatch.setattr(time, "monotonic", lambda: t_mono)

    assert loop.elapsed_h() == pytest.approx(0.0, abs=1e-3), \
        "the cap must not be charged for a sleeping host"
    assert loop.wall_h() == pytest.approx(2.0, abs=1e-3)
    assert loop.suspended_h() == pytest.approx(2.0, abs=1e-3)


def test_awake_time_is_charged_normally(loop, monkeypatch):
    t_wall, t_mono = time.time(), time.monotonic()
    monkeypatch.setattr(time, "time", lambda: t_wall + 3600)
    monkeypatch.setattr(time, "monotonic", lambda: t_mono + 3600)

    assert loop.elapsed_h() == pytest.approx(1.0, abs=1e-3)
    assert loop.suspended_h() == pytest.approx(0.0, abs=1e-3)


def test_awake_time_accumulates_across_resumes(tmp_path, monkeypatch):
    """final_01 was resumed three times; each segment got a fresh 6h cap."""
    run_dir = tmp_path / "run"
    first = make_loop(run_dir)
    t_mono = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: t_mono + 5400)   # 1.5 h
    first._save_budget()
    assert first.elapsed_h() == pytest.approx(1.5, abs=1e-3)

    monkeypatch.undo()
    resumed = make_loop(run_dir)
    assert resumed.prior_awake_s == pytest.approx(5400, abs=1.0)
    assert resumed.elapsed_h() == pytest.approx(1.5, abs=1e-2), \
        "a resume must not reset the run's time budget"


def test_a_missing_or_corrupt_budget_file_is_not_fatal(tmp_path):
    run_dir = tmp_path / "run"
    fresh = make_loop(run_dir)
    assert fresh.prior_awake_s == 0.0

    fresh.budget_path.write_text("{not json")
    assert make_loop(run_dir).prior_awake_s == 0.0


def test_the_budget_file_records_both_clocks(loop):
    loop._save_budget()
    d = json.loads(loop.budget_path.read_text())
    assert {"awake_s", "wall_s", "segment_started"} <= set(d)
