"""The whole loop, start to finish, with no API key and no dataset.

Until the synthetic task existed this was impossible: every test exercised the
policy objects in isolation, and the only end-to-end check (harness_check.py)
needed the real dataset and could not run in CI. So the parts that had failed
in production - the draft/improve/debug state machine, convergence, the
journal - were the parts with no automated coverage.

The model is stubbed. That is the point: these tests are about the loop's
behaviour given responses, not about the responses.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.llm import Backend, LLMResponse, Usage
from agent.loop import AgentLoop
from tasks.synthetic.task import SyntheticTask

GOOD = '''
HYPOTHESIS: ridge regression on the raw features, with an interaction term.

```python
import argparse, csv
from pathlib import Path
import numpy as np

def load(d, s):
    with open(Path(d) / f"{s}.csv", newline="") as fh:
        rd = csv.reader(fh); head = next(rd)
        has = head[-1] == "target"
        X, y = [], []
        for r in rd:
            v = [float(x) for x in r]
            if has: X.append(v[:-1]); y.append(v[-1])
            else: X.append(v)
    return np.asarray(X), (np.asarray(y) if y else None)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--split", default="valid")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
np.random.seed(a.seed)

def feats(X):
    return np.hstack([X, (X[:, [0]] * X[:, [1]]), np.ones((len(X), 1))])

Xtr, ytr = load(a.data_dir, "train"); Xev, _ = load(a.data_dir, a.split)
A, B = feats(Xtr), feats(Xev)
w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ ytr)
p = B @ w
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["row_id", "score"])
    for i, v in enumerate(p): wr.writerow([i, f"{v:.6f}"])
```
'''

BROKEN_SYNTAX = "HYPOTHESIS: a typo.\n\n```python\ndef f(:\n    pass\n```\n"
NO_CODE = "HYPOTHESIS: I will think about it.\n\nNo code this time.\n"


class StubBackend(Backend):
    """Replays a fixed script of responses, cycling on the last one."""
    name = "stub"
    model = "stub-1"

    def __init__(self, responses, finish_reason="STOP"):
        self.responses = list(responses)
        self.finish_reason = finish_reason
        self.calls = 0

    def complete(self, system, user, **kw):
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return LLMResponse(text=text, model=self.model,
                           usage=Usage(prompt_tokens=10, completion_tokens=20,
                                       total_tokens=30, calls=1),
                           finish_reason=self.finish_reason)


def make_loop(tmp_path, responses, **kw):
    data = tmp_path / "data"
    task = SyntheticTask(data_dir=data)
    task.prepare(force=True, verbose=False)
    loop = AgentLoop(run_dir=tmp_path / "run", task=task, skip_eda=True,
                     max_iterations=kw.pop("max_iterations", 4),
                     candidate_timeout_s=120,
                     calibrate_policy=kw.pop("calibrate_policy", False), **kw)
    loop.llm.backend = StubBackend(responses)
    loop.llm.fallbacks = []
    return loop


# ------------------------------------------------------------------ the loop
def test_a_run_completes_and_produces_a_best_node(tmp_path):
    loop = make_loop(tmp_path, [GOOD])
    best = loop.run()
    assert best is not None and not best.is_buggy
    assert best.score is not None


def test_the_journal_and_state_are_written(tmp_path):
    loop = make_loop(tmp_path, [GOOD])
    loop.run()
    assert (loop.dir / "state.json").exists()
    assert (loop.dir / "run_log.md").exists()
    events = [json.loads(l) for l in (loop.dir / "journal.jsonl").read_text().splitlines()]
    kinds = {e["event"] for e in events}
    assert {"run_start", "preflight", "iteration_start", "node_added",
            "run_end"} <= kinds


def test_scores_are_stored_oriented_and_displayed_native(tmp_path):
    """RMSE is minimised, so the policy compares -RMSE while the run log and
    every human-facing number stay in the task's own units."""
    loop = make_loop(tmp_path, [GOOD])
    best = loop.run()
    assert loop.task.sign == -1.0
    assert best.score < 0, "the policy compares an oriented score"
    assert best.native_score > 0, "display converts back to RMSE"
    assert best.native_score == pytest.approx(-best.score)


def test_the_best_node_is_the_lowest_rmse(tmp_path):
    """best() takes a max, so on a minimise task orientation is what makes it
    pick the right node. This is the assertion that would fail if the sign
    were threaded through comparisons and missed at one site."""
    loop = make_loop(tmp_path, [GOOD], max_iterations=3)
    loop.run()
    scored = [n for n in loop.state.nodes if n.score is not None]
    assert scored
    assert loop.state.best().native_score == min(n.native_score for n in scored)


# --------------------------------------------------------------- recovery
def test_a_syntax_error_is_caught_without_running_it(tmp_path):
    loop = make_loop(tmp_path, [BROKEN_SYNTAX, GOOD], max_iterations=3)
    loop.run()
    first = loop.state.get(0)
    assert first.is_buggy and "not valid Python" in first.failure_reason
    assert first.wall_s == 0.0, "a syntax error must not cost a subprocess"


def test_a_response_with_no_code_is_recorded_not_fatal(tmp_path):
    loop = make_loop(tmp_path, [NO_CODE, GOOD], max_iterations=3)
    best = loop.run()
    assert loop.state.get(0).is_buggy
    assert best is not None, "the run recovered and still produced a best node"


def test_a_run_of_only_failures_still_finishes(tmp_path):
    loop = make_loop(tmp_path, [BROKEN_SYNTAX], max_iterations=3)
    best = loop.run()
    assert best is None
    assert len(loop.state.nodes) == 3
    assert all(n.is_buggy for n in loop.state.nodes)


# ------------------------------------------------------------- calibration
def test_calibration_runs_in_preflight_and_installs_thresholds(tmp_path):
    loop = make_loop(tmp_path, [GOOD], max_iterations=2, calibrate_policy=True)
    loop.run()
    assert loop.calibration is not None
    assert loop.calibration.status == "ok"
    assert loop.policy.source == "calibrated"
    assert (loop.dir / "calibration.json").exists()
    events = [json.loads(l) for l in
              (loop.dir / "journal.jsonl").read_text().splitlines()]
    assert any(e["event"] == "calibration" for e in events), \
        "the run log must state the policy the run actually used"


def test_a_deterministic_task_turns_off_seed_verification(tmp_path):
    """jitter=0 means a re-run returns the identical number, so paying for
    verification seeds buys nothing. Only a measured noise floor can know."""
    loop = make_loop(tmp_path, [GOOD], max_iterations=2, calibrate_policy=True)
    loop.run()
    assert loop.calibration.deterministic
    assert loop.verify_seeds == []
