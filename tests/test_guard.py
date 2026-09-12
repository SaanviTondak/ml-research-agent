"""Tests for the pre-execution checks on agent-written code.

Two independent concerns live in agent/guard.py:
  * assert_clean  - can this code reach data outside the visible directory?
  * assert_parses - is this code valid Python at all?
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.guard import (assert_clean, assert_parses, check_code,
                         GuardRejection, SyntaxRejection)
from agent.paths import REAL_DATA

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- syntax gate
def test_valid_script_parses():
    assert_parses("import sys\n\ndef main():\n    return 1\n")


def test_truncated_script_is_rejected():
    """The exact shape of run final_01's truncations: cut off mid-statement."""
    with pytest.raises(SyntaxRejection):
        assert_parses("def train(x):\n    for i in range(10):\n        y = x[")


@pytest.mark.parametrize("src", [
    "def f(:\n    pass",
    "if True\n    pass",
    "x = (1, 2\ny = 3",
    "class A\n    pass",
])
def test_syntax_errors_are_rejected(src):
    with pytest.raises(SyntaxRejection):
        assert_parses(src)


def test_rejection_names_the_line_and_disclaims_truncation():
    """The message is a repair prompt, so it must not repeat final_01's
    mistake of blaming the output token limit for a plain syntax error."""
    with pytest.raises(SyntaxRejection) as e:
        assert_parses("a = 1\nb = 2\ndef f(:\n    pass")
    msg = str(e.value)
    assert "line 3" in msg
    assert "not a truncated response" in msg


def test_syntax_gate_runs_nothing():
    """compile() must not execute the candidate."""
    marker = REPO / "tests" / "_should_not_exist.tmp"
    assert_parses(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    assert not marker.exists()


# ------------------------------------------------------------- data firewall
def test_reference_candidate_passes_both_checks():
    src = (REPO / "candidates" / "fm_baseline.py").read_text()
    assert_parses(src)
    assert_clean(src)


def test_submitted_candidate_passes_both_checks():
    src = (REPO / "candidates" / "agent_best.py").read_text()
    assert_parses(src)
    assert_clean(src)


def test_hardcoded_sealed_path_is_rejected():
    with pytest.raises(GuardRejection):
        assert_clean(f"data_dir = {str(REAL_DATA)!r}\n")


def test_dataset_path_by_name_is_rejected():
    with pytest.raises(GuardRejection):
        assert_clean("df = load('KuaiRand-Pure/data/log_standard.csv')\n")


def test_mentioning_the_test_split_is_a_warning_not_a_rejection():
    """Regression: rejecting this cost a live iteration, on a line copied
    verbatim from the organizers' own data.py."""
    code = '"test": (20220429, 20220508),\n'
    rejections, warnings = check_code(code)
    assert rejections == []
    assert_clean(code)


def test_organizer_baseline_is_rejected_as_documented():
    """baseline.py defaults --data_dir to the real dataset; the agent is given
    candidates/fm_baseline.py instead, which has no default."""
    src = (REPO / "kuairand-starter-kit" / "baseline.py").read_text()
    with pytest.raises(GuardRejection):
        assert_clean(src)
