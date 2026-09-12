"""Regression tests for the search policy that lost run final_01.

The failure being guarded against: nodes 11-33 all took parent_id 10, twenty-
three consecutive repairs of one broken script, while nodes 4 and 7 sat scored
and unextended. 68% of the iteration budget went into a branch the search had
no way to leave. See docs/postmortem.md.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.state import Node, SolutionJournal, MAX_DEBUG_ATTEMPTS


def scored(journal, node_id, score, parent=None, stage="improve"):
    n = Node(id=node_id, stage=stage, hypothesis=f"node {node_id}", code="pass",
             parent_id=parent, is_buggy=False, seed_scores={0: score})
    return journal.add(n)


def broken(journal, node_id, parent=None, stage="improve"):
    n = Node(id=node_id, stage=stage, hypothesis=f"node {node_id}", code="pass",
             parent_id=parent, is_buggy=True, failure_reason="SyntaxError")
    return journal.add(n)


@pytest.fixture
def final_01():
    """The tree as it stood at node 10, the moment the trap closed."""
    j = SolutionJournal()
    scored(j, 0, 0.5986, stage="draft")
    scored(j, 1, 0.5135, parent=0)
    scored(j, 2, 0.5984, parent=0)
    scored(j, 3, 0.5996, parent=0)
    scored(j, 4, 0.6042, parent=3)
    scored(j, 5, 0.5908, parent=4)
    scored(j, 6, 0.6038, parent=4)
    scored(j, 7, 0.6046, parent=4)     # best
    scored(j, 8, 0.4665, parent=7)
    broken(j, 9, parent=None, stage="draft")
    broken(j, 10, parent=7)            # the node the search could not leave
    return j


def test_best_is_node_7(final_01):
    assert final_01.best().id == 7


def test_broken_child_of_best_is_repaired_first(final_01):
    """The original behaviour, which is correct for the first few attempts."""
    assert final_01.select_parent().id == 10


def test_repair_budget_is_consumed_by_debug_children(final_01):
    assert final_01.repair_attempts(10) == 0
    for i in range(MAX_DEBUG_ATTEMPTS):
        broken(final_01, 11 + i, parent=10, stage="debug")
        assert final_01.repair_attempts(10) == i + 1


def test_search_escapes_after_the_repair_budget(final_01):
    """The fix: node 10 is abandoned and the search returns to the best node."""
    for i in range(MAX_DEBUG_ATTEMPTS):
        assert final_01.select_parent().id == 10, "should still be repairing"
        broken(final_01, 11 + i, parent=10, stage="debug")

    assert final_01.is_abandoned(final_01.get(10))
    parent = final_01.select_parent()
    assert parent.id == 7, "should have gone back to the best scored node"
    assert not parent.is_buggy


def test_the_trap_cannot_recur(final_01):
    """Replay final_01's tail: 23 failed repairs, and check where they land.

    Under the old policy every one of these took parent 10. Under the fix the
    lineage is abandoned after MAX_DEBUG_ATTEMPTS and the rest go to node 7.
    """
    parents = []
    for i in range(23):
        p = final_01.select_parent()
        parents.append(p.id)
        stage = "debug" if p.is_buggy else "improve"
        broken(final_01, 11 + i, parent=p.id, stage=stage)

    assert parents[:MAX_DEBUG_ATTEMPTS] == [10] * MAX_DEBUG_ATTEMPTS
    assert 10 not in parents[MAX_DEBUG_ATTEMPTS:], \
        "node 10 was selected again after its repair budget ran out"
    assert parents.count(10) == MAX_DEBUG_ATTEMPTS


def test_a_successful_repair_clears_the_debug_state(final_01):
    """A node that gets fixed stops being a repair target and becomes a parent."""
    assert final_01.select_parent().id == 10          # broken: repair it

    final_01.get(10).is_buggy = False                 # the repair worked
    final_01.get(10).seed_scores = {0: 0.6100}        # and it is the new best

    assert final_01.best().id == 10
    parent = final_01.select_parent()
    assert parent.id == 10 and not parent.is_buggy, "should improve, not debug"


def test_a_broken_child_of_the_new_best_is_repaired_first(final_01):
    """Repair budget is per node, so a fresh breakage gets its own three tries."""
    final_01.get(10).is_buggy = False
    final_01.get(10).seed_scores = {0: 0.6100}
    broken(final_01, 11, parent=10, stage="improve")

    assert final_01.select_parent().id == 11, "new breakage should be repaired"
    for i in range(MAX_DEBUG_ATTEMPTS):
        broken(final_01, 12 + i, parent=11, stage="debug")
    assert final_01.select_parent().id == 10, "and abandoned on its own budget"


def test_abandonment_does_not_touch_healthy_nodes(final_01):
    for i in range(MAX_DEBUG_ATTEMPTS + 2):
        broken(final_01, 11 + i, parent=10, stage="debug")
    assert [n.id for n in final_01.abandoned()] == [10]
    assert not final_01.is_abandoned(final_01.get(7))


def test_improve_children_do_not_consume_repair_budget(final_01):
    """Only debug attempts count against a node's repair budget."""
    broken(final_01, 11, parent=10, stage="improve")
    assert final_01.repair_attempts(10) == 0
