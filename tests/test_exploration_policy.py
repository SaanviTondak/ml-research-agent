"""Regression tests for the exploration policy, and what it cost run final_02.

The first postmortem fixed a search that could not leave a broken node
(tests/test_search_policy.py). This file guards the defect underneath it: a
search that could not leave a *working* one.

`select_parent()` used to be a global argmax, so `improve` only ever extended
the single best node and a new lineage survived only if it won on its first
scored attempt. Run final_02 proposed DIN-style candidate attention twice and
had it measured inside the noise floor both times - node 2 lost by 0.0004
against a single-seed incumbent, node 9 by 0.0014, against a seed sigma of
0.0008 - and discarded it on the spot. Neither ever got a second iteration or
a second seed. See docs/postmortem.md.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.state import (Node, SolutionJournal, EPS, MAX_DEBUG_ATTEMPTS,
                         PROTECTED_SCORED_ATTEMPTS, PROTECTION_MAX_NODES,
                         PROTECTION_WRITE_OFF, VERIFY_MARGIN,
                         MIN_SCORED_BEFORE_CONVERGENCE)


def scored(journal, node_id, score, parent=None, stage="improve",
           scope="exploit", seeds=None):
    ss = dict(seeds) if seeds else {0: score}
    return journal.add(Node(id=node_id, stage=stage, hypothesis=f"node {node_id}",
                            code="pass", parent_id=parent, is_buggy=False,
                            seed_scores=ss, scope=scope))


def broken(journal, node_id, parent=None, stage="improve", scope="exploit"):
    return journal.add(Node(id=node_id, stage=stage, hypothesis=f"node {node_id}",
                            code="pass", parent_id=parent, is_buggy=True,
                            failure_reason="SyntaxError", scope=scope))


# final_02's real scores, from work/runs/final_02/state.json.
F02 = [(0, 0.59621, None, "draft"), (1, 0.59939, 0, "improve"),
       (2, 0.59899, 1, "improve"), (3, 0.60025, 1, "improve"),
       (4, 0.59347, 3, "improve"), (5, 0.60223, None, "draft"),
       (6, 0.60274, 5, "improve"), (7, 0.60261, 6, "improve"),
       (8, 0.60322, 6, "improve"), (9, 0.60180, 8, "improve"),
       (10, 0.60300, 8, "improve"), (11, 0.60329, 8, "improve")]


@pytest.fixture
def losing_draft():
    """final_02 up to node 4, then the draft that *didn't* get lucky.

    The real node 5 scored 0.60275 on seed 0 and took the incumbency on the
    spot. Here it scores 0.5990 instead - behind node 3's 0.60025, but well
    inside the write-off threshold. Under the old policy it would have been
    orphaned permanently; this is the case the fix exists for.
    """
    j = SolutionJournal()
    for nid, sc, parent, stage in F02[:5]:
        scored(j, nid, sc, parent=parent, stage=stage)
    scored(j, 5, 0.5990, parent=None, stage="draft", scope="explore")
    return j


# --------------------------------------------------------------- protection
def test_a_losing_draft_is_still_developed(losing_draft):
    """The whole fix: a direction that lost inside the noise gets developed."""
    j = losing_draft
    assert j.best().id == 3, "the incumbent is still the old lineage"
    assert j.protected_root() == 5
    assert j.select_parent().id == 5, "should develop the new root, not the best"


def test_protection_expires_after_k_scored_attempts(losing_draft):
    j = losing_draft
    nid = 6
    for i in range(PROTECTED_SCORED_ATTEMPTS - 1):
        assert j.protected_root() == 5
        parent = j.select_parent()
        assert j.root_of(parent.id) == 5, "selection stayed inside the lineage"
        scored(j, nid, 0.5991 + i * 0.0001, parent=parent.id, scope="explore")
        nid += 1
    assert j.protected_root() is None, "budget spent"
    assert j.select_parent().id == 3, "back to the global best"


def test_protection_is_not_consumed_by_crashes(losing_draft):
    """A crash routes to debug and burns no scored slot - but does cost a node."""
    j = losing_draft
    broken(j, 6, parent=5, scope="explore")
    assert j.protected_root() == 5
    assert j.select_parent().id == 6, "repair the breakage inside the lineage"
    for i in range(MAX_DEBUG_ATTEMPTS):
        broken(j, 7 + i, parent=6, stage="debug", scope="explore")
    # The repair budget is exhausted, but the lineage still has its own best.
    assert j.is_abandoned(j.get(6))
    assert j.protected_root() is None, "PROTECTION_MAX_NODES reached"
    assert len(j.lineage(5)) >= PROTECTION_MAX_NODES


def test_a_hopeless_lineage_is_written_off_immediately():
    """Protection is for losing inside the noise, not for losing outright."""
    j = SolutionJournal()
    scored(j, 0, 0.5986, stage="draft")
    scored(j, 1, 0.5135, parent=None, stage="draft", scope="explore")
    assert 0.5986 - 0.5135 > PROTECTION_WRITE_OFF
    assert j.protected_root() is None
    assert j.select_parent().id == 0, "no budget for a lineage this far back"


def test_a_winning_draft_needs_no_protection():
    """final_02's real node 5 won outright. Behaviour must be unchanged."""
    j = SolutionJournal()
    for nid, sc, parent, stage in F02[:6]:
        scored(j, nid, sc, parent=parent, stage=stage)
    assert j.best().id == 5
    assert j.protected_root() is None, "the explorer is the incumbent"
    assert j.select_parent().id == 5


def test_only_one_lineage_is_protected_at_a_time():
    j = SolutionJournal()
    scored(j, 0, 0.5986, stage="draft")
    scored(j, 1, 0.5980, parent=None, stage="draft", scope="explore")
    scored(j, 2, 0.5982, parent=None, stage="draft", scope="explore")
    assert j.protected_root() == 2, "only the newest root is a candidate"
    assert j.select_parent().id == 2


def test_a_broken_draft_root_does_not_preempt_an_in_flight_repair():
    """Protection is earned by evidence, not merely by being new.

    final_01's node 9 was a draft that crashed. It is tempting to give it a
    repair budget - it never got one in the real run - but a lineage with no
    score has produced no measurement to protect, and repairing someone
    else's broken draft costs an iteration for less than a fresh draft buys.
    A crashed draft is replaced by the draft schedule, not repaired.
    """
    j = SolutionJournal()
    scored(j, 0, 0.5986, stage="draft")
    scored(j, 7, 0.6046, parent=0)
    broken(j, 9, parent=None, stage="draft")     # the crashed exploration
    broken(j, 10, parent=7)                      # a fresh breakage off best
    assert j.protected_root() is None
    assert j.select_parent().id == 10, "the incumbent's own repair comes first"


# -------------------------------------------------------------- convergence
def _twelve_flat_exploit():
    """Twelve scored exploit attempts with no >EPS step: a converged search."""
    j = SolutionJournal()
    scored(j, 0, 0.6000, stage="draft")
    for i in range(1, 12):
        scored(j, i, 0.6000 + i * 0.0001, parent=i - 1)
    return j


def test_final_02_converges_today_at_twelve():
    """Pin the baseline, so the effect of any future change is measurable."""
    j = SolutionJournal()
    for i, (nid, sc, parent, stage) in enumerate(F02):
        scored(j, nid, sc, parent=parent, stage=stage)
        if i < len(F02) - 1:
            assert not j.has_converged(), f"converged early at node {nid}"
    assert len(j.good()) == MIN_SCORED_BEFORE_CONVERGENCE
    assert j.has_converged()


def test_protection_blocks_convergence():
    j = _twelve_flat_exploit()
    assert j.has_converged(), "precondition: converged before exploring"
    scored(j, 12, j.best().score - 0.001, parent=None, stage="draft",
           scope="explore")
    assert j.protected_root() == 12
    assert not j.has_converged(), "an exploration in budget blocks the rule"


def test_the_floor_counts_exploit_nodes_only():
    """Buying an exploration must never bring the stopping rule closer."""
    j = SolutionJournal()
    scored(j, 0, 0.6000, stage="draft")
    for i in range(1, 8):
        scored(j, i, 0.6000 + i * 0.0001, parent=i - 1)
    for i in range(8, 12):                       # four explorations
        scored(j, i, 0.5999, parent=0, scope="explore")
    assert len(j.good()) == 12
    assert not j.has_converged(), "explore nodes must not fill the floor"
    for i in range(12, 16):
        scored(j, i, 0.6007, parent=7)
    assert len([n for n in j.good() if n.scope == "exploit"]) >= 12
    assert j.has_converged()


def test_convergence_still_fires_when_nothing_is_protected():
    """Guards the block above against producing a run that never stops."""
    j = _twelve_flat_exploit()
    assert j.protected_root() is None
    assert j.has_converged()


def test_a_winning_exploration_still_counts_toward_the_best():
    """Scope filters which nodes get a history entry, not which ones score."""
    j = _twelve_flat_exploit()
    scored(j, 12, 0.6500, parent=None, stage="draft", scope="explore")
    assert j.best().id == 12
    assert j.improvement_history(only_scope="exploit")[-1] == pytest.approx(0.6011)
    assert j.improvement_history()[-1] == pytest.approx(0.6500)


# ------------------------------------------------------- draft cadence
def test_nodes_since_last_root_is_keyed_to_the_journal():
    """The old cadence used the loop counter, which restarts on resume."""
    j = SolutionJournal()
    scored(j, 0, 0.6000, stage="draft")
    assert j.nodes_since_last_root() == 0
    for i in range(1, 4):
        scored(j, i, 0.6000, parent=0)
    assert j.nodes_since_last_root() == 3
    scored(j, 4, 0.5999, parent=None, stage="draft", scope="explore")
    assert j.nodes_since_last_root() == 0


# ------------------------------------------------------------- persistence
def test_protection_survives_a_reload(tmp_path):
    """Policy state is derived from the journal, so a resume cannot lose it."""
    path = tmp_path / "state.json"
    j = SolutionJournal(path)
    scored(j, 0, 0.6000, stage="draft")
    scored(j, 1, 0.5990, parent=None, stage="draft", scope="explore")
    before = (j.protected_root(), j.select_parent().id)

    reloaded = SolutionJournal(path)
    assert (reloaded.protected_root(), reloaded.select_parent().id) == before
    assert reloaded.get(1).scope == "explore"


def test_a_journal_without_scope_fields_still_loads(tmp_path):
    """Both recorded runs predate these fields and must keep replaying."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"nodes": [
        {"id": 0, "stage": "draft", "hypothesis": "h", "code": "pass",
         "parent_id": None, "is_buggy": False, "seed_scores": {"0": 0.5986},
         "score": 0.5986},
    ]}))
    j = SolutionJournal(path)
    assert j.get(0).scope == "exploit", "old nodes default to exploit"
    assert j.get(0).protected_root is None
    assert j.best().id == 0


@pytest.mark.parametrize("run", ["final_01", "final_02"])
def test_the_real_recorded_runs_still_load(run):
    p = Path(__file__).resolve().parents[1] / "work" / "runs" / run / "state.json"
    if not p.exists():
        pytest.skip(f"{run} state.json not present (work/ is gitignored)")
    j = SolutionJournal(p)
    assert j.nodes and all(n.scope == "exploit" for n in j.nodes)


# ------------------------------------------------- multi-seed verification
def test_verification_anchors_on_best_verified_not_previous_best():
    """final_02 nodes 6-11: six sub-EPS gains that never triggered a check."""
    j = SolutionJournal()
    scored(j, 5, 0.60223, stage="draft",
           seeds={0: 0.60275, 1: 0.60187, 2: 0.60208})
    assert j.best_verified_score() == pytest.approx(0.60223, abs=1e-5)

    tail = [0.60274, 0.60261, 0.60322, 0.60180, 0.60300, 0.60329]
    old_trigger, new_trigger = 0, 0
    parent = 5
    for i, primary in enumerate(tail):
        if primary > j.best_score() + EPS:        # the old rule
            old_trigger += 1
        if j.needs_verification(primary, margin=VERIFY_MARGIN):
            new_trigger += 1
        scored(j, 6 + i, primary, parent=parent)
    assert old_trigger == 0, "this is the regression: nothing was ever checked"
    assert new_trigger >= 1


def test_best_verified_score_ignores_single_seed_nodes():
    j = SolutionJournal()
    scored(j, 0, 0.6000, stage="draft", seeds={0: 0.5999, 1: 0.6001})
    scored(j, 1, 0.6100, parent=0)                # one lucky draw
    assert j.best().id == 1
    assert j.best_verified_score() == pytest.approx(0.6000)


def test_verified_anchor_takes_the_max_not_the_latest():
    """A later, lower verification must not quietly lower the bar."""
    j = SolutionJournal()
    scored(j, 0, 0.6022, stage="draft", seeds={0: 0.6022, 1: 0.6022})
    scored(j, 1, 0.6010, parent=0, seeds={0: 0.6010, 1: 0.6010})
    assert j.best_verified_score() == pytest.approx(0.6022)
    assert not j.needs_verification(0.6015)
