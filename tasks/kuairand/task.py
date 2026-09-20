"""KuaiRand-Pure, behind the Task interface.

This is the benchmark the agent was originally built for, and the frozen
hackathon result was produced against it. Everything here was previously
spread across agent/prompts.py, agent/scorer.py, agent/firewall.py and
agent/paths.py; nothing about the behaviour changes, which is the point - the
refactor is verified by replaying both recorded runs and reproducing the
sealed test score, so any behavioural drift is a bug.

The test-split firewall is the one piece worth reading if you are new to this:
KuaiRand-Pure ships the test-window labels in the public download, so "hidden"
test data is hidden by convention only. `prepare()` materialises a data
directory that physically contains no test-window row, and the guard patterns
below reject any candidate that tries to reach the real one by absolute path.
"""
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent import scorer
from agent.paths import (CANDIDATES, EVALUATE_PY, REAL_DATA, STARTER,
                         VISIBLE_DATA)
from agent.task import Task, load_metric_module
from tasks.kuairand import briefing as kr


class KuaiRandTask(Task):
    name = "kuairand-pure"
    metrics = ("GAUC", "nDCG@5")
    higher_is_better = True
    group_label = "users"

    # The organizers' convergence epsilon. Declared, therefore honoured:
    # calibration reports how many sigma it works out to but must not replace
    # it. Silently substituting our own stopping rule for a competition's
    # stated one would invalidate the result it produced.
    declared_eps = 0.002

    # Continuous metric over ~125k rows; no meaningful granularity floor.
    metric_granularity = None

    # ------------------------------------------------------------------ data
    def prepare(self, force=False, verbose=True):
        from tasks.kuairand import firewall
        return firewall.build(force=force, verbose=verbose)

    def visible_data_dir(self):
        return VISIBLE_DATA

    def verify_isolation(self, verbose=False):
        from agent.verify_firewall import verify
        return verify(verbose=verbose)

    def extra_sys_path(self):
        return [STARTER]

    # --------------------------------------------------------------- prompts
    def system_prompt(self):
        return kr.SYSTEM

    def briefing(self):
        return kr.briefing()

    # --------------------------------------------------------------- running
    def reference_implementation(self):
        return CANDIDATES / "fm_baseline.py"

    # --------------------------------------------------------------- scoring
    def integrity_check(self):
        return scorer.assert_evaluate_untouched()

    def validate_and_score(self, out_path, split="valid", data_dir=None,
                           allow_holdout=False):
        return scorer.score_file(out_path, split=split, data_dir=data_dir,
                                 allow_test=allow_holdout)

    # ------------------------------------------------- evaluation-set noise
    def eval_contributions(self, out_path, split="valid"):
        """Per-user contributions, so the metric can be recomputed on a
        resample of users without retraining anything.

        GAUC is a positives-weighted mean of per-user AUC and nDCG@5 is a flat
        mean over users, so a user-level resample reproduces the official
        metric exactly rather than approximating it.
        """
        import collections
        import numpy as np
        ev = load_metric_module(EVALUATE_PY)
        auc, ndcg_at_k = ev.auc, ev.ndcg_at_k

        rows = scorer.load_eval_rows(self.visible_data_dir(), split)
        preds = scorer.read_scores(out_path, rows)
        byu = collections.defaultdict(list)
        for r, s in zip(rows, preds):
            byu[r[1]].append((s, r[6]))

        gn, gd, nd = [], [], []
        for u in sorted(byu):
            lst = sorted(byu[u], key=lambda x: -x[0])
            labs = [y for _, y in lst]
            npos = sum(labs)
            if 0 < npos < len(labs):
                gn.append(npos * auc(labs, [s for s, _ in lst]))
                gd.append(float(npos))
            else:
                gn.append(0.0)
                gd.append(0.0)
            nd.append(ndcg_at_k(labs, 5))
        return {"gn": np.asarray(gn), "gd": np.asarray(gd),
                "nd": np.asarray(nd)}

    def eval_n_units(self, contributions):
        return len(contributions["nd"])

    def eval_aggregate(self, contributions, idx):
        den = contributions["gd"][idx].sum()
        gauc = contributions["gn"][idx].sum() / den if den else 0.5
        return float((gauc + contributions["nd"][idx].mean()) / 2.0)

    # ----------------------------------------------------------------- guard
    def guard_patterns(self):
        forbidden = [
            (re.compile(re.escape(str(REAL_DATA))),
             "hard-codes the sealed data directory"),
            (re.compile(re.escape(str(STARTER / "KuaiRand-Pure"))),
             "hard-codes the original dataset directory"),
            (re.compile(r"""KuaiRand-Pure[/\\]data"""),
             "references the original dataset path"),
        ]
        # Reported, never rejected. Against the visible directory `test` is an
        # empty list, so every mention of the split is inert - an early version
        # rejected them and burned an iteration on a line copied verbatim from
        # the organizer's own data.py. See agent/guard.py.
        suspicious = [
            (re.compile(r"""split\s*=\s*['"]test['"]"""), "selects split='test'"),
            (re.compile(r"""--split['"]?\s*,\s*['"]test['"]"""),
             "passes --split test"),
        ]
        return forbidden, suspicious


TASK = KuaiRandTask()
