"""Step 3d - the loop. This is the agent.

Greedy tree search over solutions, in the style of AIDE. Each iteration picks
one node to work from, asks the model for a complete script, runs it, scores
it on validation, and records the outcome. The next iteration sees everything
that came before.

Policy
------
  explore   once, at the start: the agent writes and runs its own EDA, and its
            findings are injected into every later prompt.
  draft     a genuinely new approach. Used for the first N_DRAFTS iterations
            and occasionally afterwards, so the search does not collapse onto
            one lineage early.
  debug     the selected node crashed; fix it.
  improve   the selected node works; make one attributable change.

Selection is greedy on the best validated node, except that an unfixed crash
descending from it is repaired first - otherwise a broken child is abandoned
and the loop re-improves the same parent repeatedly, wasting iterations on
ground it has already covered.

Promotion requires agreement across seeds. A node that beats the incumbent on
seed 0 is re-run on further seeds and kept at the mean; the benchmark's
seed-to-seed std is 0.0008, so a single-seed gain below ~0.002 is noise. This
is the difference between an agent that improves and one that chases luck.

Failure policy
--------------
Nothing an iteration does may end the run. A model that returns prose instead
of code, a script that crashes, hangs, or writes a malformed submission, a
transient API error - each is recorded as a failed node and becomes context
for the next iteration. Only two things stop the loop early: exhausted API
quota, which no amount of waiting fixes, and the convergence rule.

Every event is written to the journal as it happens, and flushed. The run log
is a graded deliverable and must not be reconstructed afterwards.
"""
import argparse
import difflib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.executor import run_script
from agent.guard import (assert_clean, assert_parses, GuardRejection,
                         SyntaxRejection)
from agent.journal import Journal, new_run_dir, render_markdown
from agent.llm import (LLM, GeminiBackend, TokenLedger, LLMError,
                       QuotaExhausted, extract_code, DEFAULT_MODEL)
from agent.calibrate import calibrate, write as write_calibration
from agent.state import Node, PolicyConfig, SolutionJournal
from agent.task import ContractError, IntegrityError
from agent.verify_firewall import FirewallBreach

MAX_ITERATIONS = 50
MAX_HOURS = 6.0
CANDIDATE_TIMEOUT_S = 600      # 10 min; the reference model needs ~30 s
N_DRAFTS = 3                   # initial independent roots before improving
VERIFY_SEEDS = (1, 2)          # re-run seeds for a candidate worth confirming
DRAFT_EVERY = 6                # attempts between opening a new root


class AgentLoop:
    def __init__(self, run_dir=None, model=DEFAULT_MODEL,
                 max_iterations=MAX_ITERATIONS, max_hours=MAX_HOURS,
                 candidate_timeout_s=CANDIDATE_TIMEOUT_S,
                 n_drafts=N_DRAFTS, verify_seeds=VERIFY_SEEDS,
                 data_dir=None, skip_eda=False, task=None, calibrate_policy=True):
        # Defaulted rather than required: the documented CLI, harness_check.py
        # and tests/test_budget.py all construct a loop without naming a task,
        # and a required argument would break them for no benefit.
        if task is None:
            from tasks.kuairand.task import KuaiRandTask
            task = KuaiRandTask()
        self.task = task
        self.dir = Path(run_dir or new_run_dir(prefix="agent"))
        (self.dir / "nodes").mkdir(parents=True, exist_ok=True)
        (self.dir / "artifacts").mkdir(parents=True, exist_ok=True)

        self.journal = Journal(self.dir / "journal.jsonl")
        self.policy = PolicyConfig()
        self.state = SolutionJournal(self.dir / "state.json", policy=self.policy)
        self.ledger = TokenLedger(self.dir / "tokens.json")
        self.llm = LLM(backend=GeminiBackend(model=model),
                       ledger=self.ledger, journal=self.journal)

        self.max_iterations = max_iterations
        self.max_hours = max_hours
        self.candidate_timeout_s = candidate_timeout_s
        self.n_drafts = n_drafts
        self.verify_seeds = list(verify_seeds)
        self.data_dir = Path(data_dir or task.visible_data_dir())
        self.skip_eda = skip_eda
        self.calibrate_policy = calibrate_policy
        self.calibration = None

        self.eda = ""
        # Two clocks. The cap is charged in monotonic time because that is the
        # clock subprocess.communicate(timeout=...) uses in executor.py - so a
        # candidate's reported duration and its timeout can no longer disagree.
        # time.time() is kept alongside it so host suspend stays visible
        # instead of being silently absorbed. See docs/postmortem.md.
        self.t0 = time.monotonic()
        self.t0_wall = time.time()
        self.budget_path = self.dir / "budget.json"
        self.prior_awake_s = self._load_prior_awake()
        self.interventions = 0        # stays 0; a human touching this is one
        self._announced_abandoned = set()   # log each give-up once
        self._protected_seen = set()        # log each exploration once
        self._retired_seen = set()

    # ------------------------------------------------------------ utilities
    def log(self, event, **kw):
        return self.journal.append(event, **kw)

    def _load_prior_awake(self):
        """Awake seconds already spent in this run directory.

        self.t0 restarts on every resume, so without this the 6 h cap is
        charged per *segment*: run final_01 was resumed three times and each
        segment got a fresh six hours. The cap is meant to bound the run, so
        awake time accumulates across resumes.
        """
        try:
            return float(json.loads(self.budget_path.read_text())["awake_s"])
        except (OSError, ValueError, KeyError, TypeError):
            return 0.0

    def _save_budget(self):
        self.budget_path.write_text(json.dumps({
            "awake_s": round(self.awake_s(), 3),
            "wall_s": round(time.time() - self.t0_wall, 3),
            "segment_started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }) + "\n")

    def awake_s(self):
        """Seconds of running time, resumes included, host suspend excluded."""
        return self.prior_awake_s + (time.monotonic() - self.t0)

    def elapsed_h(self):
        """Awake hours. This is what the cap is charged against."""
        return self.awake_s() / 3600.0

    def wall_h(self):
        """Hours on the wall for this segment, host suspend included."""
        return (time.time() - self.t0_wall) / 3600.0

    def suspended_h(self):
        """Wall time this segment that the host spent asleep."""
        return max(0.0, self.wall_h() - (time.monotonic() - self.t0) / 3600.0)

    def say(self, msg):
        print(f"[{self.elapsed_h()*60:6.1f}m] {msg}", flush=True)

    # ------------------------------------------------------------ preflight
    def preflight(self):
        self.say("preflight ...")
        sha = self.task.integrity_check()
        counts = self.task.verify_isolation(verbose=False)
        if counts["test"] != 0:
            raise FirewallBreach("visible directory exposes test rows")
        self.log("preflight", status="ok", task=self.task.name,
                 evaluate_sha256=sha,
                 visible_counts=counts, model=self.llm.model,
                 data_dir=str(self.data_dir),
                 caps={"iterations": self.max_iterations,
                       "hours": self.max_hours,
                       "candidate_timeout_s": self.candidate_timeout_s},
                 convergence={"eps": self.policy.eps,
                              "N": self.policy.n_converge})
        self.say(f"  evaluate.py {sha[:12]}  visible train/valid/test="
                 f"{counts['train']:,d}/{counts['valid']:,d}/{counts['test']}")
        self.say(f"  model {self.llm.model}")
        self.run_calibration()

    def run_calibration(self):
        """Measure the task's noise floor and install thresholds from it.

        Without this the agent carries KuaiRand's numbers everywhere: a task
        whose sigma is 0.05 would verify on every node and never converge,
        because VERIFY_MARGIN = 0.0008 *is* KuaiRand's seed sigma. Skipping is
        always safe - the declared defaults stay in force and the run log says
        so.
        """
        if not self.calibrate_policy:
            return
        cal = calibrate(self.task, self.dir, max_hours=self.max_hours,
                        timeout_s=self.candidate_timeout_s, log=self.say)
        self.calibration = cal
        write_calibration(cal, self.dir)
        record = cal.to_dict()
        # `status` is the journal's own field name; the calibration's own
        # status travels as calibration_status rather than colliding with it.
        record["calibration_status"] = record.pop("status")
        self.log("calibration",
                 status=("ok" if cal.status == "ok" else "info"), **record)
        if cal.status != "ok":
            self.say(f"  {cal.summary()}; keeping declared thresholds")
            return
        self.policy = PolicyConfig.from_calibration(cal, base=self.policy)
        self.state.policy = self.policy
        if not self.policy.verification_useful:
            self.verify_seeds = []
            self.say("  task is deterministic; seed verification disabled")
        self.log("policy", status="info", **self.policy.to_dict())
        self.say(f"  {cal.summary()}")

    # ----------------------------------------------------------------- EDA
    def run_eda(self):
        if self.skip_eda:
            return
        self.say("exploratory analysis (agent-written) ...")
        self.log("explore_start", status="info")
        try:
            resp = self.llm.complete(self.task.system_prompt(),
                                     self.task.explore_prompt(),
                                     temperature=0.6)
        except LLMError as e:
            self.log("explore_failed", status="error", error=str(e)[:500])
            self.say("  EDA request failed; continuing without it")
            return

        code = extract_code(resp.text)
        if not code:
            self.log("explore_failed", status="error",
                     error="no code block in response")
            return

        path = self.dir / "nodes" / "eda.py"
        path.write_text(code)
        r = run_script(path, ["--data_dir", self.data_dir],
                       timeout_s=self.candidate_timeout_s,
                       extra_path=self.task.extra_sys_path())
        if r.ok:
            self.eda = r.stdout
            (self.dir / "eda_report.txt").write_text(r.stdout)
            self.log("explore_done", status="ok", wall_s=round(r.wall_s, 1),
                     hypothesis=self.task.extract_hypothesis(resp.text),
                     detail=r.stdout[-2000:])
            self.say(f"  EDA ok in {r.wall_s:.0f}s, "
                     f"{len(r.stdout.splitlines())} lines of findings")
        else:
            # Not worth a repair cycle - the loop works fine without EDA.
            self.log("explore_failed", status="error", error=r.summary(),
                     detail=r.stderr[-1200:])
            self.say(f"  EDA failed ({r.summary()}); continuing without it")

    # ------------------------------------------------------------- policy
    def draft_is_due(self):
        """Should this iteration open a new root?

        Three things were wrong with the old `iteration % 7 == 6` test. It sat
        below the debug branch, so a due draft was silently swallowed by a
        repair chain - final_01's iterations 13, 20 and 27 all came due and
        all three were eaten, leaving one alternative root opened in thirty
        iterations. It was keyed to the loop counter, which restarts at 1 on
        every resume, so the cadence restarted three times in that same run.
        And it could open an exploration with no budget left to develop it.
        """
        if self.state.protected_root() is not None:
            return False              # one exploration at a time
        if self.state.nodes_since_last_root() < DRAFT_EVERY:
            return False
        remaining = self.max_iterations - len(self.state.nodes)
        return remaining > self.policy.protection_max_nodes

    def announce_abandoned(self):
        for dead in self.state.abandoned():
            if dead.id in self._announced_abandoned:
                continue
            self._announced_abandoned.add(dead.id)
            self.log("node_abandoned", node_id=dead.id, status="info",
                     attempts=self.state.repair_attempts(dead.id),
                     detail=(dead.failure_reason or "")[:300])
            self.say(f"  giving up on #{dead.id} after "
                     f"{self.state.repair_attempts(dead.id)} repairs")

    def note_protection(self):
        """Log a lineage entering or leaving its development budget.

        This is the payoff ledger: at the end of a run it says how many
        explorations were bought and what each one returned. A policy that
        cannot be read off the run log is not auditable, and the run log is a
        graded deliverable.
        """
        root_id, reason = self.state.protection_status()
        if root_id is None:
            return
        best = self.state.best()
        lin = self.state.lineage(root_id)
        scored = [n for n in lin if not n.is_buggy and n.score is not None]
        lin_best = max((n.score for n in scored), default=None)
        if reason == "active":
            if root_id not in self._protected_seen:
                self._protected_seen.add(root_id)
                self.log("lineage_protected", root_id=root_id, status="info",
                         budget_scored=self.policy.protected_scored_attempts,
                         budget_nodes=self.policy.protection_max_nodes,
                         incumbent=best.score if best else None,
                         lineage_best=lin_best)
                self.say(f"  exploring from root #{root_id}: protected for "
                         f"{self.policy.protected_scored_attempts} scored "
                         f"attempts")
        elif root_id in self._protected_seen and root_id not in self._retired_seen:
            self._retired_seen.add(root_id)
            delta = (lin_best - best.score) if (lin_best is not None and best) else None
            self.log("lineage_retired", root_id=root_id, status="info",
                     reason=reason, best_in_lineage=lin_best,
                     delta_vs_incumbent=None if delta is None else round(delta, 5),
                     nodes_spent=len(lin))
            self.say(f"  root #{root_id} retired ({reason})"
                     + (f", best {lin_best:.4f}" if lin_best is not None else ""))

    def choose_action(self, iteration):
        """Return (stage, parent_node_or_None)."""
        self.announce_abandoned()
        self.note_protection()

        # Open the first N_DRAFTS roots outright. The old test was
        # `len(self.state.good()) == 0`, which ended the phase the moment the
        # first draft scored - so N_DRAFTS never took effect and both recorded
        # runs opened exactly one initial root.
        if len(self.state.roots()) < self.n_drafts:
            return "draft", None

        parent = self.state.select_parent()
        if parent is None:
            return "draft", None
        # Checked before the debug branch, so a repair chain cannot eat it.
        if self.draft_is_due():
            return "draft", None
        if parent.is_buggy:
            return "debug", parent
        return "improve", parent

    def build_prompt(self, stage, parent):
        summary = self.state.summary()
        if stage == "draft":
            return self.task.draft_prompt(
                summary, self.eda, n_existing=len(self.state.nodes),
                lineages=self.state.lineage_rollup())
        if stage == "debug":
            # Tell the model how much repair budget is left, so a last attempt
            # reaches for the smallest fix rather than the most ambitious one.
            return self.task.debug_prompt(
                parent, summary,
                attempt=self.state.repair_attempts(parent.id) + 1,
                max_attempts=self.policy.max_debug_attempts)
        return self.task.improve_prompt(parent, summary, self.eda)

    # -------------------------------------------------------- one iteration
    def scope_for(self, stage, parent):
        """Label this attempt exploit or explore, at insert time.

        Recorded rather than derived: the retirement test depends on an
        incumbent that moves, so a scope recomputed later would not match what
        the search actually did. The convergence floor counts exploit nodes,
        so this label has to be stable.
        """
        if stage == "draft" and len(self.state.roots()) >= self.n_drafts:
            return "explore", None            # a deliberate exploration root
        root_id = self.state.protected_root()
        if (root_id is not None and parent is not None
                and self.state.root_of(parent.id) == root_id):
            return "explore", root_id
        return "exploit", None

    def iterate(self, iteration):
        stage, parent = self.choose_action(iteration)
        node_id = self.state.next_id()
        scope, protected_root = self.scope_for(stage, parent)
        self.say(f"iter {iteration:2d} | {stage}"
                 + (f" from #{parent.id}" if parent else "")
                 + f" -> node #{node_id}")
        self.log("iteration_start", iteration=iteration, node_id=node_id,
                 stage=stage, parent_id=parent.id if parent else None,
                 scope=scope, protected_root=protected_root,
                 best_so_far=self.state.best_score(),
                 best_verified=self.state.best_verified_score(), status="info")

        # --- ask the model -------------------------------------------------
        try:
            resp = self.llm.complete(self.task.system_prompt(),
                                     self.build_prompt(stage, parent),
                                     temperature=0.8 if stage == "draft" else 0.5)
        except QuotaExhausted:
            raise
        except LLMError as e:
            self.log("llm_failed", node_id=node_id, status="error",
                     error=str(e)[:600])
            self.say(f"  LLM call failed: {str(e)[:120]}")
            return None

        hypothesis = self.task.extract_hypothesis(resp.text)
        code = extract_code(resp.text)
        node = Node(id=node_id, stage=stage, hypothesis=hypothesis,
                    code=code or "", parent_id=parent.id if parent else None,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    scope=scope,
                    protected_root=(node_id if scope == "explore"
                                    and parent is None else protected_root))
        self.say(f"  hypothesis: {hypothesis[:100]}")

        if not code:
            node.failure_reason = ("The response contained no fenced python "
                                   "block. Return the complete script.")
            return self.finish_node(node, resp)

        if resp.finish_reason == "MAX_TOKENS":
            # The script is cut off mid-line; running it only yields a
            # SyntaxError that says nothing useful. Name the real cause so the
            # debug step completes the script rather than re-deriving it.
            #
            # Asking for a *shorter* script here was a mistake in final_01:
            # the model complied by dropping features, and the three repairs
            # that did run came back at 0.4791 / 0.5961 / 0.4832 against an
            # incumbent of 0.6042. The repair pressure destroyed the solution.
            # Ask it to continue what it already wrote instead.
            node.failure_reason = (
                "Your response stopped at the output token limit, so the "
                "script below is incomplete - it ends mid-statement. The "
                "approach is fine; it was not finished.\n\n"
                "Reproduce the script above verbatim up to where it stops, "
                "then continue from that point to a complete, runnable file. "
                "Do not redesign it, do not drop features to save room, and "
                "do not start over.")
            self.log("response_truncated", node_id=node_id, status="error",
                     code_chars=len(code))
            self.say("  response truncated at output limit; queued for repair")
            return self.finish_node(node, resp)

        # Save the source before validating it: rejected code still belongs
        # in the run log, and it is what the next iteration has to repair.
        path = self.dir / "nodes" / f"node_{node_id:03d}.py"
        path.write_text(code)

        # --- syntax: compile before spending a subprocess on it ------------
        try:
            assert_parses(code)
        except SyntaxRejection as e:
            node.failure_reason = str(e)
            self.log("syntax_rejected", node_id=node_id, status="error",
                     error=str(e).splitlines()[0][:300])
            self.say(f"  invalid Python; not executed "
                     f"({str(e).splitlines()[0][:80]})")
            return self.finish_node(node, resp)

        # --- guard: static check before anything executes ------------------
        try:
            warnings = assert_clean(code, patterns=self.task.guard_patterns())
            if warnings:
                self.log("guard_warning", node_id=node_id, status="info",
                         detail="; ".join(f"line {ln}: {why}"
                                          for why, ln, _ in warnings))
        except GuardRejection as e:
            node.failure_reason = str(e)
            self.log("guard_rejected", node_id=node_id, status="error",
                     error=str(e)[:600])
            self.say("  REJECTED by guard (tried to reach sealed data)")
            return self.finish_node(node, resp)

        # --- run it --------------------------------------------------------
        out = self.dir / "artifacts" / f"scores_valid_{node_id:03d}.csv"
        r = run_script(path,
                       self.task.candidate_argv(self.data_dir, "valid", out, 0),
                       timeout_s=self.candidate_timeout_s,
                       extra_path=self.task.extra_sys_path())
        node.exec_ok = r.ok
        node.exec_summary = r.summary()
        node.stdout_tail = r.stdout
        node.stderr_tail = r.stderr
        node.wall_s = r.wall_s
        node.wall_clock_s = r.wall_clock_s

        if not r.ok:
            node.failure_reason = (
                f"The script timed out after {self.candidate_timeout_s}s."
                if r.timed_out else
                f"The script exited with code {r.returncode}. "
                f"{(r.exc_type or '')}: {(r.exc_msg or '')}")
            self.say(f"  {r.summary()}")
            return self.finish_node(node, resp)

        # --- score it ------------------------------------------------------
        try:
            s = self.task.validate_and_score(out, split="valid",
                                             data_dir=self.data_dir)
        except (ContractError, IntegrityError) as e:
            node.failure_reason = f"The output failed validation. {e}"
            self.say(f"  contract violation: {str(e).splitlines()[0][:100]}")
            return self.finish_node(node, resp)

        # Oriented once, here at the boundary. Everything in agent/state.py
        # compares in the maximise direction, so a minimise task is handled by
        # storing -score rather than by threading a sign through six separate
        # comparisons - one of which would eventually be missed.
        node.sign = self.task.sign
        node.seed_scores = {0: self.task.sign * s.primary}
        node.metrics = s.to_dict()
        node.is_buggy = False
        detail = ", ".join(f"{k} {v:.4f}" for k, v in s.metrics.items())
        self.say(f"  seed 0: primary {s.primary:.4f}"
                 + (f" ({detail})" if detail else "")
                 + f" in {r.wall_s:.0f}s")

        # --- multi-seed verification before promotion ----------------------
        # Anchored on the best *verified* score, not on the running best. Six
        # consecutive sub-EPS gains in final_02 compounded past the last
        # confirmed node without any of them tripping the old check.
        if (self.verify_seeds
                and self.state.needs_verification(self.task.sign * s.primary)):
            self.verify_across_seeds(node, path)

        return self.finish_node(node, resp)

    def verify_across_seeds(self, node, path):
        """A single-seed gain is not evidence. Re-run before promoting."""
        self.say(f"  candidate beats incumbent; verifying on seeds "
                 f"{self.verify_seeds}")
        self.log("seed_verification_start", node_id=node.id,
                 seed0=node.seed_scores.get(0), seeds=self.verify_seeds,
                 status="info")
        for seed in self.verify_seeds:
            out = self.dir / "artifacts" / f"scores_valid_{node.id:03d}_s{seed}.csv"
            r = run_script(path,
                           self.task.candidate_argv(self.data_dir, "valid",
                                                    out, seed),
                           timeout_s=self.candidate_timeout_s,
                           extra_path=self.task.extra_sys_path())
            if not r.ok:
                self.log("seed_verification_failed", node_id=node.id, seed=seed,
                         status="error", error=r.summary())
                self.say(f"    seed {seed}: FAILED ({r.summary()})")
                continue
            try:
                s = self.task.validate_and_score(out, split="valid",
                                             data_dir=self.data_dir)
            except (ContractError, IntegrityError) as e:
                self.log("seed_verification_failed", node_id=node.id, seed=seed,
                         status="error", error=str(e)[:300])
                continue
            node.seed_scores[seed] = self.task.sign * s.primary
            self.say(f"    seed {seed}: {s.primary:.4f}")

        std = node.seed_std
        self.log("seed_verification_done", node_id=node.id,
                 seed_scores=node.seed_scores, mean=node.score,
                 std=std, status="ok")
        self.say(f"  verified mean {node.native_score:.4f}"
                 + (f" +/- {std:.4f}" if std else "")
                 + f" over {node.n_seeds} seeds")

    def verify_shipped_node(self):
        """Confirm the node about to be shipped across seeds, if it never was.

        final_02 converged on a node that had only ever been run on seed 0;
        checking it was a manual step after the run. One extra two-seed run at
        the end makes that the agent's own behaviour. best() is left alone -
        the rule is to ship the validation-best checkpoint.
        """
        best = self.state.best()
        if best is None or best.n_seeds > 1:
            return
        path = self.dir / "nodes" / f"node_{best.id:03d}.py"
        if not path.exists():
            return
        self.say(f"  shipped node #{best.id} was never seed-verified; "
                 f"confirming before reporting")
        before = best.score
        self.verify_across_seeds(best, path)
        self.state.save()
        self.log("final_verification", node_id=best.id,
                 seed0=before, seed_scores=best.seed_scores,
                 mean=best.score, std=best.seed_std, status="ok")

    def finish_node(self, node, resp=None):
        prev_best = self.state.best_score()
        self.state.add(node)
        diff = self.code_diff(node)
        (self.dir / "nodes" / f"node_{node.id:03d}.diff").write_text(diff or "")
        self.log("node_added", node_id=node.id, stage=node.stage,
                 parent_id=node.parent_id,
                 status="ok" if not node.is_buggy else "error",
                 hypothesis=node.hypothesis,
                 score=node.metrics if node.metrics else None,
                 seed_scores=node.seed_scores or None,
                 mean_primary=node.score,
                 wall_s=round(node.wall_s, 1),
                 error=node.failure_reason or None,
                 code_diff_lines=len((diff or "").splitlines()))
        new_best = self.state.best_score()
        if new_best is not None and (prev_best is None or new_best > prev_best):
            sign = getattr(self.task, "sign", 1.0)
            self.say(f"  NEW BEST {sign * new_best:.4f}"
                     + (f" (was {sign * prev_best:.4f})" if prev_best else ""))
            self.log("new_best", node_id=node.id, primary=new_best,
                     previous=prev_best, status="ok")
        return node

    def code_diff(self, node):
        """Unified diff against the parent - the run log records what changed."""
        parent = self.state.get(node.parent_id) if node.parent_id is not None else None
        before = (parent.code if parent else "").splitlines(keepends=True)
        after = node.code.splitlines(keepends=True)
        return "".join(difflib.unified_diff(
            before, after,
            fromfile=f"node_{node.parent_id:03d}.py" if parent else "/dev/null",
            tofile=f"node_{node.id:03d}.py", n=2))

    # ------------------------------------------------------------- the run
    def run(self):
        self.say(f"run dir: {self.dir}")
        self.log("run_start", status="info",
                 note="Autonomous run. Any human action from here is a manual "
                      "intervention and must be recorded in docs/interventions.md.")
        self.preflight()
        self.run_eda()

        stop = "iteration cap"
        for iteration in range(1, self.max_iterations + 1):
            if self.elapsed_h() >= self.max_hours:
                stop = "wall-clock cap"
                break
            try:
                self.iterate(iteration)
                self._save_budget()
            except QuotaExhausted as e:
                stop = "API quota exhausted"
                self.log("quota_exhausted", status="error", error=str(e)[:500])
                self.say(f"  {stop}: state saved, run can resume later")
                break
            except KeyboardInterrupt:
                stop = "interrupted"
                self.log("interrupted", status="error")
                break
            except Exception as e:
                # An unexpected failure must not end the run: record and carry on.
                self.log("iteration_error", iteration=iteration, status="error",
                         error=f"{type(e).__name__}: {e}")
                self.say(f"  unexpected error, continuing: "
                         f"{type(e).__name__}: {str(e)[:120]}")
                continue

            if self.state.has_converged():
                stop = (f"converged (eps={self.policy.eps:g}, "
                        f"N={self.policy.n_converge})")
                self.log("converged", status="ok", iteration=iteration,
                         best=self.state.best_score())
                break

        return self.finalise(stop)

    def finalise(self, stop_reason):
        self.verify_shipped_node()
        best = self.state.best()
        self.log("run_end", status="ok", stop_reason=stop_reason,
                 iterations=len(self.state.nodes),
                 best_node=best.id if best else None,
                 best_primary=best.score if best else None,
                 elapsed_h=round(self.elapsed_h(), 2),
                 wall_h=round(self.wall_h(), 2),
                 suspended_h=round(self.suspended_h(), 2),
                 tokens=self.ledger.total.to_dict(),
                 roots_opened=len(self.state.roots()),
                 windows_protected=len(self._protected_seen),
                 windows_that_won=sum(
                     1 for r in self._protected_seen
                     if best is not None and self.state.root_of(best.id) == r),
                 scored_exploit=len([n for n in self.state.good()
                                     if n.scope == "exploit"]),
                 scored_explore=len([n for n in self.state.good()
                                     if n.scope == "explore"]),
                 interventions=self.interventions)
        (self.dir / "run_log.md").write_text(
            render_markdown(self.journal.read(),
                            title=f"Autonomous run - {self.dir.name}"))

        print("\n" + "=" * 68)
        print(f"stopped: {stop_reason}")
        print(f"attempts: {len(self.state.nodes)}  "
              f"({len(self.state.good())} scored, {len(self.state.buggy())} failed)")
        if best:
            print(f"best: node #{best.id} ({best.stage}) "
                  f"primary {best.native_score:.4f} over {best.n_seeds} seed(s)")
            ref = getattr(self.calibration, "seed_scores", None)
            if ref:
                base = sum(ref) / len(ref)
                delta = (best.native_score - base) * self.task.sign
                print(f"      reference {base:.4f} -> delta {delta:+.4f} "
                      f"({'better' if delta > 0 else 'worse'})")
            print(f"      {self.dir/'nodes'/f'node_{best.id:03d}.py'}")
            print(f"      hypothesis: {best.hypothesis[:150]}")
        else:
            print("best: none - no candidate scored")
        susp = self.suspended_h()
        print(f"elapsed: {self.elapsed_h():.2f} h awake"
              + (f"  ({self.wall_h():.2f} h wall, {susp:.2f} h suspended)"
                 if susp >= 0.01 else "")
              + f"    tokens: {self.ledger.summary()}")
        print(f"run log: {self.dir/'run_log.md'}")
        return best


def main():
    ap = argparse.ArgumentParser(description="Run the autonomous research loop.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max_iterations", type=int, default=MAX_ITERATIONS)
    ap.add_argument("--max_hours", type=float, default=MAX_HOURS)
    ap.add_argument("--candidate_timeout_s", type=int, default=CANDIDATE_TIMEOUT_S)
    ap.add_argument("--n_drafts", type=int, default=N_DRAFTS)
    ap.add_argument("--skip_eda", action="store_true")
    ap.add_argument("--run_dir", default=None)
    a = ap.parse_args()

    AgentLoop(run_dir=a.run_dir, model=a.model,
              max_iterations=a.max_iterations, max_hours=a.max_hours,
              candidate_timeout_s=a.candidate_timeout_s,
              n_drafts=a.n_drafts, skip_eda=a.skip_eda).run()


if __name__ == "__main__":
    main()
