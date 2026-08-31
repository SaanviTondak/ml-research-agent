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
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import prompts, scorer
from agent.executor import run_script
from agent.guard import assert_clean, GuardRejection
from agent.journal import Journal, new_run_dir, render_markdown
from agent.llm import (LLM, GeminiBackend, TokenLedger, LLMError,
                       QuotaExhausted, extract_code, DEFAULT_MODEL)
from agent.paths import VISIBLE_DATA
from agent.state import Node, SolutionJournal, EPS, N_CONVERGE
from agent.verify_firewall import verify, FirewallBreach

MAX_ITERATIONS = 50
MAX_HOURS = 6.0
CANDIDATE_TIMEOUT_S = 600      # 10 min; the reference model needs ~30 s
N_DRAFTS = 3                   # initial independent attempts before improving
VERIFY_SEEDS = (1, 2)          # re-run seeds for a candidate that beats best


class AgentLoop:
    def __init__(self, run_dir=None, model=DEFAULT_MODEL,
                 max_iterations=MAX_ITERATIONS, max_hours=MAX_HOURS,
                 candidate_timeout_s=CANDIDATE_TIMEOUT_S,
                 n_drafts=N_DRAFTS, verify_seeds=VERIFY_SEEDS,
                 data_dir=None, skip_eda=False):
        self.dir = Path(run_dir or new_run_dir(prefix="agent"))
        (self.dir / "nodes").mkdir(parents=True, exist_ok=True)
        (self.dir / "artifacts").mkdir(parents=True, exist_ok=True)

        self.journal = Journal(self.dir / "journal.jsonl")
        self.state = SolutionJournal(self.dir / "state.json")
        self.ledger = TokenLedger(self.dir / "tokens.json")
        self.llm = LLM(backend=GeminiBackend(model=model),
                       ledger=self.ledger, journal=self.journal)

        self.max_iterations = max_iterations
        self.max_hours = max_hours
        self.candidate_timeout_s = candidate_timeout_s
        self.n_drafts = n_drafts
        self.verify_seeds = list(verify_seeds)
        self.data_dir = Path(data_dir or VISIBLE_DATA)
        self.skip_eda = skip_eda

        self.eda = ""
        self.t0 = time.time()
        self.interventions = 0        # stays 0; a human touching this is one

    # ------------------------------------------------------------ utilities
    def log(self, event, **kw):
        return self.journal.append(event, **kw)

    def elapsed_h(self):
        return (time.time() - self.t0) / 3600.0

    def say(self, msg):
        print(f"[{self.elapsed_h()*60:6.1f}m] {msg}", flush=True)

    # ------------------------------------------------------------ preflight
    def preflight(self):
        self.say("preflight ...")
        sha = scorer.assert_evaluate_untouched()
        counts = verify(verbose=False)
        if counts["test"] != 0:
            raise FirewallBreach("visible directory exposes test rows")
        self.log("preflight", status="ok", evaluate_sha256=sha,
                 visible_counts=counts, model=self.llm.model,
                 data_dir=str(self.data_dir),
                 caps={"iterations": self.max_iterations,
                       "hours": self.max_hours,
                       "candidate_timeout_s": self.candidate_timeout_s},
                 convergence={"eps": EPS, "N": N_CONVERGE})
        self.say(f"  evaluate.py {sha[:12]}  visible train/valid/test="
                 f"{counts['train']:,d}/{counts['valid']:,d}/{counts['test']}")
        self.say(f"  model {self.llm.model}")

    # ----------------------------------------------------------------- EDA
    def run_eda(self):
        if self.skip_eda:
            return
        self.say("exploratory analysis (agent-written) ...")
        self.log("explore_start", status="info")
        try:
            resp = self.llm.complete(prompts.SYSTEM, prompts.explore_prompt(),
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
                       timeout_s=self.candidate_timeout_s)
        if r.ok:
            self.eda = r.stdout
            (self.dir / "eda_report.txt").write_text(r.stdout)
            self.log("explore_done", status="ok", wall_s=round(r.wall_s, 1),
                     hypothesis=prompts.extract_hypothesis(resp.text),
                     detail=r.stdout[-2000:])
            self.say(f"  EDA ok in {r.wall_s:.0f}s, "
                     f"{len(r.stdout.splitlines())} lines of findings")
        else:
            # Not worth a repair cycle - the loop works fine without EDA.
            self.log("explore_failed", status="error", error=r.summary(),
                     detail=r.stderr[-1200:])
            self.say(f"  EDA failed ({r.summary()}); continuing without it")

    # ------------------------------------------------------------- policy
    def choose_action(self, iteration):
        """Return (stage, parent_node_or_None)."""
        if len(self.state.good()) == 0 and len(self.state.nodes) < self.n_drafts:
            return "draft", None
        parent = self.state.select_parent()
        if parent is None:
            return "draft", None
        if parent.is_buggy:
            return "debug", parent
        # Periodically branch out so the search does not collapse onto one line.
        if iteration % 7 == 6:
            return "draft", None
        return "improve", parent

    def build_prompt(self, stage, parent):
        summary = self.state.summary()
        if stage == "draft":
            return prompts.draft_prompt(summary, self.eda,
                                        n_existing=len(self.state.nodes))
        if stage == "debug":
            return prompts.debug_prompt(parent, summary)
        return prompts.improve_prompt(parent, summary, self.eda)

    # -------------------------------------------------------- one iteration
    def iterate(self, iteration):
        stage, parent = self.choose_action(iteration)
        node_id = self.state.next_id()
        self.say(f"iter {iteration:2d} | {stage}"
                 + (f" from #{parent.id}" if parent else "")
                 + f" -> node #{node_id}")
        self.log("iteration_start", iteration=iteration, node_id=node_id,
                 stage=stage, parent_id=parent.id if parent else None,
                 best_so_far=self.state.best_score(), status="info")

        # --- ask the model -------------------------------------------------
        try:
            resp = self.llm.complete(prompts.SYSTEM,
                                     self.build_prompt(stage, parent),
                                     temperature=0.8 if stage == "draft" else 0.5)
        except QuotaExhausted:
            raise
        except LLMError as e:
            self.log("llm_failed", node_id=node_id, status="error",
                     error=str(e)[:600])
            self.say(f"  LLM call failed: {str(e)[:120]}")
            return None

        hypothesis = prompts.extract_hypothesis(resp.text)
        code = extract_code(resp.text)
        node = Node(id=node_id, stage=stage, hypothesis=hypothesis,
                    code=code or "", parent_id=parent.id if parent else None,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.say(f"  hypothesis: {hypothesis[:100]}")

        if not code:
            node.failure_reason = ("The response contained no fenced python "
                                   "block. Return the complete script.")
            return self.finish_node(node, resp)

        if resp.finish_reason == "MAX_TOKENS":
            # The script is cut off mid-line; running it only yields a
            # SyntaxError that says nothing useful. Name the real cause so the
            # debug step completes the script rather than re-deriving it.
            node.failure_reason = (
                "Your response was cut off at the output token limit, so the "
                "script is incomplete. Complete it, and keep it shorter - "
                "under ~250 lines. Prefer a simpler model you can finish "
                "writing over an elaborate one you cannot.")
            self.log("response_truncated", node_id=node_id, status="error",
                     code_chars=len(code))
            self.say("  response truncated at output limit; queued for repair")
            return self.finish_node(node, resp)

        # --- guard: static check before anything executes ------------------
        try:
            assert_clean(code)
        except GuardRejection as e:
            node.failure_reason = str(e)
            self.log("guard_rejected", node_id=node_id, status="error",
                     error=str(e)[:600])
            self.say("  REJECTED by guard (tried to reach sealed data)")
            return self.finish_node(node, resp)

        # --- run it --------------------------------------------------------
        path = self.dir / "nodes" / f"node_{node_id:03d}.py"
        path.write_text(code)
        out = self.dir / "artifacts" / f"scores_valid_{node_id:03d}.csv"
        r = run_script(path, ["--data_dir", self.data_dir, "--split", "valid",
                              "--out", out, "--seed", 0],
                       timeout_s=self.candidate_timeout_s)
        node.exec_ok = r.ok
        node.exec_summary = r.summary()
        node.stdout_tail = r.stdout
        node.stderr_tail = r.stderr
        node.wall_s = r.wall_s

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
            s = scorer.score_file(out, split="valid", data_dir=self.data_dir)
        except (scorer.ContractError, scorer.IntegrityError) as e:
            node.failure_reason = f"The output failed validation. {e}"
            self.say(f"  contract violation: {str(e).splitlines()[0][:100]}")
            return self.finish_node(node, resp)

        node.seed_scores = {0: s.primary}
        node.metrics = s.to_dict()
        node.is_buggy = False
        self.say(f"  seed 0: primary {s.primary:.4f} "
                 f"(GAUC {s.gauc:.4f}, nDCG@5 {s.ndcg5:.4f}) in {r.wall_s:.0f}s")

        # --- multi-seed verification before promotion ----------------------
        best = self.state.best_score()
        if best is None or s.primary > best + EPS:
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
            r = run_script(path, ["--data_dir", self.data_dir, "--split", "valid",
                                  "--out", out, "--seed", seed],
                           timeout_s=self.candidate_timeout_s)
            if not r.ok:
                self.log("seed_verification_failed", node_id=node.id, seed=seed,
                         status="error", error=r.summary())
                self.say(f"    seed {seed}: FAILED ({r.summary()})")
                continue
            try:
                s = scorer.score_file(out, split="valid", data_dir=self.data_dir)
            except (scorer.ContractError, scorer.IntegrityError) as e:
                self.log("seed_verification_failed", node_id=node.id, seed=seed,
                         status="error", error=str(e)[:300])
                continue
            node.seed_scores[seed] = s.primary
            self.say(f"    seed {seed}: {s.primary:.4f}")

        std = node.seed_std
        self.log("seed_verification_done", node_id=node.id,
                 seed_scores=node.seed_scores, mean=node.score,
                 std=std, status="ok")
        self.say(f"  verified mean {node.score:.4f}"
                 + (f" +/- {std:.4f}" if std else "")
                 + f" over {node.n_seeds} seeds")

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
            self.say(f"  NEW BEST {new_best:.4f}"
                     + (f" (was {prev_best:.4f})" if prev_best else ""))
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
                stop = f"converged (eps={EPS}, N={N_CONVERGE})"
                self.log("converged", status="ok", iteration=iteration,
                         best=self.state.best_score())
                break

        return self.finalise(stop)

    def finalise(self, stop_reason):
        best = self.state.best()
        self.log("run_end", status="ok", stop_reason=stop_reason,
                 iterations=len(self.state.nodes),
                 best_node=best.id if best else None,
                 best_primary=best.score if best else None,
                 elapsed_h=round(self.elapsed_h(), 2),
                 tokens=self.ledger.total.to_dict(),
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
                  f"primary {best.score:.4f} over {best.n_seeds} seed(s)")
            print(f"      baseline 0.6016 valid -> delta {best.score - 0.6016:+.4f}")
            print(f"      {self.dir/'nodes'/f'node_{best.id:03d}.py'}")
            print(f"      hypothesis: {best.hypothesis[:150]}")
        else:
            print("best: none - no candidate scored")
        print(f"elapsed: {self.elapsed_h():.2f} h    tokens: {self.ledger.summary()}")
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
