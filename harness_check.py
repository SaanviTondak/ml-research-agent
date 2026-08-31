"""Phase 1 milestone: prove the harness before any LLM touches it.

Runs the organizer's own FM through the full agent pipeline - executor,
candidate contract, scorer, journal - against the firewalled data directory,
and asserts it comes back out at the published baseline score.

If this passes, the foundation is trustworthy: a score produced by the loop
means what it says, and it was produced without any access to test.

    python3 harness_check.py

Checks, in order:
  1. evaluate.py is byte-identical to the organizer's
  2. the firewall is built and provably clean
  3. candidate 0 runs to completion under the executor
  4. its output satisfies the submission contract
  5. it scores the published baseline (0.6015 valid, seed 0)
  6. the executor recovers from a crashing candidate
  7. the executor recovers from a hanging candidate
  8. the scorer rejects a misaligned submission
  9. the scorer refuses to score test
"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import scorer
from agent.executor import run_script
from agent.journal import Journal, new_run_dir, render_markdown
from agent.paths import CANDIDATES, VISIBLE_DATA, REAL_DATA
from agent.verify_firewall import verify, FirewallBreach

EXPECTED_VALID_PRIMARY = 0.6015     # docs/phase0_baseline_repro.md, seed 0
TOLERANCE = 0.002                   # the organizers' own epsilon

passed, failed = [], []


def check(name):
    def deco(fn):
        print(f"\n[{name}]")
        try:
            fn()
        except Exception as e:
            failed.append((name, e))
            print(f"  FAILED: {type(e).__name__}: {e}")
        else:
            passed.append(name)
            print(f"  PASS")
        return fn
    return deco


run_dir = new_run_dir(prefix="harness_check")
jr = Journal(run_dir / "journal.jsonl")
jr.append("harness_check_start", note="Phase 1 milestone", status="info")
print(f"run dir: {run_dir}")

# ---------------------------------------------------------------- 1, 2
@check("1. evaluate.py unmodified")
def _():
    sha = scorer.assert_evaluate_untouched()
    print(f"  sha256 {sha[:16]}...")
    jr.append("integrity_check", status="ok", detail=f"evaluate.py sha256={sha}")


@check("2. firewall built and clean")
def _():
    counts = verify(verbose=False)
    assert counts["test"] == 0, f"test split not empty: {counts}"
    print(f"  visible: train={counts['train']:,d} valid={counts['valid']:,d} "
          f"test={counts['test']}")
    jr.append("firewall_verified", status="ok", counts=counts)


# ---------------------------------------------------------------- 3, 4, 5
out_csv = run_dir / "artifacts" / "scores_valid.csv"
result = None


@check("3. candidate 0 runs under the executor")
def _():
    global result
    jr.append("candidate_start", candidate="fm_baseline",
              hypothesis="Reproduce the official FM baseline through the "
                         "harness to prove the pipeline is sound.",
              status="info")
    result = run_script(
        CANDIDATES / "fm_baseline.py",
        ["--data_dir", VISIBLE_DATA, "--split", "valid", "--out", out_csv],
        timeout_s=900)
    print(f"  {result.summary()}")
    jr.append("candidate_exec", candidate="fm_baseline",
              status="ok" if result.ok else "error",
              wall_s=round(result.wall_s, 1), stdout_tail=result.stdout[-400:])
    assert result.ok, f"candidate failed: {result.summary()}\n{result.stderr}"


@check("4. output satisfies the submission contract")
def _():
    rows = scorer.load_eval_rows(VISIBLE_DATA, "valid")
    scores = scorer.read_scores(out_csv, rows)
    assert len(scores) == len(rows) == 124_909, f"got {len(scores)} rows"
    print(f"  {len(scores):,d} rows aligned to data.load()['valid']")


@check("5. reproduces the published baseline")
def _():
    s = scorer.score_file(out_csv, split="valid", data_dir=VISIBLE_DATA)
    delta = s.primary - EXPECTED_VALID_PRIMARY
    print(f"  {s}")
    print(f"  expected primary {EXPECTED_VALID_PRIMARY:.4f}, "
          f"delta {delta:+.4f} (tolerance +/-{TOLERANCE})")
    jr.append("candidate_scored", candidate="fm_baseline", status="ok",
              score=s.to_dict(), note="baseline reproduced through the harness")
    assert abs(delta) <= TOLERANCE, f"off baseline by {delta:+.4f}"


# ---------------------------------------------------------------- 6, 7
@check("6. executor survives a crashing candidate")
def _():
    bad = run_dir / "artifacts" / "_crash.py"
    bad.write_text("raise ValueError('deliberate failure')\n")
    r = run_script(bad, timeout_s=60)
    print(f"  {r.summary()}")
    jr.append("recovery_probe", probe="crash", status="error",
              error=f"{r.exc_type}: {r.exc_msg}",
              note="deliberate; loop must continue")
    assert not r.ok and r.exc_type == "ValueError", r.summary()


@check("7. executor kills a hanging candidate")
def _():
    hang = run_dir / "artifacts" / "_hang.py"
    hang.write_text(textwrap.dedent("""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])
        while True:
            time.sleep(1)
    """))
    r = run_script(hang, timeout_s=5)
    print(f"  {r.summary()}")
    jr.append("recovery_probe", probe="timeout", status="timeout",
              wall_s=round(r.wall_s, 1), note="deliberate; loop must continue")
    assert r.timed_out and r.wall_s < 45, f"not killed cleanly: {r.summary()}"


# ---------------------------------------------------------------- 8, 9
@check("8. scorer rejects a misaligned submission")
def _():
    rows = scorer.load_eval_rows(VISIBLE_DATA, "valid")
    shuffled = run_dir / "artifacts" / "_misaligned.csv"
    lines = out_csv.read_text().splitlines()
    body = [ln.split(",") for ln in lines[1:]]
    # Swap the payloads but keep row_id sequential, so this exercises the
    # alignment check specifically rather than tripping the row_id guard first.
    body[0][1:], body[1][1:] = body[1][1:], body[0][1:]
    shuffled.write_text("\n".join([lines[0]] + [",".join(r) for r in body]) + "\n")
    try:
        scorer.read_scores(shuffled, rows)
    except scorer.ContractError as e:
        print(f"  rejected: {str(e).splitlines()[0][:90]}")
        return
    raise AssertionError("misaligned submission was accepted")


@check("9. scorer refuses the test split")
def _():
    try:
        scorer.score_file(out_csv, split="test", data_dir=REAL_DATA)
    except scorer.IntegrityError as e:
        print(f"  refused: {str(e).splitlines()[0][:90]}")
        return
    raise AssertionError("test split was scored outside the seal")


# ---------------------------------------------------------------- report
jr.append("harness_check_end", status="ok" if not failed else "error",
          passed=len(passed), failed=len(failed))
(run_dir / "run_log.md").write_text(
    render_markdown(jr.read(), title="Harness check - Phase 1 milestone"))

print("\n" + "=" * 62)
print(f"{len(passed)}/{len(passed) + len(failed)} checks passed")
for name, err in failed:
    print(f"  FAILED {name}: {err}")
print(f"journal:  {run_dir / 'journal.jsonl'}")
print(f"run log:  {run_dir / 'run_log.md'}")
if failed:
    raise SystemExit(1)
print("\nHarness is sound. The loop can be built on it.")
