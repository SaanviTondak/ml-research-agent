"""Assemble the submission packet from a completed (or stopped) run.

Works from journal.jsonl and state.json, which are written and flushed as the
run proceeds - so this produces a complete packet even if the loop was stopped
mid-iteration rather than converging. Nothing here is reconstructed from
memory; every number comes from a file the loop wrote at the time.

    python3 package.py --run_dir work/runs/final_01
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.journal import Journal, render_markdown
from agent.state import SolutionJournal, EPS, N_CONVERGE
from agent.paths import REPO, WORK

BASELINE_VALID = 0.6016
BASELINE_TEST = 0.5946
ORACLE_TEST = 0.8645


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default="submission")
    a = ap.parse_args()

    run = Path(a.run_dir)
    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)

    state = SolutionJournal(run / "state.json")
    journal = Journal(run / "journal.jsonl")
    records = journal.read()
    best = state.best()

    # ---------------------------------------------------------- run log
    (out / "run_log.md").write_text(
        render_markdown(records, title=f"Autonomous run log - {run.name}"))

    # ---------------------------------------------------------- results
    sealed_path = WORK / "sealed_result" / "final_result.json"
    sealed = json.loads(sealed_path.read_text()) if sealed_path.exists() else None

    L = ["# Results", "", "## Validation (agent-visible, used for selection)", "",
         "| | GAUC | nDCG@5 | primary | vs baseline |", "|---|---|---|---|---|",
         f"| FM baseline | 0.6674 | 0.5357 | {BASELINE_VALID:.4f} | - |"]
    if best and best.metrics:
        m = best.metrics
        L.append(f"| **agent best (node #{best.id})** | {m['GAUC']:.4f} | "
                 f"{m['nDCG@5']:.4f} | **{best.score:.4f}** | "
                 f"{best.score - BASELINE_VALID:+.4f} |")
        if best.n_seeds > 1:
            L += ["", f"Agent best verified over {best.n_seeds} seeds: "
                      f"{', '.join(f'{k}={v:.4f}' for k, v in sorted(best.seed_scores.items()))}"
                      + (f" (std {best.seed_std:.4f})" if best.seed_std else "")]
    L += ["", f"Selection rule: validation-best at stop, not the running peak.",
          f"Convergence rule: eps={EPS}, N={N_CONVERGE} (organizers').", ""]

    if sealed:
        s = sealed["summary"]
        L += ["## Held-out test (scored once, by seal/final_score.py)", "",
              "| | primary |", "|---|---|",
              f"| random | 0.4757 |", f"| item popularity | 0.5715 |",
              f"| FM baseline | {BASELINE_TEST:.4f} |",
              f"| **this agent** | **{s['mean_test_primary']:.4f}**"
              + (f" +/- {s['std_test_primary']:.4f}" if s.get('std_test_primary') else "") + " |",
              f"| oracle ceiling | {ORACLE_TEST:.4f} |", "",
              f"**Delta vs baseline: {s['delta_vs_baseline']:+.4f}**  ",
              f"Fraction of the headroom the baseline left on the table: "
              f"{s['fraction_of_remaining_headroom']:+.1%}", "",
              f"Seeds: {sealed['seeds']}. Candidate sha256 "
              f"`{sealed['candidate_sha256'][:16]}...`, evaluate.py sha256 "
              f"`{sealed['evaluate_sha256'][:16]}...`.", ""]
    else:
        L += ["## Held-out test", "", "Not yet scored. Run:", "", "```bash",
              f"python3 seal/final_score.py --candidate {a.out}/best_candidate.py "
              f"--seeds 0 1 2 3 4", "```", ""]

    # -------------------------------------------------------- attempts
    L += ["## Every attempt", "",
          "| # | stage | outcome | valid primary | hypothesis |", "|---|---|---|---|---|"]
    for n in state.nodes:
        sc = f"{n.score:.4f}" if n.score is not None else "-"
        oc = "failed" if n.is_buggy else "scored"
        hyp = (n.hypothesis or "").replace("|", "/")[:120]
        L.append(f"| {n.id} | {n.stage} | {oc} | {sc} | {hyp} |")
    (out / "results.md").write_text("\n".join(L) + "\n")

    # ------------------------------------------------------- resources
    end = next((r for r in reversed(records) if r["event"] == "run_end"), None)
    tok = (end or {}).get("tokens", {})
    guard_rejects = sum(1 for r in records if r["event"] == "guard_rejected")
    recoveries = sum(1 for r in records
                     if r["event"] == "node_added" and r.get("stage") == "debug")
    R = ["# Resource report", "",
         "| | |", "|---|---|",
         f"| Iterations used | {len(state.nodes)} of 50 |",
         f"| Wall-clock | {(end or {}).get('elapsed_h', '?')} h of 6 h cap |",
         f"| LLM calls | {tok.get('calls', '?')} |",
         f"| LLM tokens | {tok.get('total_tokens', 0):,} "
         f"({tok.get('prompt_tokens', 0):,} in / {tok.get('completion_tokens', 0):,} out) |",
         f"| API retries absorbed | {tok.get('retries', '?')} |",
         "| GPU-hours | 0 (numpy on one CPU core) |",
         f"| Stop reason | {(end or {}).get('stop_reason', 'run not finalised')} |", "",
         "## Autonomy", "",
         "| | |", "|---|---|",
         f"| Manual interventions | {(end or {}).get('interventions', 0)} |",
         f"| Failures recovered from, unaided | {recoveries} |",
         f"| Candidates rejected by the leak guard | {guard_rejects} |", "",
         "Restarting a crashed process is not counted as an intervention "
         "(organizers' Q&A). Editing the agent's code, prompts or candidates "
         "mid-run is. See docs/interventions.md.", ""]
    (out / "resources.md").write_text("\n".join(R) + "\n")

    # ------------------------------------------------------- artifacts
    if best:
        src = run / "nodes" / f"node_{best.id:03d}.py"
        if src.exists():
            shutil.copy2(src, out / "best_candidate.py")
    for name in ("submission.csv", "final_result.json"):
        p = WORK / "sealed_result" / name
        if p.exists():
            shutil.copy2(p, out / name)
    shutil.copy2(run / "journal.jsonl", out / "journal.jsonl")
    if (run / "eda_report.txt").exists():
        shutil.copy2(run / "eda_report.txt", out / "eda_report.txt")

    print(f"packet written to {out}/")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:<24} {f.stat().st_size:>9,d} bytes")
    if best:
        print(f"\nbest: node #{best.id} valid {best.score:.4f} "
              f"({best.score - BASELINE_VALID:+.4f} vs baseline) "
              f"over {best.n_seeds} seed(s)")
    if not sealed:
        print("\nNOT YET SCORED ON TEST. Next:")
        print(f"  python3 seal/final_score.py --candidate {a.out}/best_candidate.py --seeds 0 1 2")


if __name__ == "__main__":
    main()
