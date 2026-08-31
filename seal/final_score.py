"""SEALED - the only script permitted to touch the test split. Run once.

Nothing under agent/ imports this, and nothing here is importable by the loop.
It takes a candidate script that the agent already selected on validation,
re-runs it against the FULL dataset with --split test, and scores it.

The agent must be finished before this runs. Choosing a candidate after
seeing its test score is exactly the leak the firewall exists to prevent, so
this script records what it ran, refuses to overwrite an existing result
without --force, and writes an audit trail next to the scores.

    python3 seal/final_score.py --candidate candidates/fm_baseline.py
    python3 seal/final_score.py --candidate ... --seeds 0 1 2 3 4

Reporting more than one seed is strongly recommended: the baseline's own
seed-to-seed std on test primary is 0.0008, so a single-seed delta under
about 0.002 is not distinguishable from noise.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import scorer
from agent.executor import run_script
from agent.paths import REAL_DATA, REPO, WORK

BASELINE_TEST_PRIMARY = 0.5946      # the number to beat
ORACLE_TEST_PRIMARY = 0.8645        # judge progress against this, not 1.0
SEAL_DIR = WORK / "sealed_result"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True,
                    help="path to the validation-selected candidate script")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--timeout_s", type=int, default=3600)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing sealed result")
    a = ap.parse_args()

    cand = Path(a.candidate).resolve()
    if not cand.exists():
        raise SystemExit(f"candidate not found: {cand}")

    SEAL_DIR.mkdir(parents=True, exist_ok=True)
    result_path = SEAL_DIR / "final_result.json"
    if result_path.exists() and not a.force:
        raise SystemExit(
            f"{result_path} already exists.\n"
            f"The test split is meant to be scored once. Re-running it to "
            f"pick a better candidate is the leak the firewall prevents.\n"
            f"Pass --force only if you genuinely intend to overwrite.")

    print(f"SEALED RUN - reading the full dataset at {REAL_DATA}")
    print(f"candidate: {cand}")
    print(f"seeds    : {a.seeds}\n")

    audit = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "candidate": str(cand.relative_to(REPO)),
        "candidate_sha256": hashlib.sha256(cand.read_bytes()).hexdigest(),
        "evaluate_sha256": scorer.assert_evaluate_untouched(),
        "data_dir": str(REAL_DATA),
        "seeds": a.seeds,
        "runs": [],
    }

    primaries = []
    for seed in a.seeds:
        out = SEAL_DIR / f"scores_test_seed{seed}.csv"
        print(f"[seed {seed}] running ...")
        r = run_script(cand,
                       ["--data_dir", REAL_DATA, "--split", "test",
                        "--out", out, "--seed", seed],
                       timeout_s=a.timeout_s)
        if not r.ok:
            print(r.stderr[-2000:])
            raise SystemExit(f"[seed {seed}] candidate failed: {r.summary()}")

        s = scorer.score_file(out, split="test", data_dir=REAL_DATA,
                              allow_test=True)
        primaries.append(s.primary)
        print(f"[seed {seed}] {s}  ({r.wall_s:.1f}s)")
        audit["runs"].append({"seed": seed, "wall_s": round(r.wall_s, 1),
                              "scores_csv": out.name, **s.to_dict()})

    mean = statistics.fmean(primaries)
    std = statistics.stdev(primaries) if len(primaries) > 1 else None
    delta = mean - BASELINE_TEST_PRIMARY
    captured = delta / (ORACLE_TEST_PRIMARY - BASELINE_TEST_PRIMARY)

    audit["summary"] = {
        "mean_test_primary": mean,
        "std_test_primary": std,
        "baseline_test_primary": BASELINE_TEST_PRIMARY,
        "delta_vs_baseline": delta,
        "oracle_test_primary": ORACLE_TEST_PRIMARY,
        "fraction_of_remaining_headroom": captured,
        "beat_baseline": delta > 0,
    }
    result_path.write_text(json.dumps(audit, indent=2) + "\n")

    print("\n" + "=" * 62)
    print(f"test primary  : {mean:.4f}" + (f" +/- {std:.4f}" if std else ""))
    print(f"FM baseline   : {BASELINE_TEST_PRIMARY:.4f}")
    print(f"delta         : {delta:+.4f}")
    print(f"oracle ceiling: {ORACLE_TEST_PRIMARY:.4f}")
    print(f"captured {captured:+.1%} of the headroom the baseline left on the table")
    print(f"\naudit trail: {result_path}")


if __name__ == "__main__":
    main()
