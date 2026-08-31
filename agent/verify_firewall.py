"""Independent proof that work/data_visible/ contains no test-window data.

firewall.py builds the directory; this module re-checks it from scratch,
without trusting the manifest, by re-reading every row that was written.
It is deliberately dumb: a second implementation is worth more as evidence
than a clever one.

Run:  python3 -m agent.verify_firewall
Exit code 0 = firewall intact. Non-zero = do not start a run.
"""
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.paths import (VISIBLE_DATA, VISIBLE_MAX_DATE, SPLIT_RANGES,
                         add_starter_to_path)


class FirewallBreach(Exception):
    """Raised when the agent-visible data directory is not provably clean."""


def _check_no_test_dates():
    """Re-read every log row and assert none falls in the test window."""
    lo_test, hi_test = SPLIT_RANGES["test"]
    findings = []
    for path in sorted(VISIBLE_DATA.glob("log_*.csv")):
        worst = None
        n = 0
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            date_col = next(reader).index("date")
            for rec in reader:
                d = int(rec[date_col])
                n += 1
                if d > VISIBLE_MAX_DATE:
                    raise FirewallBreach(
                        f"{path.name} row {n} has date {d}, past the visible "
                        f"cutoff {VISIBLE_MAX_DATE}")
                if lo_test <= d <= hi_test:      # unreachable given the above
                    raise FirewallBreach(
                        f"{path.name} row {n} is inside the test window")
                worst = d if worst is None else max(worst, d)
        findings.append((path.name, n, worst))
    return findings


def _check_loader_returns_no_test():
    """The organizer's own loader must report an empty test split."""
    add_starter_to_path()
    from data import load
    splits = load(str(VISIBLE_DATA))
    if len(splits["test"]) != 0:
        raise FirewallBreach(
            f"data.load() returned {len(splits['test'])} test rows from the "
            f"visible directory; expected 0")
    return {k: len(v) for k, v in splits.items()}


def verify(verbose=True):
    if not VISIBLE_DATA.is_dir():
        raise FirewallBreach(
            f"{VISIBLE_DATA} does not exist - run `python3 -m agent.firewall`")

    findings = _check_no_test_dates()
    if verbose:
        for name, n, worst in findings:
            print(f"  {name:44s} {n:>9,d} rows, max date {worst}  OK")

    counts = _check_loader_returns_no_test()
    if verbose:
        print(f"  data.load(data_visible) -> "
              + ", ".join(f"{k}={v:,d}" for k, v in counts.items()))
        print(f"  test split is empty                            OK")
    return counts


if __name__ == "__main__":
    print(f"verifying {VISIBLE_DATA} ...")
    try:
        verify()
    except FirewallBreach as e:
        print(f"\nFIREWALL BREACH: {e}")
        raise SystemExit(2)
    print("\nfirewall intact: no test-window row is reachable by the agent.")
