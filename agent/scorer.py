"""Step 2b - validate a candidate's output and score it with the official metric.

Two jobs, both about trust.

1. Integrity. `evaluate.py` defines the entire scoring convention and the rules
   say never modify it. This module refuses to score anything if that file's
   checksum has changed, so a candidate that "improves its score" by rewriting
   the metric fails loudly instead of silently.

2. Alignment. The submission format is positional: row N of the CSV scores row
   N of `data.load()[split]`. `(user_id, video_id)` is NOT a key - the test
   split has 3.06% duplicate pairs. A candidate that shuffles its rows would
   otherwise score like noise and be indistinguishable from a bad idea. Every
   row is checked against the eval set before anything is scored.

The alignment logic mirrors the organizer's `submit.py:read_submission`; it is
reimplemented rather than imported so that `submit.py` stays untouched and the
error messages can be fed to the agent in English.
"""
import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.paths import EVALUATE_PY, VISIBLE_DATA, add_starter_to_path

HEADER = ["row_id", "user_id", "video_id", "score"]

# sha256 of the organizer's pristine evaluate.py, recorded at Phase 1.
EVALUATE_SHA256 = "ecfde28392eb14fec4f488083251df50624e1af2b86278b962daecfb42d195de"


class IntegrityError(Exception):
    """The official metric implementation has been modified."""


class ContractError(Exception):
    """The candidate's output does not satisfy the submission contract."""


def assert_evaluate_untouched():
    actual = hashlib.sha256(EVALUATE_PY.read_bytes()).hexdigest()
    if actual != EVALUATE_SHA256:
        raise IntegrityError(
            f"{EVALUATE_PY} has been modified.\n"
            f"  expected sha256 {EVALUATE_SHA256}\n"
            f"  actual   sha256 {actual}\n"
            f"Restore it from git before scoring anything.")
    return actual


def load_eval_rows(data_dir, split):
    """The ground-truth rows a submission must align to."""
    add_starter_to_path()
    from data import load
    splits = load(str(data_dir))
    rows = splits[split]
    if not rows:
        raise ContractError(
            f"split '{split}' is empty in {data_dir}. If this is the agent's "
            f"visible directory, 'test' is empty by design - that is the "
            f"firewall working, not a bug.")
    return rows


def read_scores(path, rows):
    """Parse a candidate's CSV, checking the contract row by row."""
    path = Path(path)
    if not path.exists():
        raise ContractError(f"candidate produced no output file at {path}")
    scores = []
    n = 0
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        head = next(reader, None)
        if head != HEADER:
            raise ContractError(
                f"header must be exactly {','.join(HEADER)} - got {head}")
        for lineno, rec in enumerate(reader, start=2):
            if len(rec) != 4:
                raise ContractError(
                    f"line {lineno}: {len(rec)} fields, expected 4")
            rid, uid, vid, sc = rec
            if n >= len(rows):
                raise ContractError(
                    f"line {lineno}: more rows than the eval set "
                    f"({len(rows):,d} rows)")
            try:
                if int(rid) != n:
                    raise ContractError(
                        f"line {lineno}: row_id={rid}, expected {n} "
                        f"(must start at 0 and increment by 1)")
            except ValueError:
                raise ContractError(f"line {lineno}: row_id={rid!r} is not an integer")
            if uid != rows[n][1] or vid != rows[n][2]:
                raise ContractError(
                    f"line {lineno}: misaligned - submission has "
                    f"({uid},{vid}) but eval row {n} is "
                    f"({rows[n][1]},{rows[n][2]}). Row order must match "
                    f"data.load()[split] exactly; do not sort or shuffle.")
            try:
                v = float(sc)
            except ValueError:
                raise ContractError(f"line {lineno}: score {sc!r} is not a number")
            if v != v or v in (float("inf"), float("-inf")):
                raise ContractError(f"line {lineno}: score is NaN or Inf")
            scores.append(v)
            n += 1
    if n != len(rows):
        raise ContractError(
            f"submission has {n:,d} rows, eval set has {len(rows):,d}")
    return scores


@dataclass
class Score:
    gauc: float
    ndcg5: float
    primary: float
    users: int
    rows: int
    split: str

    def to_dict(self):
        return {"GAUC": self.gauc, "nDCG@5": self.ndcg5,
                "primary": self.primary, "users": self.users,
                "rows": self.rows, "split": self.split}

    def __str__(self):
        return (f"GAUC {self.gauc:.4f} | nDCG@5 {self.ndcg5:.4f} | "
                f"primary {self.primary:.4f}  ({self.rows:,d} rows, "
                f"{self.users:,d} users, split={self.split})")


def score_file(path, split="valid", data_dir=None, allow_test=False):
    """Validate a candidate's output and score it with the official metric."""
    if split == "test" and not allow_test:
        raise IntegrityError(
            "refusing to score the test split. Only seal/final_score.py may "
            "do that, once, at the end of the project.")
    data_dir = Path(data_dir) if data_dir else VISIBLE_DATA

    assert_evaluate_untouched()
    rows = load_eval_rows(data_dir, split)
    scores = read_scores(path, rows)

    add_starter_to_path()
    from evaluate import evaluate
    r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
    return Score(gauc=r["GAUC"], ndcg5=r["nDCG@5"], primary=r["primary"],
                 users=r["users"], rows=r["rows"], split=split)
