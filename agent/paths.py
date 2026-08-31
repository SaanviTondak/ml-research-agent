"""Canonical paths + the sys.path shim for the organizer's starter kit.

Every other module imports its paths from here so that the location of the
sealed data directory is stated in exactly one place.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STARTER = REPO / "kuairand-starter-kit"
EVALUATE_PY = STARTER / "evaluate.py"

# The full dataset, including the test-window rows and their labels.
# SEALED: only seal/final_score.py may point a candidate at this directory.
REAL_DATA = STARTER / "KuaiRand-Pure" / "data"

WORK = REPO / "work"
VISIBLE_DATA = WORK / "data_visible"   # what the agent is allowed to see
RUNS = WORK / "runs"

CANDIDATES = REPO / "candidates"
DOCS = REPO / "docs"

# Splits, restated from the organizer's data.py so that firewall.py can run
# before (and independently of) importing it.
SPLIT_RANGES = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test":  (20220429, 20220508),
}
# Everything strictly after this date belongs to the test window.
VISIBLE_MAX_DATE = SPLIT_RANGES["valid"][1]
VISIBLE_MIN_DATE = SPLIT_RANGES["train"][0]
VISIBLE_SPLITS = ("train", "valid")


def add_starter_to_path():
    """Make `import data` / `import evaluate` resolve to the organizer's files."""
    p = str(STARTER)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p
