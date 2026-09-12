"""Render the run's score-per-attempt chart from the loop's own state file.

    python3 tools/render_progress.py [--run final_01]

Prints to stdout. The README embeds the output verbatim; regenerate and paste
if the run changes, so the picture can never drift from the journal.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="final_01")
ap.add_argument("--baseline", type=float, default=0.6016,
                help="FM baseline on validation")
a = ap.parse_args()

nodes = json.loads((ROOT / "work" / "runs" / a.run / "state.json").read_text())["nodes"]
scores = [n["score"] for n in nodes if n.get("score") is not None]

LO, HI, W = 0.46, 0.61, 46
STAGE = {"draft": "draft  ", "improve": "improve", "debug": "debug  "}


def col(v):
    return max(1, min(W, int(round((v - LO) / (HI - LO) * W))))


base_col = col(a.baseline)
head = " " * (base_col - 1) + "|"
print(f"                {head}  FM baseline {a.baseline:.4f}")

for n in nodes:
    sc = n.get("score")
    stage = STAGE[n["stage"]]
    if sc is None:
        row = " " * (base_col - 1) + ":"
        print(f"  {n['id']:>2}  {stage}  {row:<{W}}  failed")
    else:
        c = col(sc)
        bar = "█" * c
        if c < base_col:
            bar += " " * (base_col - c - 1) + ":"
        mark = "  <-- new best" if sc > a.baseline and sc >= max(
            x for x in scores[:nodes.index(n) + 1]) else ""
        print(f"  {n['id']:>2}  {stage}  {bar:<{W}}  {sc:.4f}{mark}")
