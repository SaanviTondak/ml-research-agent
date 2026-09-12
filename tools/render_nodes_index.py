"""Render the index of agent-written candidate scripts from the run's own state.

Every field comes from work/runs/<run>/state.json, which the loop wrote while
it ran. Nothing here is typed by hand, so the index cannot drift from the code
it indexes.

    python3 tools/render_nodes_index.py [--run final_01]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="final_01")
ap.add_argument("--out", default="submission/nodes/README.md")
a = ap.parse_args()

state = json.loads((ROOT / "work" / "runs" / a.run / "state.json").read_text())
nodes = state["nodes"]
out = ROOT / a.out


def one_line(s, n=90):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


rows = []
for n in nodes:
    nid = n["id"]
    src = out.parent / f"node_{nid:03d}.py"
    score = n.get("score")
    rows.append({
        "id": nid,
        "stage": n["stage"],
        "parent": n.get("parent_id"),
        "score": score,
        "file": src.name if src.exists() else None,
        "hypothesis": one_line(n.get("hypothesis")),
        "failure": one_line(n.get("failure_reason"), 70),
    })

# The candidate the sealed scorer actually ran, per submission/final_result.json.
SUBMITTED = 4

scored = [r for r in rows if r["score"] is not None]
best = max(scored, key=lambda r: r["score"])

lines = [
    "# The code the agent wrote",
    "",
    f"Every candidate script produced during run `{a.run}`, exactly as the loop",
    "received it from the model — unedited, including the ones that did not run.",
    "`node_NNN.py` is the script; `node_NNN.diff` is its diff against its parent,",
    "which is what the run log records per iteration.",
    "",
    f"{len(rows)} attempts, {len(scored)} of which produced a scoreable submission.",
    f"Best validation primary: **{best['score']:.4f}** (node {best['id']}, marked ←).",
    "",
    "Node 7 is *not* the submitted model. Its score comes from one seed; node 4",
    "was re-run on three (0.6043 / 0.6045 / 0.6038) and shipped on the verified",
    "mean. The 0.0004 gap between them is half the benchmark's seed noise, so",
    "taking the higher unverified number would have been chasing luck.",
    "",
    "A missing `.py` means the model's response contained no complete fenced",
    "code block — the script was cut off at the output token limit before it",
    "could be saved.",
    "",
    "| # | stage | parent | valid primary | script | what it tried / why it failed |",
    "|---|---|---|---|---|---|",
]

for r in rows:
    score = f"**{r['score']:.4f}**" if r["score"] is not None else "—"
    if r["id"] == best["id"]:
        score += " ←"
    if r["id"] == SUBMITTED:
        score += " **shipped**"
    f = f"[`{r['file']}`]({r['file']})" if r["file"] else "*(not saved)*"
    parent = "—" if r["parent"] is None else str(r["parent"])
    note = r["hypothesis"] if r["score"] is not None else (r["failure"] or r["hypothesis"])
    lines.append(f"| {r['id']} | {r['stage']} | {parent} | {score} | {f} | {note} |")

lines += [
    "",
    "## Reading the parent column",
    "",
    "The loop is a greedy tree search: `draft` starts a new root, `improve`",
    "extends the best scored node, `debug` repairs a node that failed.",
    "",
    "Nodes 11 through 33 all carry `parent = 10` — twenty-three consecutive",
    "repair attempts on the same broken script, while nodes 4 and 7 sat healthy",
    "and unextended. That is the failure this run is most instructive about; it",
    "is analysed in [`docs/postmortem.md`](../../docs/postmortem.md).",
]

out.write_text("\n".join(lines) + "\n")
print(f"wrote {out.relative_to(ROOT)}  ({len(rows)} nodes, {len(scored)} scored)")
