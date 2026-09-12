"""Compare two agent runs from their own journals. No numbers typed by hand.

    python3 tools/compare_runs.py final_01 final_02

Reports what each run spent its iteration budget on, which is the thing the
search-policy fix was meant to change. See docs/postmortem.md.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def classify(reason):
    """Bucket a node's failure_reason into the taxonomy used in the postmortem."""
    if not reason:
        return None
    r = reason
    if "stopped at the output token limit" in r or "cut off at the output token limit" in r:
        return "truncated at output limit"
    if "not valid Python" in r:
        return "SyntaxError (caught pre-exec)"
    if "SyntaxError" in r:
        return "SyntaxError (found by running it)"
    if "NaN or Inf" in r:
        return "NaN/Inf in output scores"
    if "timed out" in r:
        return "timeout"
    if "rejected before execution" in r:
        return "guard rejection"
    if "no fenced python" in r:
        return "no code block in response"
    for exc in ("IndexError", "ValueError", "KeyError", "TypeError",
                "MemoryError", "AttributeError", "ZeroDivisionError"):
        if exc in r:
            return exc
    if "failed validation" in r:
        return "contract violation"
    return "other"


def load(run):
    d = ROOT / "work" / "runs" / run
    state = json.loads((d / "state.json").read_text())["nodes"]
    events = []
    jp = d / "journal.jsonl"
    if jp.exists():
        events = [json.loads(l) for l in jp.read_text().splitlines() if l.strip()]
    return d, state, events


def stats(run):
    d, nodes, events = load(run)
    scored = [n for n in nodes if n.get("score") is not None]
    failed = [n for n in nodes if n.get("score") is None]
    by_stage = Counter(n["stage"] for n in nodes)

    # Budget concentration: the largest number of nodes sharing one parent.
    parents = Counter(n["parent_id"] for n in nodes if n["parent_id"] is not None)
    worst_parent, worst_n = (parents.most_common(1)[0] if parents else (None, 0))

    end = [e for e in events if e.get("event") == "run_end"]
    end = end[-1] if end else {}
    tok = end.get("tokens", {}) or {}

    return {
        "run": run,
        "nodes": len(nodes),
        "scored": len(scored),
        "failed": len(failed),
        "best": max((n["score"] for n in scored), default=None),
        "best_node": max(scored, key=lambda n: n["score"])["id"] if scored else None,
        "stages": by_stage,
        "debug_share": by_stage["debug"] / len(nodes) if nodes else 0,
        "worst_parent": worst_parent,
        "worst_parent_n": worst_n,
        "abandoned": [e.get("node_id") for e in events
                      if e.get("event") == "node_abandoned"],
        "failures": Counter(classify(n.get("failure_reason")) for n in failed),
        "hours": end.get("elapsed_h"),
        "tokens": tok.get("total_tokens"),
        "stop_reason": end.get("reason") or end.get("stop_reason"),
    }


def fmt(v, spec="", dash="-"):
    return dash if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs=2)
    a = ap.parse_args()
    A, B = (stats(r) for r in a.runs)

    print(f"# {A['run']} vs {B['run']}\n")
    rows = [
        ("attempts", str(A["nodes"]), str(B["nodes"])),
        ("scored", f"{A['scored']} ({A['scored']/max(A['nodes'],1):.0%})",
                   f"{B['scored']} ({B['scored']/max(B['nodes'],1):.0%})"),
        ("failed", str(A["failed"]), str(B["failed"])),
        ("best valid primary", fmt(A["best"], ".4f"), fmt(B["best"], ".4f")),
        ("best node", f"#{A['best_node']}", f"#{B['best_node']}"),
        ("iterations in `debug`",
         f"{A['stages']['debug']} ({A['debug_share']:.0%})",
         f"{B['stages']['debug']} ({B['debug_share']:.0%})"),
        ("most attempts on one parent",
         f"{A['worst_parent_n']} (node {A['worst_parent']})",
         f"{B['worst_parent_n']} (node {B['worst_parent']})"),
        ("branches abandoned",
         str(len(A["abandoned"])) + (f" {A['abandoned']}" if A["abandoned"] else ""),
         str(len(B["abandoned"])) + (f" {B['abandoned']}" if B["abandoned"] else "")),
        ("wall-clock", fmt(A["hours"], ".2f") + " h", fmt(B["hours"], ".2f") + " h"),
        ("LLM tokens", fmt(A["tokens"], ","), fmt(B["tokens"], ",")),
    ]
    w = max(len(r[0]) for r in rows)
    print(f"| {'':<{w}} | {A['run']:>14} | {B['run']:>14} |")
    print(f"|{'-'*(w+2)}|{'-'*16}|{'-'*16}|")
    for label, x, y in rows:
        print(f"| {label:<{w}} | {x:>14} | {y:>14} |")

    print("\n## Failures by cause\n")
    causes = sorted(set(A["failures"]) | set(B["failures"]),
                    key=lambda c: -(A["failures"][c] + B["failures"][c]))
    print(f"| cause | {A['run']} | {B['run']} |")
    print("|---|---|---|")
    for c in causes:
        if c is None:
            continue
        print(f"| {c} | {A['failures'][c]} | {B['failures'][c]} |")


if __name__ == "__main__":
    main()
