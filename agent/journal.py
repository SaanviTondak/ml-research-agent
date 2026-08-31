"""Step 2c - the run log, emitted live.

The competition grades the run log as a deliverable, and a log reconstructed
after the fact is both dishonest and detectable. So this is append-only and
flushed on every write: whatever the loop has done so far is on disk before
the next thing starts, and a hard crash loses at most the event in flight.

Format is JSON Lines - one self-contained JSON object per line. Append-only
means it can be tailed while a run is in progress, and a truncated final line
(from a kill -9) costs you one record rather than the file.

`render_markdown` turns the jsonl into the human-readable run log that ships
with the submission. The markdown is always regenerated from the jsonl; the
jsonl is the source of truth and is never rewritten.
"""
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.paths import RUNS


class Journal:
    """Append-only JSONL event log."""

    def __init__(self, path, run_id=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.path.parent.name
        self._seq = self._count_existing()

    def _count_existing(self):
        if not self.path.exists():
            return 0
        with open(self.path) as fh:
            return sum(1 for _ in fh)

    def append(self, event, **fields):
        """Write one record. Returns it, so callers can log and use in one line."""
        rec = {
            "seq": self._seq,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
            "event": event,
        }
        rec.update(fields)
        line = json.dumps(rec, default=str, ensure_ascii=False)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())      # survive a hard kill, not just an exit
        self._seq += 1
        return rec

    def read(self):
        """Every record, skipping a torn final line from an unclean shutdown."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass               # torn last write; the rest of the log stands
        return out


def new_run_dir(root=None, prefix="run"):
    """Timestamped directory for one run's journal and artifacts."""
    root = Path(root or RUNS)
    d = root / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    return d


_ICON = {"ok": "OK", "error": "ERR", "timeout": "TIMEOUT", "info": "-"}


def render_markdown(records, title="Run log"):
    """Human-readable log for the submission packet."""
    lines = [f"# {title}", ""]
    if records:
        lines += [f"Run `{records[0].get('run_id','?')}` - "
                  f"{records[0]['ts']} to {records[-1]['ts']} - "
                  f"{len(records)} events", ""]
    for r in records:
        head = f"### `{r['seq']:03d}` {r['event']}  <sub>{r['ts']}</sub>"
        lines.append(head)
        status = r.get("status")
        if status:
            lines.append(f"**{_ICON.get(status, status)}** {status}")
        for key in ("hypothesis", "note", "detail"):
            if r.get(key):
                lines.append(f"\n{r[key]}")
        if r.get("score"):
            s = r["score"]
            lines.append(
                f"\n| GAUC | nDCG@5 | primary |\n|---|---|---|\n"
                f"| {s.get('GAUC', float('nan')):.4f} "
                f"| {s.get('nDCG@5', float('nan')):.4f} "
                f"| **{s.get('primary', float('nan')):.4f}** |")
        if r.get("error"):
            lines.append(f"\n```\n{r['error']}\n```")
        skip = {"seq", "ts", "run_id", "event", "status", "score", "error",
                "hypothesis", "note", "detail"}
        extras = {k: v for k, v in r.items() if k not in skip}
        if extras:
            lines.append("\n" + "  \n".join(
                f"`{k}`: {v}" for k, v in extras.items()))
        lines.append("")
    return "\n".join(lines) + "\n"
