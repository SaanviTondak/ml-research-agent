"""Step 3b - the solution journal: every attempt, its code, and what it scored.

This is the agent's memory. Without it the loop has no way to tell an
improvement from noise, no way to avoid re-trying what already failed, and
nothing to hand the model as context on the next iteration.

Structure is a tree, not a list. Each node records the attempt it was derived
from, so the run reconstructs as "this idea came from that one, and here is
what changed" rather than an undifferentiated pile of scripts. Debug attempts
hang off the broken node they are fixing; improvements hang off the node they
build on.

Two rules from the project brief are enforced here rather than left to the
loop, because both are easy to get subtly wrong:

  * Promotion requires multiple seeds. Seed-to-seed std on this benchmark is
    0.0008, so a single-seed gain under ~0.002 is indistinguishable from luck.
    A node keeps every seed it was run on and reports the mean.
  * Convergence is the organizers' rule, not ours: stop when N=3 consecutive
    iterations fail to improve the best validation score by more than
    eps=0.002.
"""
import json
import statistics
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EPS = 0.002          # organizers' convergence epsilon (~2.5 sigma)
N_CONVERGE = 3       # consecutive non-improving iterations before stopping


@dataclass
class Node:
    """One attempt: an idea, the code it produced, and what happened."""
    id: int
    stage: str                       # draft | improve | debug
    hypothesis: str                  # why the agent tried this
    code: str
    parent_id: int | None = None

    # execution
    exec_ok: bool = False
    exec_summary: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    wall_s: float = 0.0

    # scoring: one entry per seed, so promotion can require agreement
    seed_scores: dict = field(default_factory=dict)   # {seed: primary}
    metrics: dict | None = None                       # full metrics, seed 0
    is_buggy: bool = True                             # until proven otherwise
    failure_reason: str = ""
    created_at: str = ""

    @property
    def score(self):
        """Mean validation primary across every seed this node was run on."""
        if not self.seed_scores:
            return None
        return statistics.fmean(self.seed_scores.values())

    @property
    def seed_std(self):
        v = list(self.seed_scores.values())
        return statistics.stdev(v) if len(v) > 1 else None

    @property
    def n_seeds(self):
        return len(self.seed_scores)

    def one_line(self):
        s = f"{self.score:.4f}" if self.score is not None else "  --  "
        seeds = f" x{self.n_seeds}" if self.n_seeds > 1 else ""
        flag = "BROKE" if self.is_buggy else "ok   "
        return (f"#{self.id:<3} {self.stage:<7} {flag} {s}{seeds:<4} "
                f"{self.hypothesis.splitlines()[0][:88] if self.hypothesis else ''}")

    def to_dict(self):
        d = asdict(self)
        d["score"] = self.score
        return d


class SolutionJournal:
    """The tree of attempts, persisted after every change."""

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.nodes: list[Node] = []
        if self.path and self.path.exists():
            self.load()

    # ----------------------------------------------------------- persistence
    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"nodes": [n.to_dict() for n in self.nodes]},
            indent=2, default=str) + "\n")

    def load(self):
        d = json.loads(self.path.read_text())
        self.nodes = []
        for raw in d.get("nodes", []):
            raw.pop("score", None)                       # derived, not stored
            raw["seed_scores"] = {int(k): v for k, v
                                  in (raw.get("seed_scores") or {}).items()}
            self.nodes.append(Node(**raw))

    # ------------------------------------------------------------- mutation
    def add(self, node):
        self.nodes.append(node)
        self.save()
        return node

    def next_id(self):
        return len(self.nodes)

    # -------------------------------------------------------------- queries
    def get(self, node_id):
        return next((n for n in self.nodes if n.id == node_id), None)

    def good(self):
        """Nodes that ran and scored."""
        return [n for n in self.nodes if not n.is_buggy and n.score is not None]

    def buggy(self):
        return [n for n in self.nodes if n.is_buggy]

    def best(self):
        g = self.good()
        return max(g, key=lambda n: n.score) if g else None

    def best_score(self):
        b = self.best()
        return b.score if b else None

    def children(self, node_id):
        return [n for n in self.nodes if n.parent_id == node_id]

    # ------------------------------------------------------------ selection
    def select_parent(self):
        """Greedy: improve the best working solution.

        If the best node's most recent child broke, fix that child instead -
        an unfixed crash otherwise gets abandoned in favour of re-improving
        the same parent over and over, which wastes iterations on ground the
        agent has already covered.
        """
        best = self.best()
        if best is None:
            return None
        broken_kids = [c for c in self.children(best.id) if c.is_buggy]
        if broken_kids:
            return broken_kids[-1]
        return best

    # ---------------------------------------------------------- convergence
    def improvement_history(self):
        """Best-so-far after each node, in order."""
        hist, best = [], None
        for n in self.nodes:
            if not n.is_buggy and n.score is not None:
                best = n.score if best is None else max(best, n.score)
            hist.append(best)
        return hist

    def has_converged(self, eps=EPS, n=N_CONVERGE):
        """The organizers' rule: N consecutive iterations gaining <= eps."""
        hist = [h for h in self.improvement_history() if h is not None]
        if len(hist) < n + 1:
            return False
        window = hist[-(n + 1):]
        return all(window[i + 1] - window[i] <= eps for i in range(n))

    # -------------------------------------------------------------- context
    def summary(self, limit=25):
        """Compact history for the model's prompt. Newest last."""
        if not self.nodes:
            return "(no attempts yet)"
        rows = [n.one_line() for n in self.nodes[-limit:]]
        head = f"id   stage   status  valid   hypothesis"
        return "\n".join([head, "-" * 100] + rows)
