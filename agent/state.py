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
MAX_DEBUG_ATTEMPTS = 3
"""How many times one broken node may be repaired before it is abandoned.

Run final_01 spent 23 of its 34 iterations - 68% of the budget - repairing a
single node. select_parent() looked for broken children *of the best node*,
but a repair attaches to the node it repairs, not to best. So the repairs were
never children of best, the original breakage stayed selected, and the search
could not leave it. Nodes 4 and 7 sat scored and unextended throughout.

Three attempts is enough to fix a typo or an off-by-one. A node that survives
three is not going to be repaired by a fourth; the budget is better spent
improving something that already works. See docs/postmortem.md."""
VERIFY_MARGIN = 0.0008
"""How far a candidate must beat the best *verified* score to earn more seeds.

This used to be EPS, which conflated two different questions. EPS is the
organizers' convergence epsilon - "is the search finished?". This answers "is
this gain worth two more seed runs?". One seed sigma is the natural scale for
the second question and has nothing to do with the first.

Using EPS here is what let run final_02 converge six unverified steps past
solid ground: nodes 6-11 each advanced the incumbent by less than EPS
(+0.0005, -0.0001, +0.0005, -0.0014, -0.0002, +0.0001), so none of them
triggered verification, even though they compound. Replayed against node 5's
verified 0.60223, this margin fires on nodes 8 and 11."""

PROTECTED_SCORED_ATTEMPTS = 3
"""Scored attempts a new lineage gets before it must beat the global best.

Selection used to be a global argmax, so a new direction survived only if it
won on its very first scored attempt. Run final_02 proposed DIN-style
candidate attention twice and had it measured inside the noise floor both
times - node 2 lost by 0.0004 against a *single-seed* incumbent, node 9 by
0.0014, against a seed sigma of 0.0008 - and discarded it on the spot. Neither
ever got a second iteration or a second seed.

Three attempts is enough to tell a wrong direction from the first draft of a
right one. It is the same shape of budget as MAX_DEBUG_ATTEMPTS, and for the
same reason: the alternative is an unbounded commitment."""

PROTECTION_MAX_NODES = 5
"""Total nodes one exploration may consume, repairs included.

PROTECTED_SCORED_ATTEMPTS counts only nodes that scored, so without this a
lineage that keeps crashing would be protected forever. Five caps the
worst-case cost of a single exploration at a fifth of a typical run."""

PROTECTION_WRITE_OFF = 3 * EPS
"""How far behind the incumbent a lineage may be and still be developed.

Protection is for a direction that lost inside the noise, not for one that
lost outright. final_01 node 1 scored 0.5135 against a 0.5986 incumbent and
node 8 scored 0.4665; a lineage that far back is not two attempts away from
winning, and spending three iterations on it is how an exploration budget
becomes another trap."""

MIN_SCORED_BEFORE_CONVERGENCE = 12
"""The organizers' rule - three consecutive iterations gaining <= 0.002 -
presumes a search that has had a chance to get going. Applied from the first
iteration it terminates almost any run with a slow start: the first live run
stopped after four attempts, on the same iteration it set a new best, because
the two attempts before it happened not to improve.

That is the rule read literally and it is not what the rule is for. Convergence
now requires a floor of scored attempts first. The floor is a documented
interpretation, not a change to eps or N.

The floor counts *exploit* attempts only. Every protected iteration is by
construction one where the global best does not move, so charging exploration
against this floor would have exploration feed the rule that ends it - run
final_02 stopped at exactly twelve scored nodes, which is exactly this number.
Counting exploit attempts only keeps the figure and its justification
literally intact and makes exploration additive to it rather than deducted
from it."""


@dataclass
class PolicyConfig:
    """The thresholds the search runs under.

    These were module constants tuned against one benchmark, in that
    benchmark's units - VERIFY_MARGIN *was* its seed sigma. Holding them on an
    instance is what lets agent/calibrate.py measure a task's noise floor and
    install thresholds derived from it, instead of every task inheriting
    KuaiRand's.

    The defaults are exactly the previous constants, so a journal constructed
    without a config behaves as it always did.
    """
    eps: float = EPS
    n_converge: int = N_CONVERGE
    verify_margin: float = VERIFY_MARGIN
    max_debug_attempts: int = MAX_DEBUG_ATTEMPTS
    protected_scored_attempts: int = PROTECTED_SCORED_ATTEMPTS
    protection_max_nodes: int = PROTECTION_MAX_NODES
    protection_write_off: float = PROTECTION_WRITE_OFF
    min_scored_before_convergence: int = MIN_SCORED_BEFORE_CONVERGENCE
    verification_useful: bool = True
    source: str = "default"          # default | calibrated

    @classmethod
    def from_calibration(cls, cal, base=None):
        """Install measured thresholds, keeping the budget-shaped ones."""
        b = base or cls()
        if getattr(cal, "status", "") != "ok":
            return b
        return cls(eps=cal.eps,
                   n_converge=b.n_converge,
                   verify_margin=cal.verify_margin,
                   max_debug_attempts=b.max_debug_attempts,
                   protected_scored_attempts=b.protected_scored_attempts,
                   protection_max_nodes=b.protection_max_nodes,
                   protection_write_off=cal.protection_write_off,
                   min_scored_before_convergence=b.min_scored_before_convergence,
                   verification_useful=getattr(cal, "verification_useful", True),
                   source="calibrated")

    def to_dict(self):
        return asdict(self)


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
    wall_s: float = 0.0               # monotonic; agrees with the timeout
    wall_clock_s: float = 0.0         # time.time(); includes host suspend

    # scoring: one entry per seed, so promotion can require agreement
    seed_scores: dict = field(default_factory=dict)   # {seed: primary}
    metrics: dict | None = None                       # full metrics, seed 0
    is_buggy: bool = True                             # until proven otherwise
    failure_reason: str = ""
    created_at: str = ""

    # search policy, recorded at insert time and never recomputed: the
    # retirement test depends on an incumbent that moves, so a scope derived
    # after the fact would not be reproducible from the journal.
    scope: str = "exploit"            # exploit | explore
    protected_root: int | None = None

    # +1 when the task maximises, -1 when it minimises. seed_scores hold the
    # *oriented* value so every comparison below stays written in the maximise
    # direction; native_score converts back for display. Orienting once at the
    # boundary beats threading a sign through six comparison sites, one of
    # which would eventually be missed.
    sign: float = 1.0

    @property
    def score(self):
        """Mean validation primary across every seed this node was run on."""
        if not self.seed_scores:
            return None
        return statistics.fmean(self.seed_scores.values())

    @property
    def native_score(self):
        """The score in the task's own units and direction, for display."""
        sc = self.score
        return None if sc is None else self.sign * sc

    @property
    def seed_std(self):
        v = list(self.seed_scores.values())
        return statistics.stdev(v) if len(v) > 1 else None

    @property
    def n_seeds(self):
        return len(self.seed_scores)

    def one_line(self):
        s = (f"{self.native_score:.4f}"
             if self.native_score is not None else "  --  ")
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

    def __init__(self, path=None, policy=None):
        self.path = Path(path) if path else None
        self.policy = policy or PolicyConfig()
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

    def best_in(self, pool):
        """Best scored node within an arbitrary pool of nodes."""
        g = [n for n in pool if not n.is_buggy and n.score is not None]
        return max(g, key=lambda n: n.score) if g else None

    def best(self):
        return self.best_in(self.nodes)

    # ------------------------------------------------------------- lineages
    def roots(self):
        return [n for n in self.nodes if n.parent_id is None]

    def root_of(self, node_id):
        """The root this node descends from. Journals are small; walk it."""
        seen = set()
        node = self.get(node_id)
        while node is not None and node.parent_id is not None:
            if node.id in seen:                 # defensive: never loop
                break
            seen.add(node.id)
            node = self.get(node.parent_id)
        return node.id if node is not None else None

    def lineage(self, root_id):
        return [n for n in self.nodes if self.root_of(n.id) == root_id]

    def nodes_since_last_root(self):
        """How many attempts since a new root was opened.

        Keyed off the journal rather than the loop's iteration counter, which
        restarts at 1 on every resume - run final_01 was resumed three times,
        so its draft cadence restarted three times too.
        """
        for i, n in enumerate(reversed(self.nodes)):
            if n.parent_id is None:
                return i
        return len(self.nodes)

    # ------------------------------------------------- multi-seed verification
    def verified(self, min_seeds=2):
        """Scored nodes confirmed on more than one seed."""
        return [n for n in self.good() if n.n_seeds >= min_seeds]

    def best_verified_score(self, min_seeds=2):
        """The highest score actually confirmed across seeds, or None.

        The *max*, not the most recent. A later verification coming back lower
        than an earlier one must not quietly lower the bar a candidate has to
        clear.
        """
        v = self.verified(min_seeds)
        return max(n.score for n in v) if v else None

    def needs_verification(self, primary, margin=None, min_seeds=2):
        """Has this single-seed result earned the cost of more seeds?

        Anchored on the best verified score, so a run of individually tiny
        gains is measured against confirmed ground and gets checked once it
        has cumulatively drifted past the margin - which is precisely what
        final_02 did without ever being checked.
        """
        if margin is None:
            margin = self.policy.verify_margin
        anchor = self.best_verified_score(min_seeds)
        return True if anchor is None else primary > anchor + margin

    def best_score(self):
        b = self.best()
        return b.score if b else None

    def children(self, node_id):
        return [n for n in self.nodes if n.parent_id == node_id]

    def repair_attempts(self, node_id):
        """How many debug nodes have already been spent on this one."""
        return sum(1 for n in self.nodes
                   if n.parent_id == node_id and n.stage == "debug")

    def is_abandoned(self, node, max_attempts=None):
        """True once a broken node has exhausted its repair budget."""
        if max_attempts is None:
            max_attempts = self.policy.max_debug_attempts
        return node.is_buggy and self.repair_attempts(node.id) >= max_attempts

    def abandoned(self, max_attempts=None):
        return [n for n in self.nodes if self.is_abandoned(n, max_attempts)]

    # ----------------------------------------------------------- protection
    def protected_root(self, max_attempts=None):
        """The root of a young lineage still entitled to develop, or None.

        Only the newest root is ever a candidate. If it is already the best
        node's own lineage it needs no protection - it won - and if it has
        exhausted any of its budgets, or fallen too far behind, or broken
        beyond repair, protection is over and the search goes back to being
        globally greedy.
        """
        best = self.best()
        if best is None:
            return None
        roots = self.roots()
        if not roots:
            return None
        root = max(roots, key=lambda n: n.id)
        if root.id == self.root_of(best.id):
            return None                       # the explorer is the incumbent

        lin = self.lineage(root.id)
        if len(lin) >= self.policy.protection_max_nodes:
            return None
        scored = [n for n in lin if not n.is_buggy and n.score is not None]
        if not scored:
            return None                       # no measurement to protect yet
        if len(scored) >= self.policy.protected_scored_attempts:
            return None
        if max(n.score for n in scored) < best.score - self.policy.protection_write_off:
            return None                       # lost outright, not in the noise
        return root.id

    def protection_status(self, max_attempts=None):
        """(root_id, reason) describing why protection is or is not active.

        The loop logs this; a policy that cannot be read off the run log is
        not auditable, and the run log is a graded deliverable.
        """
        root_id = self.protected_root(max_attempts)
        if root_id is not None:
            return root_id, "active"
        best = self.best()
        roots = self.roots()
        if best is None or not roots:
            return None, "no incumbent"
        root = max(roots, key=lambda n: n.id)
        if root.id == self.root_of(best.id):
            return root.id, "won"
        lin = self.lineage(root.id)
        scored = [n for n in lin if not n.is_buggy and n.score is not None]
        if len(scored) >= PROTECTED_SCORED_ATTEMPTS:
            return root.id, "budget_exhausted"
        if len(lin) >= self.policy.protection_max_nodes:
            return root.id, "budget_exhausted"
        if scored:
            return root.id, "written_off"
        return root.id, "never_scored"

    # ------------------------------------------------------------ selection
    def select_parent(self, max_attempts=None):
        """Greedy - but over the *active lineage*, not the whole journal.

        The unit of greed is a lineage. Normally that is the best node's own
        lineage, which makes this identical to the global argmax it replaces.
        While a young lineage is still protected, it is that lineage instead,
        so a direction that lost inside the noise floor gets developed before
        it is judged against the incumbent.

        Within the chosen lineage the rule is unchanged: improve its best
        node, unless that node's most recent child broke and still has repair
        budget, in which case fix the child first. Once a broken node has had
        max_attempts it is abandoned - the escape hatch run final_01 lacked.
        """
        root = self.protected_root(max_attempts)
        pool = self.lineage(root) if root is not None else self.nodes
        anchor = self.best_in(pool)

        if anchor is None:
            return None                       # nothing has scored anywhere yet

        broken_kids = [c for c in self.children(anchor.id)
                       if c.is_buggy and not self.is_abandoned(c, max_attempts)]
        if broken_kids:
            return broken_kids[-1]
        return anchor

    # ---------------------------------------------------------- convergence
    def improvement_history(self, only_scope=None):
        """Best-so-far after each node, in order.

        `only_scope` filters which nodes get an *entry*, not which nodes count
        towards the running best - an exploration that wins still raises the
        best-so-far that later entries report. Convergence reads the exploit
        entries only, so the protected iterations (which by construction
        cannot move the global best) are not mistaken for a stalled search.
        """
        hist, best = [], None
        for n in self.nodes:
            if not n.is_buggy and n.score is not None:
                best = n.score if best is None else max(best, n.score)
            if only_scope is None or n.scope == only_scope:
                hist.append(best)
        return hist

    def has_converged(self, eps=None, n=None, min_scored=None):
        """The organizers' rule: N consecutive iterations gaining <= eps,
        applied only once the search has had min_scored attempts to get going.

        Two guards sit in front of it. An exploration still inside its budget
        blocks convergence outright - it is bounded by PROTECTION_MAX_NODES
        and the run's own caps, so this cannot keep a run alive indefinitely.
        And the floor counts exploit attempts only, so buying an exploration
        never brings the stopping rule closer.
        """
        eps = self.policy.eps if eps is None else eps
        n = self.policy.n_converge if n is None else n
        min_scored = (self.policy.min_scored_before_convergence
                      if min_scored is None else min_scored)
        if self.protected_root() is not None:
            return False
        if len([n_ for n_ in self.good() if n_.scope == "exploit"]) < min_scored:
            return False
        hist = [h for h in self.improvement_history(only_scope="exploit")
                if h is not None]
        if len(hist) < n + 1:
            return False
        window = hist[-(n + 1):]
        return all(window[i + 1] - window[i] <= eps for i in range(n))

    # -------------------------------------------------------------- context
    def lineage_rollup(self):
        """One line per lineage: what it tried and how far it got.

        summary() truncates each hypothesis to 88 characters, which is enough
        to tell attempts apart but not enough to tell *directions* apart. A
        draft is asked for something genuinely different, so it needs to see
        the directions already spent rather than the last 25 attempts.
        """
        if not self.roots():
            return "(no lineages yet)"
        rows = []
        for root in self.roots():
            lin = self.lineage(root.id)
            scored = [n for n in lin if not n.is_buggy and n.score is not None]
            best = max(scored, key=lambda n: n.score) if scored else None
            head = (root.hypothesis or "").splitlines()
            rows.append(
                f"root #{root.id}: {len(lin)} attempt(s), "
                + (f"best {best.score:.4f} (#{best.id})" if best
                   else "nothing scored")
                + f"\n    started from: {head[0][:160] if head else '(no hypothesis)'}"
                + (f"\n    best so far:  {best.hypothesis.splitlines()[0][:160]}"
                   if best and best.id != root.id and best.hypothesis else ""))
        return "\n".join(rows)

    def summary(self, limit=25):
        """Compact history for the model's prompt. Newest last."""
        if not self.nodes:
            return "(no attempts yet)"
        rows = [n.one_line() for n in self.nodes[-limit:]]
        head = f"id   stage   status  valid   hypothesis"
        return "\n".join([head, "-" * 100] + rows)
