"""A static check on agent-written code, before it is ever executed.

The firewall makes the test labels physically unreachable through the data
directory the agent is given. This closes the remaining path: nothing stops a
model from writing a hard-coded absolute path to the original dataset, which is
still on disk. It has no reason to - but "no reason to" is not a guarantee, and
one such candidate silently invalidates the entire run.

Scope, learned the hard way
---------------------------
The first version of this file also rejected any mention of the 'test' split.
That was wrong, and it cost a live iteration: a candidate was rejected for the
line

    "test": (20220429, 20220508),

which is copied verbatim from the organizer's own data.py. The guard would have
rejected the reference implementation.

The reasoning error was treating a *name* as a *capability*. Against the visible
directory there are no test rows, so `splits['test']` is an empty list and every
mention of the split is inert. The only way to reach real test data is an
absolute path to the sealed directory - so that, and only that, is a rejection.

Mentions of the test split are still reported, because they are worth seeing in
the run log, but they are warnings rather than rejections. A false rejection is
not free: it burns an iteration and teaches the agent to avoid a construct that
was never dangerous.

What still trips it, deliberately
---------------------------------
The organizer's `baseline.py` and `submit.py` are rejected, because both carry
`--data_dir` defaults of './KuaiRand-Pure/data'. That is intended. Neither file
is shown to the agent: the template it is given is `candidates/fm_baseline.py`,
which makes `--data_dir` required with no default and passes cleanly. Any
candidate reaching for the dataset by name rather than taking the directory it
was handed is doing something it was told not to, and one iteration is a cheap
price for catching it.
"""
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.paths import REAL_DATA, STARTER

# Hard rejections: the only constructs that can actually reach sealed data.
FORBIDDEN_PATTERNS = [
    (re.compile(re.escape(str(REAL_DATA))),
     "hard-codes the sealed data directory"),
    (re.compile(re.escape(str(STARTER / "KuaiRand-Pure"))),
     "hard-codes the original dataset directory"),
    (re.compile(r"""KuaiRand-Pure[/\\]data"""),
     "references the original dataset path"),
]

# Reported, not rejected. Inert against the visible directory, but worth
# surfacing in the run log so an unexpected pattern is visible to a reader.
SUSPICIOUS_PATTERNS = [
    (re.compile(r"""split\s*=\s*['"]test['"]"""), "selects split='test'"),
    (re.compile(r"""--split['"]?\s*,\s*['"]test['"]"""), "passes --split test"),
]


class GuardRejection(Exception):
    """Generated code could reach data outside the agent's visible directory."""


def _scan(code, patterns):
    findings = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        stripped = line.split("#")[0]
        for pat, why in patterns:
            if pat.search(stripped):
                findings.append((why, lineno, line.strip()[:120]))
    return findings


def check_code(code):
    """Return (rejections, warnings). Empty rejections = safe to execute."""
    return _scan(code, FORBIDDEN_PATTERNS), _scan(code, SUSPICIOUS_PATTERNS)


def assert_clean(code):
    """Raise GuardRejection if the code can reach sealed data. Return warnings."""
    rejections, warnings = check_code(code)
    if rejections:
        detail = "\n".join(f"  line {ln}: {why}\n      {src}"
                           for why, ln, src in rejections)
        raise GuardRejection(
            "The candidate was rejected before execution because it hard-codes "
            "a path to data outside your working directory.\n" + detail +
            "\n\nRead data ONLY from the --data_dir argument that is passed to "
            "you. Never construct an absolute path to the dataset.")
    return warnings
