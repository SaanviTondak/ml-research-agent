"""A static check on agent-written code, before it is ever executed.

The firewall makes the test labels physically unreachable through the data
directory the agent is given. This closes the remaining path: nothing stops a
model from writing a hard-coded absolute path to the original dataset, which
is still on disk. It has no reason to - but "no reason to" is not a guarantee,
and one such candidate silently invalidates the entire run.

So generated code is scanned before execution and rejected if it references
the sealed directory or tries to evaluate on test. A rejection is fed back to
the model as an ordinary failure, which it can then correct - it costs one
iteration, not the run.

This is a backstop, not the primary control. The firewall is the primary
control. Both are cheap; a leaked test score is not recoverable.
"""
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.paths import REAL_DATA

# Filenames that exist only in the sealed directory, plus the directory itself.
FORBIDDEN_PATTERNS = [
    (re.compile(re.escape(str(REAL_DATA))), "hard-codes the sealed data directory"),
    (re.compile(r"KuaiRand-Pure[/\\]data"), "references the original dataset path"),
    (re.compile(r"""\[\s*['"]test['"]\s*\]"""), "indexes the 'test' split"),
    (re.compile(r"""split\s*=\s*['"]test['"]"""), "selects split='test'"),
    (re.compile(r"""['"]test['"]\s*:"""), "builds a dict keyed on 'test'"),
]


class GuardRejection(Exception):
    """Generated code touches something it must not."""


def check_code(code):
    """Return a list of (pattern description, offending line). Empty = clean."""
    findings = []
    for lineno, line in enumerate(code.splitlines(), start=1):
        stripped = line.split("#")[0]
        for pat, why in FORBIDDEN_PATTERNS:
            if pat.search(stripped):
                findings.append((why, lineno, line.strip()[:120]))
    return findings


def assert_clean(code):
    findings = check_code(code)
    if findings:
        detail = "\n".join(f"  line {ln}: {why}\n      {src}"
                           for why, ln, src in findings)
        raise GuardRejection(
            "The candidate was rejected before execution because it tries to "
            "reach the sealed test data.\n" + detail +
            "\n\nRead data ONLY from the --data_dir argument, and evaluate "
            "ONLY on the 'valid' split. The test split does not exist in that "
            "directory and must never be referenced.")
    return True
