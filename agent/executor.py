"""Step 2a - run a candidate script without letting it kill the loop.

Everything the agent writes is untrusted: it may not compile, may raise, may
hang, may spawn children, may print a hundred megabytes. The loop has to
survive all of that unattended, because every crash that reaches the human
costs autonomy points.

Guarantees:
  * a hard wall-clock timeout, enforced by killing the whole process group
    (a bare Popen.kill leaves grandchildren running and the pipe open);
  * stdout/stderr always captured, never inherited;
  * output clipped head+tail so one runaway loop cannot blow up the journal
    or, later, the LLM context window;
  * the exception type and traceback tail extracted, so the debug step has
    something concrete to react to;
  * PYTHONPATH pre-seeded with the starter kit, so candidates can
    `from data import load, encode` and `from evaluate import evaluate`
    exactly as the organizer's README documents.
"""
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.paths import STARTER, REPO

DEFAULT_TIMEOUT_S = 900          # 15 min; a baseline FM run is ~32 s
CLIP_HEAD = 4000
CLIP_TAIL = 4000


def clip(text, head=CLIP_HEAD, tail=CLIP_TAIL):
    """Keep the start and the end; the middle of a runaway log is never useful."""
    if len(text) <= head + tail:
        return text
    omitted = len(text) - head - tail
    return (text[:head]
            + f"\n\n... [{omitted:,d} characters omitted] ...\n\n"
            + text[-tail:])


def _exception_from_traceback(stderr):
    """Pull `ExceptionType: message` off the tail of a Python traceback."""
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    if not lines:
        return None, None
    last = lines[-1]
    if ":" in last:
        head, _, msg = last.partition(":")
        if head.strip().replace(".", "").isidentifier():
            return head.strip(), msg.strip()
    if last.strip().replace(".", "").isidentifier():
        return last.strip(), ""
    return None, last


@dataclass
class ExecResult:
    ok: bool                       # exited 0, in time
    returncode: int | None
    timed_out: bool
    wall_s: float
    stdout: str
    stderr: str
    exc_type: str | None = None
    exc_msg: str | None = None
    argv: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def summary(self):
        if self.timed_out:
            return f"TIMEOUT after {self.wall_s:.1f}s"
        if self.ok:
            return f"ok in {self.wall_s:.1f}s"
        if self.exc_type:
            return f"{self.exc_type}: {self.exc_msg} (rc={self.returncode})"
        return f"exit {self.returncode} in {self.wall_s:.1f}s"


def run_script(script, args=(), timeout_s=DEFAULT_TIMEOUT_S, cwd=None, env=None):
    """Run `python3 script *args` under a hard timeout. Never raises."""
    argv = [sys.executable, "-u", str(script), *map(str, args)]

    child_env = dict(os.environ if env is None else env)
    existing = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (
        f"{STARTER}{os.pathsep}{existing}" if existing else str(STARTER))

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd or REPO),
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,       # own process group, so we can killpg
        )
    except OSError as e:
        return ExecResult(ok=False, returncode=None, timed_out=False,
                          wall_s=time.time() - t0, stdout="",
                          stderr=f"failed to launch: {e}",
                          exc_type=type(e).__name__, exc_msg=str(e),
                          argv=argv)

    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        try:
            out, err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        err = (err or "") + f"\n[executor] killed after {timeout_s}s timeout\n"

    wall = time.time() - t0
    rc = proc.returncode
    exc_type, exc_msg = (None, None) if timed_out else _exception_from_traceback(err or "")

    return ExecResult(
        ok=(rc == 0 and not timed_out),
        returncode=rc,
        timed_out=timed_out,
        wall_s=wall,
        stdout=clip(out or ""),
        stderr=clip(err or ""),
        exc_type=exc_type,
        exc_msg=exc_msg,
        argv=argv,
    )


def _kill_group(proc):
    """SIGTERM the process group, then SIGKILL anything still standing."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        for _ in range(20):
            if proc.poll() is not None:
                return
            time.sleep(0.1)
