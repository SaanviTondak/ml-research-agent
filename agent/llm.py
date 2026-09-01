"""Step 3a - the LLM client, with the two things the loop actually needs.

Provider-agnostic on purpose. The loop talks to `LLM`; swapping Gemini for
Anthropic or anything else means adding one Backend subclass and changing one
config line, not touching the loop.

Two requirements drive the design:

1. A rate-limit response must never end the run. The free Gemini tier is
   generous but finite, and an unhandled 429 five hours into an unattended run
   is a manual intervention - which is 20% of the grade. Retries are automatic,
   with exponential backoff that honours the server's own suggested delay, and
   a hard daily-quota failure is raised as a distinct, catchable exception so
   the loop can park cleanly instead of crashing.

2. Every token is counted. "Feasibility" is graded on LLM spend, so the ledger
   is persisted alongside the run rather than estimated afterwards. Gemini
   reports exact usage in `usageMetadata`; we record what the API says, not
   what we guessed.

Deliberately stdlib-only (urllib, no SDK, no requests): the project's
dependency footprint stays numpy-only, and there is no install step to fail
on a fresh machine.
"""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.paths import REPO

DEFAULT_MODEL = "gemini-3.6-flash"
# Tried in order when the primary is saturated (503). Model
# availability moves faster than the code; `python3 check_llm.py
# --list` re-derives this from the API.
FALLBACK_MODELS = ("gemini-3.5-flash", "gemini-3-flash-preview",
                   "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
                   "gemini-flash-latest", "gemini-flash-lite-latest")
GEMINI_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/"
                   "models/{model}:generateContent")


class LLMError(Exception):
    """A request failed in a way retrying will not fix."""


class QuotaExhausted(LLMError):
    """Rate/quota limit that survived every retry - the loop should park."""


# --------------------------------------------------------------- credentials
def load_api_key(env_names=("GEMINI_API_KEY", "GOOGLE_API_KEY")):
    """Environment first, then a gitignored .env at the repo root."""
    for name in env_names:
        if os.environ.get(name):
            return os.environ[name].strip()

    dotenv = REPO / ".env"
    if dotenv.exists():
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() in env_names:
                return v.strip().strip("'\"")

    raise LLMError(
        f"No API key found. Set one of {', '.join(env_names)} in the "
        f"environment, or put it in {REPO/'.env'} as:\n"
        f"    GEMINI_API_KEY=your-key-here\n"
        f"(.env is gitignored.)")


# ------------------------------------------------------------------- ledger
@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    retries: int = 0
    wall_s: float = 0.0

    def add(self, other):
        for f in ("prompt_tokens", "completion_tokens", "total_tokens",
                  "calls", "retries", "wall_s"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        return self

    def to_dict(self):
        return asdict(self)


class TokenLedger:
    """Cumulative token accounting, flushed to disk after every call.

    This is a graded artifact, not telemetry: it is the evidence behind the
    resource report. Written eagerly so a killed run still has honest numbers.
    """

    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.total = Usage()
        self.by_model = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self):
        d = json.loads(self.path.read_text())
        self.total = Usage(**d.get("total", {}))
        self.by_model = {k: Usage(**v) for k, v in d.get("by_model", {}).items()}

    def record(self, model, usage):
        self.total.add(usage)
        self.by_model.setdefault(model, Usage()).add(usage)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(
                {"total": self.total.to_dict(),
                 "by_model": {k: v.to_dict() for k, v in self.by_model.items()}},
                indent=2) + "\n")

    def summary(self):
        t = self.total
        return (f"{t.calls} calls, {t.total_tokens:,d} tokens "
                f"({t.prompt_tokens:,d} in / {t.completion_tokens:,d} out), "
                f"{t.retries} retries, {t.wall_s:.0f}s")


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: Usage
    finish_reason: str = ""
    raw: dict = field(default_factory=dict, repr=False)


# ------------------------------------------------------------------ backend
class Backend:
    name = "abstract"

    def complete(self, system, user, **kw):
        raise NotImplementedError


class GeminiBackend(Backend):
    name = "gemini"

    def __init__(self, api_key=None, model=DEFAULT_MODEL, timeout_s=180):
        self.api_key = api_key or load_api_key()
        self.model = model
        self.timeout_s = timeout_s

    def _post(self, payload):
        url = GEMINI_ENDPOINT.format(model=self.model)
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": self.api_key},   # header, never the URL
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode())

    def complete(self, system, user, temperature=0.7, max_tokens=32768):
        payload = {
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        data = self._post(payload)

        cands = data.get("candidates") or []
        if not cands:
            fb = data.get("promptFeedback", {})
            raise LLMError(f"no candidates returned; promptFeedback={fb}")
        cand = cands[0]
        finish = cand.get("finishReason", "")
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts)

        um = data.get("usageMetadata", {}) or {}
        usage = Usage(
            prompt_tokens=um.get("promptTokenCount", 0),
            # thinking tokens are billed; count them or the report understates
            completion_tokens=(um.get("candidatesTokenCount", 0)
                               + um.get("thoughtsTokenCount", 0)),
            total_tokens=um.get("totalTokenCount", 0),
            calls=1)
        return LLMResponse(text=text, model=self.model, usage=usage,
                           finish_reason=finish, raw=data)


# ---------------------------------------------------------------- retry glue
_RETRY_DELAY_RE = re.compile(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _suggested_delay(body):
    """Gemini returns RetryInfo on a 429; obeying it beats guessing."""
    m = _RETRY_DELAY_RE.search(body or "")
    return float(m.group(1)) if m else None


def _is_daily_quota(body):
    b = (body or "").lower()
    return "perday" in b or "per day" in b


class LLM:
    """Retrying, model-failing-over wrapper. `complete()` returns text or raises.

    Two layers of resilience, because they fail differently:

      * within a model - transient 429/5xx are retried with backoff, obeying
        the server's own suggested delay when it offers one;
      * across models - if the primary stays saturated after its full retry
        budget, the next model in the chain takes over for the rest of the run.

    The second layer exists because model availability is not under our
    control and does not recover on our schedule. `gemini-3.7-flash` returned
    503 on every attempt while three others answered in under ten seconds. An
    unattended run that dies for that reason costs an autonomy point for a
    reason that has nothing to do with the research.

    Model switches are logged, and the ledger keys usage by model, so the
    resource report says exactly which model produced which work.
    """

    def __init__(self, backend=None, ledger=None, max_retries=6,
                 base_delay=2.0, max_delay=120.0, journal=None,
                 fallback_models=FALLBACK_MODELS):
        self.backend = backend or GeminiBackend()
        self.ledger = ledger or TokenLedger()
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.journal = journal
        self.fallbacks = list(fallback_models or ())

    @property
    def model(self):
        return getattr(self.backend, "model", self.backend.name)

    def _log(self, event, **kw):
        if self.journal:
            self.journal.append(event, **kw)

    def complete(self, system, user, **kw):
        """Try the current model, then each fallback, before giving up."""
        tried = []
        while True:
            try:
                return self._complete_one_model(system, user, **kw)
            except QuotaExhausted as e:
                # Free-tier quota is metered PER MODEL, so an exhausted daily
                # allowance on one model says nothing about the next. The first
                # version parked the run here; it cost a live run after nine
                # minutes while six other models still had quota. Fall over
                # like any other failure, and park only when every model is
                # spent - which is the only state waiting could fix.
                tried.append(self.model)
                if not self.fallbacks:
                    raise QuotaExhausted(
                        f"quota exhausted on every model "
                        f"({', '.join(tried)}): {e}") from e
                nxt = self.fallbacks.pop(0)
                self._log("llm_model_switch", status="error",
                          note=f"{self.model} out of quota, switching to {nxt}",
                          error=str(e)[:300])
                self.backend.model = nxt
                continue
            except LLMError as e:
                tried.append(self.model)
                if not self.fallbacks:
                    raise LLMError(
                        f"all models exhausted ({', '.join(tried)}): {e}") from e
                nxt = self.fallbacks.pop(0)
                self._log("llm_model_switch", status="error",
                          note=f"{self.model} unavailable, switching to {nxt}",
                          error=str(e)[:300])
                self.backend.model = nxt

    def _complete_one_model(self, system, user, **kw):
        t0 = time.time()
        retries = 0
        last = None

        for attempt in range(self.max_retries + 1):
            try:
                resp = self.backend.complete(system, user, **kw)
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                last = f"HTTP {e.code}: {body[:400]}"
                if e.code == 429 and _is_daily_quota(body):
                    raise QuotaExhausted(
                        f"daily quota exhausted for {self.model}. The run "
                        f"should park and resume tomorrow, or switch model.\n"
                        f"{body[:400]}")
                if e.code not in RETRYABLE_STATUS or attempt == self.max_retries:
                    raise LLMError(last) from e
                delay = _suggested_delay(body) or self._backoff(attempt)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = f"{type(e).__name__}: {e}"
                if attempt == self.max_retries:
                    raise LLMError(last) from e
                delay = self._backoff(attempt)
            else:
                resp.usage.retries = retries
                resp.usage.wall_s = time.time() - t0
                self.ledger.record(resp.model, resp.usage)
                self._log("llm_call", model=resp.model, status="ok",
                          retries=retries,
                          tokens=resp.usage.total_tokens,
                          wall_s=round(resp.usage.wall_s, 1),
                          finish_reason=resp.finish_reason)
                return resp

            retries += 1
            delay = min(delay, self.max_delay)
            self._log("llm_retry", model=self.model, status="error",
                      attempt=attempt + 1, sleep_s=round(delay, 1), error=last)
            time.sleep(delay)

        raise LLMError(last or "exhausted retries")

    def _backoff(self, attempt):
        """Exponential with jitter, so parallel retries do not resonate."""
        return self.base_delay * (2 ** attempt) * (0.5 + random.random())


# ------------------------------------------------------------------ parsing
_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*)\Z", re.DOTALL)


def extract_code(text):
    """Pull the python from a fenced block, tolerating chatty models.

    Returns the LONGEST closed fenced block: models that narrate before the
    code often emit a short illustrative snippet first, and taking the first
    block silently feeds a broken candidate into the loop.

    Falls back to an unclosed fence, which is what a response truncated at the
    output-token limit looks like. The result is incomplete and will not run,
    but handing the partial script to the debug step lets the next iteration
    finish it, instead of discarding the work entirely.
    """
    blocks = _FENCE_RE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    m = _OPEN_FENCE_RE.search(text or "")
    if m and m.group(1).strip():
        return m.group(1).strip()
    if text and text.lstrip().startswith(("import ", "from ", '"""', "#")):
        return text.strip()      # model skipped the fence entirely
    return None
