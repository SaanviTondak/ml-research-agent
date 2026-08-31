"""Verify the LLM backend before the loop depends on it.

Checks the key loads, the endpoint is reachable, the model answers, usage
accounting is populated, and code extraction works on a real response.

    python3 check_llm.py [--model MODEL]
    python3 check_llm.py --list        # what this key can actually reach
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.llm import (LLM, GeminiBackend, TokenLedger, LLMError,
                       QuotaExhausted, extract_code, load_api_key,
                       DEFAULT_MODEL)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=DEFAULT_MODEL)
ap.add_argument("--list", action="store_true",
                help="list models this key can call, then exit")
a = ap.parse_args()

if a.list:
    import json, urllib.request
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": load_api_key()})
    d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    names = sorted(m["name"].replace("models/", "") for m in d.get("models", [])
                   if "generateContent" in m.get("supportedGenerationMethods", []))
    print(f"{len(names)} models support generateContent:\n")
    for n in names:
        print(f"  {n}")
    print("\nModel availability moves faster than this repo. If the default "
          "404s,\npick a current one here and update DEFAULT_MODEL in agent/llm.py.")
    raise SystemExit(0)

print(f"[1] api key")
try:
    key = load_api_key()
except LLMError as e:
    print(f"    FAILED\n{e}")
    raise SystemExit(1)
print(f"    found, {len(key)} chars, ends ...{key[-4:]}")

print(f"[2] reaching {a.model}")
llm = LLM(backend=GeminiBackend(model=a.model), ledger=TokenLedger())
try:
    r = llm.complete(
        system="You are terse. Reply with code only, in one fenced block.",
        user="Write a Python function `add(a, b)` returning a + b.",
        temperature=0.0, max_tokens=2000)
except QuotaExhausted as e:
    print(f"    QUOTA: {e}")
    raise SystemExit(2)
except LLMError as e:
    print(f"    FAILED: {e}")
    raise SystemExit(1)

print(f"    ok in {r.usage.wall_s:.1f}s, finish={r.finish_reason or 'STOP'}, "
      f"{r.usage.retries} retries")

print(f"[3] usage accounting")
u = r.usage
print(f"    {u.prompt_tokens} in / {u.completion_tokens} out "
      f"/ {u.total_tokens} total")
if u.total_tokens == 0:
    print("    WARNING: API reported no usage - the resource report would be blank")

print(f"[4] code extraction")
code = extract_code(r.text)
if not code:
    print(f"    FAILED - no code block found in:\n{r.text[:300]}")
    raise SystemExit(1)
print("    " + "\n    ".join(code.splitlines()[:4]))
ns = {}
exec(code, ns)
assert ns["add"](2, 3) == 5, "extracted code is wrong"
print("    extracted code runs and is correct")

print(f"\nledger: {llm.ledger.summary()}")
print(f"\nLLM backend is ready. Model: {a.model}")
