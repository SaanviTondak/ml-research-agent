# Tests

```bash
python3 -m pytest tests/ -v
```

No API key and no dataset required — these run in CI on every push.

| file | covers |
|---|---|
| `test_search_policy.py` | the greedy tree search, and the repair-budget fix for the failure that cost run `final_01` 68% of its iterations |
| `test_guard.py` | the pre-execution checks: syntax gate, and the data firewall's rejection rules |

`test_search_policy.py` replays the real `final_01` tree up to node 10 — the
moment the trap closed — and asserts the search can now leave it. See
[`docs/postmortem.md`](../docs/postmortem.md).

## What these tests do not cover

The end-to-end path — executor, submission contract, scorer, journal, and
reproducing the published baseline — needs the dataset, so it lives in
[`harness_check.py`](../harness_check.py) at the repo root and is run locally:

```bash
python3 harness_check.py    # 9 checks, ~45 s, no API key
```
