"""Cross-process determinism test for ``graphagate.model.tgn.stable_hash``.

The hashed-identity trick is only inductive/reproducible if the SAME entity key maps
to the SAME bucket across processes, machines and restarts. Python's builtin ``hash()``
is salted per process (``PYTHONHASHSEED``), so the previous ``hash(str(key)) % buckets``
broke this. ``stable_hash`` uses BLAKE2b and must be invariant to ``PYTHONHASHSEED``.

This script proves it by computing ``stable_hash`` (and, for contrast, the builtin
``hash``) for the same keys in two child processes launched with different
``PYTHONHASHSEED`` values, then asserting:
  * ``stable_hash`` is identical across the two processes (the fix), and
  * the builtin ``hash`` differs (the bug it replaces) — informational, not fatal.

No pytest dependency (not in the image): run directly inside the Docker image::

    docker compose run --rm --no-deps --entrypoint python train-tgn \
        /app/tests/test_stable_hash.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

KEYS = ["user:42", "10.0.0.7", "/api/v1/nuclear-materials", 60, 1000000]
BUCKETS = 10000

# Snippet run in each child: emit {stable: [...], builtin: [...]} for KEYS.
_CHILD = (
    "import json;"
    "from graphagate.model.tgn import stable_hash;"
    f"keys={KEYS!r};"
    f"b={BUCKETS};"
    "print(json.dumps({"
    "'stable':[stable_hash(k,b) for k in keys],"
    "'builtin':[hash(str(k))%b for k in keys]}))"
)


def _run(seed: str) -> dict:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    out = subprocess.check_output([sys.executable, "-c", _CHILD], env=env, text=True)
    return json.loads(out.strip().splitlines()[-1])


def main() -> int:
    a = _run("1")
    b = _run("2")

    ok = True

    if a["stable"] == b["stable"]:
        print(f"[PASS] stable_hash invariant across PYTHONHASHSEED: {a['stable']}")
    else:
        print(f"[FAIL] stable_hash differs across processes: {a['stable']} != {b['stable']}")
        ok = False

    if all(0 <= h < BUCKETS for h in a["stable"]):
        print(f"[PASS] stable_hash within [0, {BUCKETS})")
    else:
        print(f"[FAIL] stable_hash out of range: {a['stable']}")
        ok = False

    # Informational: the builtin hash it replaces is NOT stable across PYTHONHASHSEED
    # (string keys are salted; small ints happen to be stable — hence the regression
    # showed up mainly for string entity keys such as resource URIs).
    if a["builtin"] != b["builtin"]:
        print(f"[INFO] builtin hash() differs as expected (the bug fixed): "
              f"{a['builtin']} != {b['builtin']}")
    else:
        print(f"[INFO] builtin hash() matched here (key-dependent); not relied upon.")

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
