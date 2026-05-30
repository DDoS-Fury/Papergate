"""Verification harness for the streaming TGN serving path.

Run *after* ``graphagate.train_tgn`` has produced the artifacts in ``public/``:

    docker compose --profile training-tgn run --rm --entrypoint python \\
        train-tgn -m graphagate.verify_tgn

Checks, each independent of the trained metric value:
  1. Reload determinism — two independent loads score an event identically.
  2. Anti-poisoning gate — an event classified anomalous never advances memory;
     a benign one does (forced deterministically via the decision threshold).
  3. Dynamic node — a never-before-seen entity key is admitted without error and
     grows the registry by exactly one slot.
"""

from __future__ import annotations

import math
import sys

import torch

from graphagate.config import TGN_CHECKPOINT_PATH, TGN_STATS_PATH
from graphagate.serve_tgn import load_model, score_event

KEY_SRC = 60          # a preregistered IP entity
KEY_DST = 155         # a preregistered resource entity
TS = 10**9            # a timestamp far beyond any training time
BENIGN_FEAT = [1.0, 0.0, 0.0, 0.0, 0.0, 2.0]


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load():
        # load_model returns (model, registry, threshold, hp); the harness ignores hp.
        model, registry, threshold, _ = load_model(
            TGN_CHECKPOINT_PATH, TGN_STATS_PATH, device
        )
        return model, registry, threshold

    results = []

    # 1. Reload determinism.
    m1, r1, thr = load()
    m2, r2, _ = load()
    s1, _ = score_event(m1, r1, thr, KEY_SRC, KEY_DST, TS, BENIGN_FEAT, device, update=False)
    s2, _ = score_event(m2, r2, thr, KEY_SRC, KEY_DST, TS, BENIGN_FEAT, device, update=False)
    results.append(_check("reload determinism", abs(s1 - s2) < 1e-6, f"s1={s1:.6f} s2={s2:.6f}"))

    # 2a. Benign event (threshold forced high -> not anomaly -> memory advances).
    m, r, _ = load()
    idx = r.get(KEY_SRC)
    _, is_anom = score_event(m, r, 2.0, KEY_SRC, KEY_DST, TS, BENIGN_FEAT, device, update=True)
    lu = int(m.memory.last_update[idx])
    results.append(_check("benign event updates memory", (not is_anom) and lu == TS,
                          f"is_anom={is_anom} last_update={lu}"))

    # 2b. Anomalous event (threshold forced low -> anomaly -> memory frozen).
    m, r, _ = load()
    idx = r.get(KEY_SRC)
    lu0 = int(m.memory.last_update[idx])
    _, is_anom = score_event(m, r, -1.0, KEY_SRC, KEY_DST, TS, BENIGN_FEAT, device, update=True)
    lu1 = int(m.memory.last_update[idx])
    results.append(_check("anomaly does NOT poison memory", is_anom and lu1 == lu0,
                          f"is_anom={is_anom} before={lu0} after={lu1}"))

    # 3. Dynamic node admission.
    m, r, thr = load()
    n_before = len(r)
    new_key = 10**6
    seen_before = r.get(new_key)
    score, _ = score_event(m, r, thr, new_key, KEY_DST, TS, BENIGN_FEAT, device, update=False)
    new_idx = r.get(new_key)
    ok = (
        seen_before is None
        and math.isfinite(score)
        and len(r) == n_before + 1
        and new_idx is not None
        and new_idx < m.memory.memory.shape[0]
    )
    results.append(_check("dynamic node admitted", ok,
                          f"new_idx={new_idx} score={score:.4f} registry {n_before}->{len(r)}"))

    passed = all(results)
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'} "
          f"({sum(results)}/{len(results)})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
