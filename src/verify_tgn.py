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
  4. Source fallback — scoring without ``key_source`` (no client IP available)
     works and skips the source→device edge.
"""

from __future__ import annotations

import math
import sys

import torch

from graphagate.config import TGN_CHECKPOINT_PATH, TGN_STATS_PATH
from graphagate.serve_tgn import load_model, score_event

KEY_USER = 0          # a preregistered user entity
KEY_DEVICE = "ck:verify-device"                # admitted dynamically on first use
KEY_SOURCE = "src:10.99.99.1"                  # namespaced source key, admitted on first use
KEY_DST = "/api/v1/reactor-parameters"         # a preregistered resource entity URI
TS = 10**9            # a timestamp far beyond any training time
BENIGN_FEAT = [1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load():
        # load_model returns (model, registry, threshold, threshold_dirty, hp); the
        # harness exercises the single-threshold path, so it ignores the latter two.
        model, registry, threshold, _, _ = load_model(
            TGN_CHECKPOINT_PATH, TGN_STATS_PATH, device
        )
        return model, registry, threshold

    results = []

    def score(m, r, thr, *, key_source=KEY_SOURCE, update=False):
        return score_event(
            m, r, thr, KEY_USER, KEY_DEVICE, KEY_DST, TS, BENIGN_FEAT, device,
            key_source=key_source, update=update,
        )

    # 1. Reload determinism.
    m1, r1, thr = load()
    m2, r2, _ = load()
    s1, _ = score(m1, r1, thr)
    s2, _ = score(m2, r2, thr)
    results.append(_check("reload determinism", abs(s1 - s2) < 1e-6, f"s1={s1:.6f} s2={s2:.6f}"))

    # 2a. Benign event (threshold forced high -> not anomaly -> memory advances).
    m, r, _ = load()
    _, is_anom = score(m, r, 2.0, update=True)
    idx = r.get(KEY_USER)
    lu = int(m.memory.last_update[idx])
    results.append(_check("benign event updates memory", (not is_anom) and lu == TS,
                          f"is_anom={is_anom} last_update={lu}"))

    # 2b. Anomalous event (threshold forced low -> anomaly -> memory frozen).
    m, r, _ = load()
    idx = r.get(KEY_USER)
    lu0 = int(m.memory.last_update[idx])
    _, is_anom = score(m, r, -1.0, update=True)
    lu1 = int(m.memory.last_update[idx])
    results.append(_check("anomaly does NOT poison memory", is_anom and lu1 == lu0,
                          f"is_anom={is_anom} before={lu0} after={lu1}"))

    # 3. Dynamic node admission (one fresh device key -> exactly one new slot).
    m, r, thr = load()
    new_key = "ck:never-seen-before"
    seen_before = r.get(new_key)
    n_before = len(r)
    s, _ = score_event(m, r, thr, KEY_USER, new_key, KEY_DST, TS, BENIGN_FEAT, device,
                       update=False)
    new_idx = r.get(new_key)
    ok = (
        seen_before is None
        and math.isfinite(s)
        and len(r) == n_before + 1
        and new_idx is not None
        and new_idx < m.memory.memory.shape[0]
    )
    results.append(_check("dynamic node admitted", ok,
                          f"new_idx={new_idx} score={s:.4f} registry {n_before}->{len(r)}"))

    # 4. Source fallback: no client IP -> the source→device edge is skipped.
    m, r, thr = load()
    n_before = len(r)
    s, _ = score(m, r, thr, key_source=None)
    no_src = r.get(KEY_SOURCE) is None
    results.append(_check("key_source=None fallback", math.isfinite(s) and no_src,
                          f"score={s:.4f} source_admitted={not no_src} "
                          f"registry {n_before}->{len(r)}"))

    passed = all(results)
    print(f"\n{'ALL CHECKS PASSED' if passed else 'SOME CHECKS FAILED'} "
          f"({sum(results)}/{len(results)})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
