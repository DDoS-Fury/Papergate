"""Stateful set-membership rules over a :class:`SyntheticStream` — the no-learning baseline.

Each rule is one dict lookup per event ("has this config ever been seen with this
device?"). They are what a SIEM correlation rule does, and what a reviewer will run first
against a synthetic benchmark: if one of them separates a class, the class measures the
generator, not the model. ``tests/test_leakage_audit.py`` bounds them; the learned models
must beat their combination.

State-commit ``gate`` (the rule's memory, cf. the TGN's benign-gated memory):
  ``label``      commit ground-truth benign events only, test window included (optimistic)
  ``all``        commit every event — uses no labels at all (pessimistic)
  ``proto-self`` the paper protocol: ground-truth benign commits before ``test_start``
                 (train/val are labelled benign for the TGN too), then only events no
                 binding rule fired on (predicted-benign gate)
  ``proto-all``  as ``proto-self``, but commit every test event
"""

from __future__ import annotations

import numpy as np

# Single-lookup flags. ``x|y_new`` = the (x, y) pair was never committed before.
FLAGS = ("cfg_new", "src_new", "dev_new", "cfg|dev_new", "cfg|usr_new", "dev|usr_new",
         "src|usr_new", "role_changed", "sensor")
# History-dependent binding flags (nothing the static single-column audit can see).
BINDING = ("cfg|dev_new", "cfg|usr_new", "dev|usr_new", "src|usr_new", "role_changed")
GATES = ("label", "all", "proto-self", "proto-all")


def lookup_flags(s, gate: str = "proto-self", test_start: int = 0) -> dict[str, np.ndarray]:
    """Per-event boolean flags for every rule in :data:`FLAGS`, plus two scores:
    ``stateful`` (sum of :data:`BINDING`) and ``combined`` (``stateful`` + ``sensor``)."""
    y = s.y.numpy()
    msg = s.msg.numpy()
    cf, src, dev, usr = (x.numpy() for x in (s.config, s.source, s.device, s.user))
    n = len(y)
    out = {k: np.zeros(n, dtype=bool) for k in FLAGS}
    cols = [out[k] for k in FLAGS]
    seen = [set() for _ in range(7)]  # cfg, src, dev, cfg|dev, cfg|usr, dev|usr, src|usr
    first_role: dict[int, float] = {}
    for i in range(n):
        c, sr, d, u, role = cf[i], src[i], dev[i], usr[i], msg[i, 5]
        keys = (c, sr, d, (c, d), (c, u), (d, u), (sr, u))
        f = [k not in st for k, st in zip(keys, seen)]
        f.append(u in first_role and first_role[u] != role)
        f.append(bool(msg[i, 0] == 0 or msg[i, 1:4].any()))
        for col, v in zip(cols, f):
            col[i] = v
        if gate == "label" or (gate.startswith("proto") and i < test_start):
            commit = y[i] == 0
        elif gate == "proto-self":
            commit = not any(f[3:8])  # no binding / role rule fired
        else:
            commit = True
        if commit:
            for k, st in zip(keys, seen):
                st.add(k)
            first_role.setdefault(u, role)
    out["stateful"] = sum(out[k].astype(float) for k in BINDING)
    out["combined"] = out["stateful"] + out["sensor"]
    return out
