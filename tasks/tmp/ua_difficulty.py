"""Cross-dataset difficulty probe: the 'Unknown Authentication' rule.

Euler (NDSS'22, Table VI) reports that on LANL the rule 'flag any edge not seen in the
training data' alone reaches TPR 72.0 at FPR 4.4. That is the single number which says how
much of LANL lateral movement is plain graph memorisation. Running the SAME rule on our
stream is the only apples-to-apples statement we can make about relative difficulty, since
the datasets, units and prevalences all differ.

The rule is causal and parameter-free: an event is flagged if that pair has never been seen
before in the stream up to that point.

Run:
  docker compose run --rm --no-deps -T -v "$PWD/tasks:/app/tasks" \
      --entrypoint python train-tgn /app/tasks/tmp/ua_difficulty.py
"""
import dataclasses

import numpy as np

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
n = len(s.types)
va = int(n * (cfg.train_frac + cfg.val_frac))
ty, lab = s.types.numpy(), (s.types.numpy() != 0).astype(int)
user, dst, dev = s.user.numpy(), s.dst.numpy(), s.device.numpy()

def never_seen(a, b, frozen: bool):
    """'This pair is new' flag.

    ``frozen=True`` reproduces Euler's variant exactly: the reference graph is the TRAINING
    slice and nothing is added afterwards, so a test event is flagged iff its pair is absent
    from the training data. ``frozen=False`` is the streaming variant, whose memory keeps
    growing — more forgiving, and the one our own serving path would use.
    """
    out = np.zeros(n, dtype=bool)
    seen = {(int(a[i]), int(b[i])) for i in range(va)}
    for i in range(n):
        k = (int(a[i]), int(b[i]))
        out[i] = k not in seen
        if not frozen:
            seen.add(k)
    return out

def build(frozen):
    d = {"user->risorsa": never_seen(user, dst, frozen),
         "device->user": never_seen(dev, user, frozen),
         "device->risorsa": never_seen(dev, dst, frozen)}
    d["OR delle tre"] = d["user->risorsa"] | d["device->user"] | d["device->risorsa"]
    return d

t_ty, t_lab = ty[va:], lab[va:]
ben, lat = t_lab == 0, t_ty == 3
print(f"finestra test: {len(t_ty)} eventi | benigni={ben.sum()} laterali={lat.sum()}")
for frozen in (True, False):
    tag = "grafo di training CONGELATO (variante di Euler)" if frozen else "memoria che cresce (streaming)"
    print(f"\nRegola 'Unknown Authentication' -- {tag}")
    print(f"{'coppia':18s} {'TPR laterale':>13s} {'FPR benigno':>12s} {'lift':>7s} {'precision':>10s}")
    print("-" * 66)
    for k, m in build(frozen).items():
        mt = m[va:]
        tp, fp = int((mt & lat).sum()), int((mt & ben).sum())
        tpr, fpr = tp / lat.sum(), fp / ben.sum()
        print(f"{k:18s} {tpr * 100:12.2f}% {fpr * 100:11.2f}% {tpr / fpr if fpr else float('nan'):7.2f} "
              f"{tp / (tp + fp) if tp + fp else float('nan'):10.4f}")
print("\nRiferimento Euler su LANL (Tab. VI):  TPR 72.00%   FPR 4.40%   lift 16.36   P 0.0010")
