"""Points 1 and 6 of tasks/todo.md, offline from a precursor_sweep dump.

Point 1: what the benign events above the global 1%-FPR threshold actually are, now that
the legacy-client cohort is gone from the generator.
Point 6: the campaign-level recall, which is the aggregation the LANL literature reports.
A campaign = the lateral events of ONE machine separated by less than 7 days (grouping by
device and not by adjacency in the stream: several machines are compromised at the same
time, so their lateral events interleave — see tasks/tmp/lateral_chain_diag.py).

Run:
  docker compose run --rm --no-deps -T -v "$PWD/tasks:/app/tasks" \
      --entrypoint python train-tgn /app/tasks/tmp/fp_breakdown.py <dump.npz>
"""
import dataclasses, sys

import numpy as np

from graphagate.config import TGNConfig
from graphagate.data.lookup_rules import lookup_flags
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

d = np.load(sys.argv[1])
cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))

va = int(d["test_start"])
sco, lab = d["test_scores"], d["test_labels"]
thr, thr_clean = float(d["thr_global"]), float(d["thr_clean"])
ty, mt, sc_flags = s.types.numpy()[va:], s.msg.numpy()[va:], s.scenario.numpy()[va:]
user, tt = s.user.numpy()[va:], s.t.numpy()[va:]
dev = s.device.numpy()[va:]
assert len(sco) == len(ty), (len(sco), len(ty))
f = lookup_flags(s, gate="proto-self", test_start=va)

ben, lat = lab == 0, ty == 3
fp = ben & (sco >= thr)
print(f"thr_global={thr:.8f} thr_clean={thr_clean:.8f}")
print(f"benign={ben.sum()} FP={fp.sum()} FPR={fp[ben].mean():.4f} | "
      f"lateral={lat.sum()} recall@thr={(sco[lat] >= thr).mean():.4f}")
print(f"benign score==1.0: {(sco[ben] >= 1.0).sum()} | lateral score==1.0: {(sco[lat] >= 1.0).sum()}")

masks = {
    "ja3=0 (legacy client)": mt[:, 0] == 0,
    "snort fp": (mt[:, 1:4] == 1).any(1),
    "signal-dirty": (mt[:, 0] == 0) | (mt[:, 1:4] == 1).any(1),
    "guest": user >= s.user_lo + cfg.num_users,
    "roaming": (sc_flags & 1) > 0, "wiped": (sc_flags & 2) > 0,
    "shared device": (sc_flags & 4) > 0, "new hire": (sc_flags & 8) > 0,
    **{k: f[k][va:].astype(bool) for k in ("dev|usr_new", "cfg|usr_new", "src|usr_new",
                                           "cfg|dev_new", "sensor")},
}
print(f"\n{'mask':24s} {'%ofFP':>7s} {'%benign':>8s} {'FPR within':>11s} {'lift':>6s}")
print("-" * 60)
base = fp[ben].mean()
for k, m in sorted(masks.items(), key=lambda kv: -(fp & kv[1]).sum()):
    within = (sco[ben & m] >= thr).mean() if (ben & m).any() else float("nan")
    print(f"{k:24s} {m[fp].mean():7.3f} {m[ben].mean():8.3f} {within:11.3f} "
          f"{within / base if base else float('nan'):6.2f}")

nov = masks["dev|usr_new"] | masks["cfg|usr_new"] | masks["src|usr_new"] | masks["cfg|dev_new"]
exp = masks["signal-dirty"] | nov | masks["guest"]
print(f"\nFP explained by dirty|novelty|guest: {exp[fp].mean():.3f} | "
      f"unexplained: {int((fp & ~exp).sum())}")

# --- point 6: campaign-level recall -----------------------------------------------
idx_lat = np.nonzero(lat)[0]
camps = []
for m in np.unique(dev[idx_lat]):
    idx = idx_lat[dev[idx_lat] == m]
    cur = [idx[0]]
    for i in idx[1:]:
        if tt[i] - tt[cur[-1]] < 7 * 86400:
            cur.append(i)
        else:
            camps.append(cur)
            cur = [i]
    camps.append(cur)
sizes = np.array([len(c) for c in camps])
print(f"\ncampaigns={len(camps)} events/campaign: median={np.median(sizes):.0f} "
      f"mean={sizes.mean():.1f} max={sizes.max()}")
for name, t in (("global @1%FPR", thr), ("clean cost-sensitive", thr_clean)):
    hit = np.array([bool((sco[c] >= t).any()) for c in camps])
    ev = float((sco[lat] >= t).mean())
    print(f"  {name:22s} per-event recall={ev:.4f} | campaign recall={hit.mean():.4f} "
          f"({hit.sum()}/{len(camps)}) | i.i.d. prediction 1-(1-r)^median={1 - (1 - ev) ** np.median(sizes):.4f}")
