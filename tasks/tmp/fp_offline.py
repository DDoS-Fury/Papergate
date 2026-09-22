"""Breakdown of the benign test events above the global threshold, from the scores saved
by fp_diag.py (stream regenerated with the generator that run used)."""
import dataclasses, json, sys, numpy as np
import graphagate
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.data.lookup_rules import lookup_flags
print(graphagate.__file__)
d = np.load(sys.argv[1])
cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
va = int(d["test_start"]); ty = s.types.numpy()[va:]; mt = s.msg.numpy()[va:]; sc = s.scenario.numpy()[va:]
user = s.user.numpy()[va:]
sco, bst, thr = d["test_scores"], d["test_boost"], float(d["thr_global"])
assert len(sco) == len(ty), (len(sco), len(ty))
f = lookup_flags(s, gate="proto-self", test_start=va)
ben = ty == 0
masks = {
    "ja3=0 (legacy)": mt[:, 0] == 0,
    "snort fp": (mt[:, 1:4] == 1).any(1),
    "signal-dirty (ja3=0|snort)": (mt[:, 0] == 0) | (mt[:, 1:4] == 1).any(1),
    "boost>1": bst[:len(sco)] > 1 if len(bst) >= len(sco) else np.zeros(len(sco), bool),
    "guest": user >= s.user_lo + cfg.num_users,
    "roaming": (sc & 1) > 0, "wiped": (sc & 2) > 0, "shared": (sc & 4) > 0, "new hire": (sc & 8) > 0,
    **{k: f[k][va:].astype(bool) for k in ("dev|usr_new", "cfg|usr_new", "src|usr_new", "cfg|dev_new", "sensor")},
}
fp = ben & (sco >= thr)
print(f"thr={thr:.6f} len boost={len(bst)} scores={len(sco)} benign={ben.sum()} FP={fp.sum()} FPR={fp[ben].mean():.4f}")
print(f"benign score>=0.9999: {(sco[ben]>=0.9999).mean():.4f}  ==1: {(sco[ben]>=1).mean():.4f}  p50/p95/p99: {np.quantile(sco[ben],[.5,.95,.99])}")
print(f"{'mask':28s} {'%FP':>6s} {'%benign':>8s} {'FPR within':>10s}")
for k, m in masks.items():
    print(f"{k:28s} {m[fp].mean():6.3f} {m[ben].mean():8.3f} {(sco[ben & m] >= thr).mean() if (ben & m).any() else float('nan'):10.3f}")
nov = masks["dev|usr_new"] | masks["cfg|usr_new"] | masks["src|usr_new"] | masks["cfg|dev_new"]
exp = masks["signal-dirty (ja3=0|snort)"] | nov | masks["guest"]
print(f"FP explained by dirty|novelty|guest: {exp[fp].mean():.3f}; FP none of these: {int((fp & ~exp).sum())}")
for g, t in (("lateral", 3), ("theft", 4)):
    m = ty == t
    print(f"{g}: n={m.sum()} >=thr {(sco[m]>=thr).mean():.3f} median {np.median(sco[m]):.4f}")
leg = ben & masks["ja3=0 (legacy)"]
q = np.array_split(np.arange(len(sco)), 4)
print("legacy benign FPR by test quarter:", [round(float((sco[i][leg[i]] >= thr).mean()), 3) for i in q])
print("legacy benign score>=0.9999 share:", round(float((sco[leg] >= 0.9999).mean()), 3),
      "| non-legacy dirty FPR:", round(float((sco[ben & masks['snort fp'] & ~masks['ja3=0 (legacy)']] >= thr).mean()), 3))
cl = ben & ~masks["signal-dirty (ja3=0|snort)"]
print(f"signal-clean benign: n={cl.sum()} FPR at global thr={(sco[cl]>=thr).mean():.4f}")
