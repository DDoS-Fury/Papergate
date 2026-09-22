"""Seed 2000, default config: which benign test events end up above the global 1%-FPR
threshold? Captures test scores and the precursor boost per event, saves them to
fp_diag.npz and prints the breakdown of the false positives."""
import dataclasses, json, numpy as np
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.data.lookup_rules import lookup_flags
from graphagate.train_tgn import stream_to_data
import graphagate.train_tgn as T

cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
cap = {"on": False, "boost": [], "val": None}
orig_replay, orig_boost = T._replay, T.precursor_boost

def boost(*a, **k):
    b = orig_boost(*a, **k)
    if cap["on"]:
        cap["boost"].append(b)
    return b

def replay(*a, **k):
    desc = k.get("desc", "")
    cap["on"] = "test" in desc.lower()
    cap["boost"] = [] if cap["on"] else cap["boost"]
    out = orig_replay(*a, **k)
    if cap["on"]:
        cap["test"] = out[0]
    elif "pass B" in desc:
        cap["val"] = out[0]
    cap["on"] = False
    return out

T._replay, T.precursor_boost = replay, boost
m = T.train_tgn(cfg, dataset=stream_to_data(s), save=False)

N = len(s.types); tr = int(N * cfg.train_frac); va = int(N * (cfg.train_frac + cfg.val_frac))
ty = s.types.numpy(); msg = s.msg.numpy(); sc = s.scenario.numpy(); user = s.user.numpy()
sco = np.asarray(cap["test"]); bst = np.asarray(cap["boost"], dtype=float)
thr = m["threshold_dirty"]
f = lookup_flags(s, gate="proto-self", test_start=va)
is_guest = user >= s.user_lo + cfg.num_users
np.savez("/app/tasks/tmp/fp_diag.npz", test_scores=sco, test_boost=bst, val_scores=cap["val"],
         thr_global=thr, thr_clean_unsup=m["threshold_clean_unsup"], test_start=va, val_start=tr)

ben = ty[va:] == 0
mt = msg[va:]
masks = {
    "ja3=0 (legacy client)": mt[:, 0] == 0,
    "snort fp (s1|s2|s3)": (mt[:, 1:4] == 1).any(1),
    "precursor boost > 1": bst > 1.0,
    "guest": is_guest[va:],
    "hire first 25": (sc[va:] & 8) > 0,
    "dev|usr_new": f["dev|usr_new"][va:],
    "cfg|usr_new": f["cfg|usr_new"][va:],
    "src|usr_new": f["src|usr_new"][va:],
    "cfg|dev_new": f["cfg|dev_new"][va:],
}
fp = ben & (sco >= thr)
out = {"len_scores": len(sco), "len_boost": len(bst), "thr_global": thr,
       "benign_test": int(ben.sum()), "fp": int(fp.sum()), "fpr": float(fp[ben].mean()),
       "benign_score_ge_0.9999": float((sco[ben] >= 0.9999).mean()),
       "benign_score_eq_1": float((sco[ben] >= 1.0).mean()),
       "val_benign_ge_0.9999": float((np.asarray(cap["val"])[ty[tr:va] == 0] >= 0.9999).mean()),
       "breakdown": {}}
for k, mk in masks.items():
    out["breakdown"][k] = {"share_of_fp": float(mk[fp].mean()), "share_of_benign": float(mk[ben].mean()),
                           "fpr_within": float((sco[ben & mk] >= thr).mean()) if (ben & mk).any() else None}
for g, t in (("lateral", 3), ("theft", 4)):
    sel = ty[va:] == t
    out[g] = {"n": int(sel.sum()), "score_ge_0.9999": float((sco[sel] >= 0.9999).mean()),
              "median_score": float(np.median(sco[sel]))}
print(json.dumps(out, indent=1))
json.dump(out, open("/app/tasks/tmp/fp_diag.json", "w"), indent=1)
