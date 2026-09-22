"""Why do recently-hired users score as anomalous while never-seen guests don't?
Train the TGN on dev seed 2000 (save=False), capture test scores, break benign FPR
down by user group. argv[1] = 'hash' | 'nohash' (use_hash_identity)."""
import sys, json, dataclasses, numpy as np
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.train_tgn import stream_to_data
import graphagate.train_tgn as T

mode = sys.argv[1]
cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
cap = {}
orig = T._replay
def wrap(*a, **k):
    out = orig(*a, **k)
    if "test" in k.get("desc", ""):
        cap["scores"] = out[0]
    return out
T._replay = wrap
m = T.train_tgn(cfg, dataset=stream_to_data(s), use_hash_identity=(mode == "hash"), save=False)

N = len(s.types); tr = int(N*cfg.train_frac); va = int(N*(cfg.train_frac+cfg.val_frac))
types = s.types.numpy(); sc = s.scenario.numpy(); user = s.user.numpy()
# per-event: index of this event within its user's history; stream position of user's first event
k_in_user = np.zeros(N, int); first = {}; cnt = {}
for i, u in enumerate(user):
    cnt[u] = cnt.get(u, 0) + 1; k_in_user[i] = cnt[u] - 1; first.setdefault(u, i)
first_pos = np.array([first[u] for u in user])
nreg_min = s.user_lo
is_guest = user >= nreg_min + cfg.num_users
sco = cap["scores"]; thr = m["threshold_clean_unsup"]  # FPR@1% on signal-clean val benign
np.save(f"/app/tasks/tmp/newuser_diag_scores_{mode}.npy", sco)
ben = types[va:] == 0
g = {
  "hire_first25": (sc[va:] & 8) > 0,
  "guest_new_first25 (first seen in test)": is_guest[va:] & (first_pos[va:] >= va) & (k_in_user[va:] < 25),
  "guest_established": is_guest[va:] & (k_in_user[va:] >= 25),
  "registered_established": ~is_guest[va:] & ((sc[va:] & 8) == 0),
}
bs = sco[ben]
out = {"mode": mode, "threshold_clean_unsup": thr, "lateral_auc": m["per_type"]["lateral"]["auc"],
       "theft_auc": m["per_type"]["cred-theft"]["auc"], "groups": {}}
for name, mask in g.items():
    mk = mask & ben
    if mk.sum() == 0: continue
    x = sco[mk]
    out["groups"][name] = {"n": int(mk.sum()), "fpr_at_thr": float((x >= thr).mean()),
        "mean_pct_rank_among_test_benign": float(np.mean([(bs < v).mean() for v in x]))}
# training exposure
trm = types[:tr] == 0
out["train_exposure"] = {
  "benign_train": int(trm.sum()),
  "hire_first25": int((((sc[:tr] & 8) > 0) & trm).sum()),
  "guest_first25": int((is_guest[:tr] & (k_in_user[:tr] < 25) & trm).sum()),
  "distinct_new_guests_train": int(len({u for u in user[:tr] if u >= nreg_min + cfg.num_users})),
}
print(json.dumps(out, indent=1))
json.dump(out, open(f"/app/tasks/tmp/newuser_diag_{mode}.json", "w"), indent=1)
