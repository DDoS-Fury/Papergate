"""Analysis of diag_scores.py output: where laterals/thefts fall, what squashes them, is the
signal in the inputs (supervised probe, diagnostic only), negatives vs attacks, ablations."""
import sys
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

D = sys.argv[1] if len(sys.argv) > 1 else "/diag/out"
G = ["access", "dev>user", "cfg>user", "cfg>dev", "src>cfg"]
ABL = ["no_hist", "no_hash", "no_struct", "zero_msg", "zero_z"]
CLS = {3: "lateral", 4: "theft"}


def load(ph):
    return dict(np.load(f"{D}/diag_{ph}.npz"))


def dirty(msg):
    return (msg[:, 0] == 0) | (msg[:, 1:4] > 0.5).any(1)


def rec_at_fpr(yt, s, fpr=0.01):
    thr = np.quantile(s[yt == 0], 1 - fpr)
    return float((s[yt == 1] > thr).mean())


def pct(ref, x):
    ref = np.sort(ref)
    return np.searchsorted(ref, x, side="right") / len(ref) * 100


def sec(t):
    print(f"\n=== {t} ===")


te = load("test")
va = load("val")
cb = (te["types"] == 0) & ~dirty(te["msg"])  # clean benign test
ben = te["types"] == 0
per = te["per"]
print(f"test events={len(te['idx'])} clean-benign={cb.sum()} "
      f"lateral={(te['types']==3).sum()} (dirty {(dirty(te['msg']) & (te['types']==3)).sum()}) "
      f"theft={(te['types']==4).sum()}")

# ---------------------------------------------------------------- 1. where they fall
sec("1. Percentile dello score nella distribuzione del benigno pulito (test)")
ref = te["score"][cb]
for q in (0.5, 0.9, 0.95, 0.99):
    print(f"  benign clean q{q:.2f} score = {np.quantile(ref, q):.4g}")
for c, name in CLS.items():
    m = te["types"] == c
    p = pct(ref, te["score"][m])
    print(f"  {name:8s} n={m.sum():4d} percentile: median={np.median(p):.1f} p75={np.quantile(p,.75):.1f} "
          f"p90={np.quantile(p,.9):.1f} | >benign p90: {(p>90).mean():.2f} >p99: {(p>99).mean():.3f} "
          f"| precursor shift>0: {(te['shift'][m]>1e-9).mean():.2f}")

sec("1b. Sottotipi lateral (dedotti dai contatori storici al momento dell'evento) vs benigno con la stessa novità")
lat = te["types"] == 3
new_bind = per[:, 1, 3] == 0      # device->user never seen (foreign-cred pivot / hot-desking)
new_cfg = per[:, 3, 3] == 0       # config->device never seen (new tool / new client)
new_acc = per[:, 0, 3] == 0       # user->resource never seen (exploration)
new_devres = per[:, 0, 6] == 0    # device->resource never seen
for nm, f in (("dev>user nuovo", new_bind), ("cfg>dev nuovo", new_cfg),
              ("user>res nuovo", new_acc), ("dev>res nuovo", new_devres)):
    print(f"  {nm:16s} | lateral: {f[lat].mean():.2f} | theft: {f[te['types']==4].mean():.2f} | "
          f"benign pulito: {f[cb].mean():.3f}")
for nm, f in (("dev>user nuovo", new_bind), ("dev>user visto", ~new_bind),
              ("cfg>dev nuovo", new_cfg), ("user>res nuovo", new_acc), ("user>res visto", ~new_acc)):
    m = lat & f
    if m.sum() < 3:
        continue
    sel = cb | m
    yt = m[sel].astype(int)
    b_same = cb & f
    sel2 = b_same | m
    auc_same = roc_auc_score(m[sel2].astype(int), te["score"][sel2]) if b_same.any() else float("nan")
    print(f"  lateral con {nm:15s} n={m.sum():4d} | AUC vs benigno={roc_auc_score(yt, te['score'][sel]):.3f} "
          f"AP={average_precision_score(yt, te['score'][sel]):.4f} (caso {yt.mean():.4f}) | "
          f"AUC vs benigno con stessa novità (n={b_same.sum()}): {auc_same:.3f}")

# ---------------------------------------------------------------- 2. decomposition
sec("2. Quale edge dà il max e con quale testa (mediane dei logit di anomalia)")
arg = per[:, :, 0].argmax(1)
for nm, m in (("benign pulito", cb), ("lateral", lat), ("theft", te["types"] == 4)):
    share = np.bincount(arg[m], minlength=5) / m.sum()
    print(f"  {nm:13s} argmax: " + " ".join(f"{g}={s:.2f}" for g, s in zip(G, share)))
print("  per edge: logit totale / feature-head / struct-head  [benign | lateral | theft]"
      "  + percentile mediano del lateral / theft nel benigno per quell'edge")
for g in range(5):
    cells = []
    for m in (cb, lat, te["types"] == 4):
        cells.append("/".join(f"{np.median(per[m, g, k]):+.2f}" for k in (0, 1, 2)))
    pl = np.median(pct(per[cb, g, 0], per[lat, g, 0]))
    pt = np.median(pct(per[cb, g, 0], per[te["types"] == 4, g, 0]))
    print(f"  {G[g]:9s} " + " | ".join(cells) + f"   pct lat={pl:.0f} theft={pt:.0f}")
sec("2b. AUC per singolo edge / singola testa (benigno pulito vs classe)")
for c, name in CLS.items():
    m = te["types"] == c
    sel = cb | m
    yt = m[sel].astype(int)
    row = []
    for g in range(5):
        row.append(f"{G[g]}: tot={roc_auc_score(yt, per[sel, g, 0]):.2f} "
                   f"feat={roc_auc_score(yt, per[sel, g, 1]):.2f} struct={roc_auc_score(yt, per[sel, g, 2]):.2f}")
    print(f"  {name}:\n    " + "\n    ".join(row))

# ---------------------------------------------------------------- 3. inputs + probe
sec("3. Input grezzi: mediane [log1p pair_count, log1p src_count, ratio | aux 3 | log1p Δt_pair, log1p Δt_src]")
for g in range(5):
    for nm, m in (("benign", cb), ("lateral", lat), ("theft", te["types"] == 4)):
        v = np.median(per[m, g, 3:11], axis=0)
        print(f"  {G[g]:9s} {nm:8s} " + " ".join(f"{x:6.2f}" for x in v))


def feats(d, which):
    p = d["per"]
    hand = np.concatenate([p[:, :, 3:11].reshape(len(p), -1), d["msg"]], 1)
    if which == "hand":
        return hand
    if which == "logits":
        return p[:, :, :3].reshape(len(p), -1)
    if which == "hand+z":
        return np.concatenate([hand, d["z"].astype(np.float32)], 1)
    raise ValueError(which)


sec("3b. Probe supervisionato (solo diagnostico), GroupKFold per device su val+test, benigno pulito vs classe")
al = {k: np.concatenate([va[k], te[k]]) for k in ("per", "msg", "z", "types", "device", "score")}
cb_all = (al["types"] == 0) & ~dirty(al["msg"])
for c, name in CLS.items():
    m = cb_all | (al["types"] == c)
    yt = (al["types"][m] == c).astype(int)
    grp = al["device"][m]
    print(f"  {name}: n_pos={yt.sum()} n_benign={(yt==0).sum()} (caso AP={yt.mean():.4f})")
    res = {"score modello": al["score"][m]}
    for which in ("logits", "hand", "hand+z"):
        X = feats({k: v[m] for k, v in al.items()}, which)
        oof = np.zeros(len(yt))
        for tr, ts in GroupKFold(n_splits=5).split(X, yt, grp):
            if which == "hand+z":
                clf = make_pipeline(StandardScaler(), PCA(64, random_state=0),
                                    HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                   class_weight="balanced", random_state=0))
            else:
                clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                     class_weight="balanced", random_state=0)
            clf.fit(X[tr], yt[tr])
            oof[ts] = clf.predict_proba(X[ts])[:, 1]
        res[f"probe {which}"] = oof
    for k, s in res.items():
        print(f"    {k:18s} AUC={roc_auc_score(yt, s):.3f} AP={average_precision_score(yt, s):.4f} "
              f"rec@FPR1%={rec_at_fpr(yt, s):.3f} rec@FPR5%={rec_at_fpr(yt, s, .05):.3f}")

# ---------------------------------------------------------------- 4. negatives
sec("4. Negativi strutturali (come in training) vs positivi benigni vs attacchi: logit di anomalia per edge")
xi = te["x_idx"]
pos = {v: i for i, v in enumerate(te["idx"])}
rows = np.array([pos[v] for v in xi])
xt = te["types"][rows]
xneg = te["x_neg"]  # [n, group, K]
xb = (xt == 0) & ~dirty(te["msg"][rows])
for g in range(5):
    nb = xneg[xb, g].ravel()
    bpos = per[cb, g, 0]
    lp = per[lat, g, 0]
    thp = per[te["types"] == 4, g, 0]
    thr99 = np.quantile(bpos, 0.99)
    print(f"  {G[g]:9s} mediana: benign pos={np.median(bpos):+.2f} neg={np.median(nb):+.2f} "
          f"lateral={np.median(lp):+.2f} theft={np.median(thp):+.2f} | AUC benign-vs-neg="
          f"{roc_auc_score(np.r_[np.zeros(len(bpos)), np.ones(len(nb))], np.r_[bpos, nb]):.3f} "
          f"| >benign p99: neg={(nb>thr99).mean():.2f} lat={(lp>thr99).mean():.2f} theft={(thp>thr99).mean():.2f}")

# ---------------------------------------------------------------- 5. ablations
sec("5. Ablation per blocco di input (logit evento, max sugli edge)")
xraw = te["raw"][rows]
xabl = te["x_abl"]
for c, name in CLS.items():
    m = xt == c
    if m.sum() == 0 or xb.sum() == 0:
        continue
    sel = xb | m
    yt = m[sel].astype(int)
    base = roc_auc_score(yt, xraw[sel])
    base_ap = average_precision_score(yt, xraw[sel])
    print(f"  {name}: completo AUC={base:.3f} AP={base_ap:.3f} (su campione benigno n={xb.sum()})")
    for a in range(5):
        d_c = np.median(xabl[m, a] - xraw[m])
        d_b = np.median(xabl[xb, a] - xraw[xb])
        print(f"    {ABL[a]:9s} Δlogit mediano {name}={d_c:+.2f} benign={d_b:+.2f} | "
              f"AUC={roc_auc_score(yt, xabl[sel, a]):.3f} AP={average_precision_score(yt, xabl[sel, a]):.3f}")
