"""Per-epoch probe of the structural head during training (TODO 3: why it collapses).

No change to src/: ``model.score`` is wrapped on the instance to record, for every scored
edge group, the cosine the structural head sees; parameter hooks record gradient norms.
Nothing here consumes RNG or touches autograd state (stats run under no_grad, the
projection is recomputed without its Dropout layer), so the training trajectory is the
one of a normal run. The run stops at the end of training (calibration / replay skipped).

Per batch the training loop calls ``score`` in a fixed order (train_tgn): access pos,
access neg, contextual, then pos/neg per binding edge (dev>user, cfg>user, cfg>dev,
src>cfg). Per epoch and edge kind it reports:
  * cos_pos / cos_neg      mean (std) of the deterministic cosine, positives vs negatives
  * R                      mean resultant length of the projected, normalised endpoints
                           (1 = every node maps to one direction: collapse)
  * zcos                   mean pairwise cosine of the raw embeddings z (input collapse?)
  * p_pos, s_pos           InfoNCE softmax prob of the positive and sigmoid(pos logit):
                           (1 - p) and (1 - s) are the logit-gradient weights of the
                           InfoNCE and positive-BCE terms on the positive
  * auc_full / auc_struct  pos-vs-neg separability of the full logit / of the cosine
and globally ``struct_scale`` plus the gradient norms of struct_proj, struct_scale and
link_pred (sum over the epoch's batches, pre-optimizer).
"""
import json
import numpy as np
import torch
import torch.nn.functional as F

from graphagate import train_tgn as T
from graphagate.config import TGNConfig

KINDS = ["user>res", "dev>user", "cfg>user", "cfg>dev", "src>cfg"]
# call index within a batch -> (kind, role)
ORDER = [("user>res", "pos"), ("user>res", "neg"), ("user>res", "ctx")]
for k in KINDS[1:]:
    ORDER += [(k, "pos"), (k, "neg")]
EVERY = 5  # record stats every EVERY-th batch (gradients: every batch)

S = {"model": None, "call": 0, "batch": 0, "epoch": 0, "acc": None, "out": None, "rows": []}


def _new_acc():
    return {"cos": {}, "logit": {}, "R": {}, "zcos": [], "grad": {"proj": 0.0, "scale": 0.0, "feat": 0.0}}


def _auc(pos, neg):
    """Mann-Whitney AUC, P(pos logit > neg logit) (benign scored higher)."""
    x = np.concatenate([pos, neg])
    r = x.argsort().argsort().astype(np.float64) + 1
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _flush():
    acc, m = S["acc"], S["model"]
    if acc is None or S["epoch"] == 0:
        return
    row = {"epoch": S["epoch"], "struct_scale": float(m.struct_scale.detach()),
           "grad": acc["grad"], "zcos": float(np.mean(acc["zcos"])) if acc["zcos"] else None}
    for k in KINDS:
        if (k, "pos") not in acc["cos"]:
            continue
        cp, cn = np.concatenate(acc["cos"][(k, "pos")]), np.concatenate(acc["cos"][(k, "neg")])
        lp = np.concatenate(acc["logit"][(k, "pos")])
        ln = np.concatenate(acc["logit"][(k, "neg")])  # [n, K]
        lse = np.logaddexp.reduce(np.concatenate([lp[:, None], ln], 1), axis=1)
        row[k] = {
            "cos_pos": [float(cp.mean()), float(cp.std())],
            "cos_neg": [float(cn.mean()), float(cn.std())],
            "R": float(np.mean(acc["R"][k])),
            "p_pos": float(np.exp(lp - lse).mean()),
            "s_pos": float((1 / (1 + np.exp(-lp))).mean()),
            "auc_full": _auc(lp, ln.ravel()),
            "auc_struct": _auc(cp, cn),
        }
    S["rows"].append(row)
    print("[probe] " + json.dumps(row), flush=True)
    with open(S["out"], "w") as f:
        json.dump(S["rows"], f, indent=1)


def _det_proj(m, x):
    p = m.struct_proj
    return F.normalize(p[3](p[1](p[0](x))), dim=-1)  # Linear, ReLU, (Dropout skipped), Linear


def _wrap_score(m):
    orig = m.score

    def score(z, nf, h_idx, src_local, dst_local, *a, **kw):
        out = orig(z, nf, h_idx, src_local, dst_local, *a, **kw)
        if not m.training:
            return out
        i = S["call"]
        S["call"] += 1
        if S["batch"] % EVERY or i >= len(ORDER):
            return out
        kind, role = ORDER[i]
        if role == "ctx":
            return out
        with torch.no_grad():
            hs, hd = _det_proj(m, z[src_local]), _det_proj(m, z[dst_local])
            cos = (hs * hd).sum(-1).float().cpu().numpy()
            lg = out.detach().float().cpu().numpy()
            acc = S["acc"]
            acc["cos"].setdefault((kind, role), []).append(cos)
            acc["logit"].setdefault((kind, role), []).append(lg if role == "pos" else lg.reshape(-1, T_K))
            if role == "pos":
                ends = torch.cat([hs, hd])
                acc["R"].setdefault(kind, []).append(float(ends.mean(0).norm()))
                if i == 0:
                    zz = F.normalize(z[torch.unique(torch.cat([src_local, dst_local]))].float(), dim=-1)
                    n = zz.size(0)
                    acc["zcos"].append(float(((zz.sum(0).norm() ** 2 - n) / max(n * (n - 1), 1))))
        return out

    m.score = score


def _grad_hooks(m):
    def add(key):
        def h(g):
            S["acc"]["grad"][key] += float(g.detach().norm()) ** 2
        return h
    for p in m.struct_proj.parameters():
        p.register_hook(add("proj"))
    m.struct_scale.register_hook(add("scale"))
    for p in m.link_pred.parameters():
        p.register_hook(add("feat"))


class _Stop(Exception):
    pass


_orig_pbar = T._pbar


class _Bar:
    """Epoch progress bar proxy: marks batch boundaries, delegates the rest (set_postfix...)."""

    def __init__(self, bar):
        self._bar = bar

    def __iter__(self):
        for x in self._bar:
            S["call"] = 0
            yield x
            S["batch"] += 1

    def __getattr__(self, name):
        return getattr(self._bar, name)


def _pbar(iterable=None, *, total=None, desc=None):
    desc = desc or ""
    if desc.startswith("Epoch"):
        _flush()
        S.update(epoch=int(desc.split()[1].split("/")[0]), acc=_new_acc(), batch=0)
        return _Bar(_orig_pbar(iterable, total=total, desc=desc))
    if desc.startswith("Calibration"):
        _flush()
        raise _Stop
    return _orig_pbar(iterable, total=total, desc=desc)


if __name__ == "__main__":
    import argparse, dataclasses
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--out", default="/diag/struct_probe.json")
    a = ap.parse_args()
    S["out"] = a.out
    cfg = dataclasses.replace(TGNConfig(), num_events=a.events, epochs=a.epochs)
    T_K = cfg.infonce_k

    # Catch the model at its first ``model.train()`` (start of epoch 1), before any batch.
    _orig_mod_train = torch.nn.Module.train

    def _train(self, mode=True):
        if S["model"] is None and hasattr(self, "struct_proj"):
            S["model"] = self
            _wrap_score(self)
            _grad_hooks(self)
            S["acc"] = _new_acc()
        return _orig_mod_train(self, mode)

    torch.nn.Module.train = _train
    T._pbar = _pbar
    try:
        T.train_tgn(cfg, save=False)
    except _Stop:
        pass
    print("[probe] done", flush=True)
