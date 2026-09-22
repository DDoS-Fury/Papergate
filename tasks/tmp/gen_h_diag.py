"""Fix H (cached-credential pivot): lookup-rule firing on lateral / theft, with the
harvest drawn only from users who never signed in on the machine (p_harvest_cached=0)
vs mostly from its logon cache (default). Dev seeds only."""
import dataclasses
import numpy as np
from sklearn.metrics import roc_auc_score
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
from graphagate.data.lookup_rules import lookup_flags

for seed in (2000, 2001, 2002):
    for p in (0.0, TGNConfig().p_harvest_cached):
        cfg = dataclasses.replace(TGNConfig(), seed=seed, p_harvest_cached=p)
        s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
        N = len(s.types); va = int(N * (cfg.train_frac + cfg.val_frac))
        ty = s.types.numpy()[va:]
        f = lookup_flags(s, gate="proto-self", test_start=va)
        ben, lat, th = ty == 0, ty == 3, ty == 4
        st = f["stateful"][va:]
        auc = {g: roc_auc_score(m[ben | m], st[ben | m]) for g, m in (("lat", lat), ("theft", th))}
        rates = " ".join(f"{k}={f[k][va:][lat].mean():.2f}" for k in ("dev|usr_new", "src|usr_new", "cfg|usr_new"))
        print(f"seed {seed} p_cached={p:.1f} lat={lat.sum():4d} theft={th.sum():4d} "
              f"prev={(ty != 0).mean():.3f} | lateral {rates} | stateful AUC lat={auc['lat']:.3f} theft={auc['theft']:.3f}",
              flush=True)
