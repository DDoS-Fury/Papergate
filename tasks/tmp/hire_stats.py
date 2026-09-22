import dataclasses, numpy as np
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
import sys
old = len(sys.argv)>1
cfg = dataclasses.replace(TGNConfig(), seed=2000, **(dict(num_users=50,num_new_users=12,num_devices=80,num_sources=150) if old else {}))
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
N=len(s.types); tr=int(N*cfg.train_frac); va=int(N*(cfg.train_frac+cfg.val_frac))
sc=s.scenario.numpy(); u=s.user.numpy(); ty=s.types.numpy(); t=s.t.numpy() if hasattr(s,'t') else None
hire=(sc&8)>0
for name,a,b in [("train",0,tr),("val",tr,va),("test",va,N)]:
    h=hire[a:b]; print(name, "cold-hire events", int(h.sum()), "benign", int((h&(ty[a:b]==0)).sum()), "distinct hires", len(set(u[a:b][h])))
is_guest=u>=s.user_lo+cfg.num_users
print("guest first-seen users in train", len(set(u[:tr][is_guest[:tr]])))
print("attack prevalence", float((ty!=0).mean()))
reg=u[~is_guest]; c=np.bincount(reg-s.user_lo, minlength=cfg.num_users); print("events/registered user min/med/max", c.min(), int(np.median(c)), c.max())
for attr in ("t","timestamps","ts"):
    if hasattr(s,attr): x=getattr(s,attr).numpy(); print("duration months", (x[-1]-x[0])/86400/30.4); break
from collections import Counter; print("types", sorted(Counter(ty.tolist()).items()))
