import dataclasses, numpy as np
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
N = len(s.types); tr = int(N*cfg.train_frac); va = int(N*(cfg.train_frac+cfg.val_frac))
types = s.types.numpy(); names = {0:"benign",1:"policy",2:"contextual",3:"lateral",4:"theft",5:"exfil",6:"benign-denied"}
cold = {}
for f in ["source","config","device","user"]:
    a = getattr(s,f).numpy(); seen=set(); c=np.zeros(N,bool)
    for i,x in enumerate(a):
        c[i] = x not in seen; seen.add(x)
    cold[f]=c
anyc = cold["source"]|cold["config"]|cold["device"]|cold["user"]
for part,(lo,hi) in {"train":(0,tr),"test":(va,N)}.items():
    print(f"== {part} ==")
    for t,n in names.items():
        m = np.zeros(N,bool); m[lo:hi]=True; m &= types==t
        if m.sum()==0: continue
        print(f"{n:14s} n={m.sum():6d} " + " ".join(f"{f}={cold[f][m].mean():.3f}" for f in cold) + f" any={anyc[m].mean():.3f}")
print("== counts of first-seen nodes in test ==")
for f in cold:
    m = np.zeros(N,bool); m[va:]=True; m&=cold[f]
    print(f, {names[t]: int((m&(types==t)).sum()) for t in names})
sc = s.scenario.numpy()
print("test wiped scenario benign:", int(((sc&2)>0)[va:][types[va:]==0].sum()))
print("new users total in stream:", len(set(s.user.numpy().tolist())), "first-seen after train:", int(cold['user'][tr:].sum()))
reg = s.user.numpy() < cfg.num_users
print("registered users first-seen: train", int((cold['user'][:tr] & reg[:tr]).sum()), "val", int((cold['user'][tr:va] & reg[tr:va]).sum()), "test", int((cold['user'][va:] & reg[va:]).sum()))
print("guest users first-seen: train", int((cold['user'][:tr] & ~reg[:tr]).sum()), "val", int((cold['user'][tr:va] & ~reg[tr:va]).sum()), "test", int((cold['user'][va:] & ~reg[va:]).sum()))
print("test SCEN_NEW_USER benign:", int(((sc&8)>0)[va:][types[va:]==0].sum()), "| train:", int(((sc&8)>0)[:tr][types[:tr]==0].sum()))
nu = (sc&8)>0
print("theft on recently-hired victims:", int((nu & (types==4)).sum()))
print("num nodes", s.num_nodes)
