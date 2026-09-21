import dataclasses, numpy as np
from collections import defaultdict
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
cfg = dataclasses.replace(TGNConfig(), seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
ty=s.types.numpy(); u=s.user.numpy(); d=s.device.numpy(); c=s.config.numpy(); src=s.source.numpy()
b=(ty==0)&(u<cfg.num_users)&(u!=0)
for name,a in [("devices",d),("configs",c),("sources",src)]:
    per=defaultdict(set)
    for x,y in zip(u[b],a[b]): per[x].add(y)
    print(f"{name} per user: median", np.median([len(v) for v in per.values()]))
per=defaultdict(set)
for x,y in zip(d[b],u[b]): per[x].add(y)
print("users per device: median", np.median([len(v) for v in per.values()]), "share multi-user", np.mean([len(v)>1 for v in per.values()]))
N=len(ty); va=int(N*0.8)
seen=set(); first=np.zeros(N,bool)
for i,x in enumerate(d): first[i]= x not in seen; seen.add(x)
print("theft with never-seen device node: train", first[:int(N*.7)][ty[:int(N*.7)]==4].mean(), "test", first[va:][ty[va:]==4].mean())
