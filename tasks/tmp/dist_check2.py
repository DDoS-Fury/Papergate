import dataclasses, sys, json, numpy as np
from collections import defaultdict
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg
def B(g):
    g=np.asarray(g,float); m,s=g.mean(),g.std(); return round(s/m,3), round((s-m)/(s+m),3)
for label, over in [("base",{}),("no_hotdesk",{"p_hotdesk":0.0})]:
    cfg = dataclasses.replace(TGNConfig(), seed=2000, **over)
    s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
    t=s.t.numpy().astype(float); u=s.user.numpy(); d=s.device.numpy(); c=s.config.numpy(); ty=s.types.numpy(); src=s.source.numpy()
    wk=((t//86400)%7<5)&((t%86400)>=9*3600)&((t%86400)<17*3600)
    day=(t//86400).astype(int)
    # global gaps inside weekday 9-17 same day
    tw=t[wk]; dw=day[wk]; g=np.diff(tw)[np.diff(dw)==0]
    out={"global_weekday_work_cv_B":B(g)}
    humans=set(range(s.user_lo+1,s.user_lo+cfg.num_users))
    cv=[];bb=[]
    for uu in humans:
        m=(u==uu)&wk; tt=t[m]; dd=day[m]; gg=np.diff(tt)[np.diff(dd)==0]
        if len(gg)>30: a,b=B(gg); cv.append(a); bb.append(b)
    out["per_user_weekday_work_cv_med"]=float(np.median(cv)); out["per_user_B_med"]=float(np.median(bb))
    ben=(ty==0)&np.isin(u,list(humans))
    for name,(a,b) in {"dev_per_user":(u,d),"cfg_per_user":(u,c),"src_per_user":(u,src),"users_per_dev":(d,u)}.items():
        dd=defaultdict(set)
        for x,y in zip(a[ben].tolist(),b[ben].tolist()): dd[x].add(y)
        v=np.array([len(z) for z in dd.values()]); out[name+"_median"]=float(np.median(v))
    # share of each user's benign events on their top device
    dd=defaultdict(lambda: defaultdict(int))
    for x,y in zip(u[ben].tolist(),d[ben].tolist()): dd[x][y]+=1
    out["top_device_share_median"]=float(np.median([max(v.values())/sum(v.values()) for v in dd.values()]))
    ds=defaultdict(lambda: defaultdict(int))
    for x,y in zip(u[ben].tolist(),src[ben].tolist()): ds[x][y]+=1
    out["top2_src_share_median"]=float(np.median([sum(sorted(v.values())[-2:])/sum(v.values()) for v in ds.values()]))
    # daily active users per weekday
    dau=[len(set(u[(day==k)&ben].tolist())) for k in range(int(day.max())) if k%7<5]
    out["weekday_DAU_median"]=float(np.median(dau))
    print(label, json.dumps(out))
