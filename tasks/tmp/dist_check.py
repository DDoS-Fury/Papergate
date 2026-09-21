"""Distribution audit: synthetic generator vs PicoDomain. Writes JSON to tasks/tmp/dist_check_<name>.json.

Run: docker compose run --rm -v "$PWD/data:/data" --entrypoint python regen-report /app/tasks/tmp/dist_check.py
"""
import dataclasses, json, sys, os
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, "/app")
from graphagate.config import TGNConfig
from graphagate.data import stream_synthetic as ss
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

NAMES = {0: "benign", 1: "policy", 2: "contextual", 3: "lateral", 4: "theft", 5: "exfil", 6: "benign-denied"}


def zipf_fit(counts, top=None):
    c = np.sort(np.asarray([x for x in counts if x > 0], float))[::-1]
    if top:
        c = c[:top]
    if len(c) < 5:
        return None
    r = np.arange(1, len(c) + 1)
    a, _ = np.polyfit(np.log(r), np.log(c), 1)
    return round(float(-a), 3)


def zipf_mle(counts, xmin_rank_max=None):
    """Discrete MLE of s for a rank-frequency Zipf over N ranks (Clauset-style on ranks)."""
    c = np.sort(np.asarray([x for x in counts if x > 0], float))[::-1]
    N = len(c)
    if N < 5:
        return None
    r = np.arange(1, N + 1)
    ss_ = np.linspace(0.05, 3.0, 296)
    ll = [-(s * (c * np.log(r)).sum()) - c.sum() * np.log((r ** -s).sum()) for s in ss_]
    return round(float(ss_[int(np.argmax(ll))]), 3)


def q(a, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)):
    a = np.asarray(a, float)
    if len(a) == 0:
        return None
    return {str(k): round(float(np.quantile(a, k)), 3) for k in qs}


def gini(a):
    a = np.sort(np.asarray(a, float))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return None
    return round(float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * a.sum())), 3)


def burst(g):
    g = np.asarray(g, float)
    if len(g) < 3:
        return None, None
    m, s = g.mean(), g.std()
    return round(float(s / m), 3), round(float((s - m) / (s + m)), 3)


def analyse(name, src, cfg, dev, usr, dst, t, types, tr_frac=0.7, va_frac=0.1, user_filter=None, t0_epoch=0):
    N = len(t)
    out = {"name": name, "N": int(N), "span_days": round(float((t.max() - t.min()) / 86400), 2)}
    benign = types == 0
    out["class_counts"] = {NAMES.get(k, str(k)): int((types == k).sum()) for k in sorted(set(types.tolist()))}
    out["attack_prevalence_etype1_5"] = round(float(np.isin(types, [1, 2, 3, 4, 5]).mean()), 5)
    out["label1_prevalence_incl_denied"] = round(float((types != 0).mean()), 5)
    out["distinct"] = {k: int(len(set(v.tolist()))) for k, v in
                       dict(source=src, config=cfg, device=dev, user=usr, dst=dst).items()}

    # users to analyse (humans/registered): user_filter(key) -> bool
    um = np.array([user_filter(u) for u in usr]) if user_filter else np.ones(N, bool)

    # 1. resource popularity (benign)
    gc = Counter(dst[benign].tolist())
    cnt = np.array(sorted(gc.values(), reverse=True))
    out["res_global"] = {
        "n_resources_hit": int(len(cnt)),
        "zipf_ols_all": zipf_fit(cnt), "zipf_ols_top100": zipf_fit(cnt, 100), "zipf_ols_top20": zipf_fit(cnt, 20),
        "zipf_mle": zipf_mle(cnt),
        "top1_share": round(float(cnt[0] / cnt.sum()), 4), "top10_share": round(float(cnt[:10].sum() / cnt.sum()), 4),
        "top10pct_res_share": round(float(cnt[: max(1, len(cnt) // 10)].sum() / cnt.sum()), 4),
    }
    per_user_alpha, per_user_mle = [], []
    ud = defaultdict(list)
    for u, d in zip(usr[benign & um].tolist(), dst[benign & um].tolist()):
        ud[u].append(d)
    for u, ds in ud.items():
        if len(ds) >= 200:
            c = list(Counter(ds).values())
            a = zipf_fit(c); b = zipf_mle(c)
            if a is not None:
                per_user_alpha.append(a); per_user_mle.append(b)
    out["res_per_user"] = {"n_users_ge200ev": len(per_user_alpha), "zipf_ols": q(per_user_alpha, (0.1, 0.5, 0.9)),
                           "zipf_mle": q(per_user_mle, (0.1, 0.5, 0.9))}

    # 2. inter-arrival
    ts = np.sort(t.astype(float))
    g = np.diff(ts)
    cv, B = burst(g)
    out["iat_global"] = {"mean_s": round(float(g.mean()), 2), "median_s": round(float(np.median(g)), 2),
                         "cv": cv, "B": B, "q": q(g, (0.5, 0.9, 0.99, 0.999, 1.0)),
                         "frac_zero": round(float((g == 0).mean()), 4)}
    hours = ((ts + t0_epoch) % 86400 // 3600).astype(int)
    wd = (((ts + t0_epoch) // 86400) % 7).astype(int)
    hh = np.bincount(hours, minlength=24) / len(hours)
    out["hour_profile_pct"] = [round(float(x) * 100, 2) for x in hh]
    out["peak_to_trough_hour"] = round(float(hh.max() / max(hh.min(), 1e-9)), 1)
    # within-regime CV (generator: weekday 08-18)
    work = (hours >= 9) & (hours < 17)
    gw = np.diff(ts[work])
    gw = gw[gw < 3600]
    out["iat_workhours_cv_B"] = burst(gw)
    if name.startswith("gen"):
        days = (ts // 86400).astype(int)
        dv = np.bincount(days)
        wdv = np.array([dv[d] for d in range(len(dv)) if d % 7 < 5]); wev = np.array([dv[d] for d in range(len(dv)) if d % 7 >= 5])
        out["daily_volume"] = {"weekday_mean": round(float(wdv.mean()), 1), "weekend_mean": round(float(wev.mean()), 1),
                               "weekday_cv": round(float(wdv.std() / wdv.mean()), 3)}
    pu_cv, pu_B = [], []
    ut = defaultdict(list)
    for u, tt in zip(usr[um].tolist(), t[um].tolist()):
        ut[u].append(tt)
    for u, tt in ut.items():
        if len(tt) >= 50:
            c, b = burst(np.diff(np.sort(np.asarray(tt, float))))
            pu_cv.append(c); pu_B.append(b)
    out["iat_per_user"] = {"n": len(pu_cv), "cv": q(pu_cv, (0.1, 0.5, 0.9)), "B": q(pu_B, (0.1, 0.5, 0.9))}

    # 3. per-user activity
    ev = np.array([len(v) for v in ut.values()])
    out["events_per_user"] = {"n_users": int(len(ev)), "q": q(ev), "gini": gini(ev),
                              "cv": round(float(ev.std() / ev.mean()), 3),
                              "max_over_median": round(float(ev.max() / np.median(ev)), 2),
                              "top10pct_share": round(float(np.sort(ev)[::-1][: max(1, len(ev) // 10)].sum() / ev.sum()), 3)}

    # 4. fan-out (benign, analysed users)
    def fan(a, b, mask):
        d = defaultdict(set)
        for x, y in zip(a[mask].tolist(), b[mask].tolist()):
            d[x].add(y)
        return np.array([len(v) for v in d.values()])
    m = benign & um
    out["fanout"] = {
        "dst_per_user": q(fan(usr, dst, m), (0.1, 0.5, 0.9, 1.0)),
        "dev_per_user": q(fan(usr, dev, m), (0.1, 0.5, 0.9, 1.0)),
        "src_per_user": q(fan(usr, src, m), (0.1, 0.5, 0.9, 1.0)),
        "cfg_per_user": q(fan(usr, cfg, m), (0.1, 0.5, 0.9, 1.0)),
        "users_per_dev": q(fan(dev, usr, m), (0.1, 0.5, 0.9, 1.0)),
        "frac_dev_multiuser": round(float((fan(dev, usr, m) > 1).mean()), 3),
        "cfg_per_dev": q(fan(dev, cfg, m), (0.1, 0.5, 0.9, 1.0)),
        "src_per_dev": q(fan(dev, src, m), (0.1, 0.5, 0.9, 1.0)),
        "devs_per_src": q(fan(src, dev, m), (0.1, 0.5, 0.9, 1.0)),
    }
    cc = np.array(sorted(Counter(cfg[benign].tolist()).values(), reverse=True), float)
    p = cc / cc.sum()
    out["ja3_fleet"] = {"distinct_benign": int(len(cc)), "top1_share": round(float(p[0]), 4),
                        "top5_share": round(float(p[:5].sum()), 4),
                        "entropy_bits": round(float(-(p * np.log2(p)).sum()), 3),
                        "configs_covering_90pct": int(np.searchsorted(np.cumsum(p), 0.9) + 1)}

    # 5. novelty
    tr, va = int(N * tr_frac), int(N * (tr_frac + va_frac))
    cold = {}
    for f, a in dict(source=src, config=cfg, device=dev, user=usr, dst=dst).items():
        seen = set(); c = np.zeros(N, bool)
        for i, x in enumerate(a.tolist()):
            c[i] = x not in seen; seen.add(x)
        cold[f] = c
    anyc = cold["source"] | cold["config"] | cold["device"] | cold["user"]
    nov = {}
    for part, (lo, hi) in {"train": (0, tr), "val": (tr, va), "test": (va, N)}.items():
        sl = slice(lo, hi)
        d = {"all": {f: round(float(cold[f][sl].mean()), 4) for f in cold} | {"any4": round(float(anyc[sl].mean()), 4)}}
        for k in sorted(set(types.tolist())):
            mk = types[sl] == k
            if mk.sum() == 0:
                continue
            d[NAMES.get(k, str(k))] = {f: round(float(cold[f][sl][mk].mean()), 4) for f in cold} | {
                "any4": round(float(anyc[sl][mk].mean()), 4), "n": int(mk.sum())}
        nov[part] = d
    out["novelty"] = nov
    # novelty relative to a benign-only history of all previous events in test window (per 1k events)
    # daily new-entity rate (benign) in second half
    daysec = 86400
    dd = ((t - t.min()) // daysec).astype(int)
    half = dd >= np.median(dd)
    out["new_per_day_2nd_half"] = {f: round(float(cold[f][half & benign].sum() / max(1, len(set(dd[half].tolist())))), 3)
                                   for f in ("source", "config", "device", "user")}
    out["new_frac_benign_2nd_half"] = {f: round(float(cold[f][half & benign].mean()), 5) for f in cold}
    return out


def run_gen(seed):
    cfg = dataclasses.replace(TGNConfig(), seed=seed)
    inc = {"compromise": [], "remediate": [], "theft": []}
    Sim = ss.ZTAStreamSimulator
    oc, orr = Sim._compromise, Sim._remediate

    def c(self, m):
        inc["compromise"].append((self.step_count, self.t, m)); return oc(self, m)

    def r(self, m):
        inc["remediate"].append((self.step_count, self.t, m)); return orr(self, m)
    Sim._compromise, Sim._remediate = c, r
    ev_machine = []
    ost = Sim.step

    def st(self):
        e = ost(self)
        return e
    s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
    Sim._compromise, Sim._remediate = oc, orr
    A = lambda x: x.numpy()
    src, cf, dv, us, ds, t, ty = map(A, (s.source, s.config, s.device, s.user, s.dst, s.t, s.types))
    reg = set(range(s.user_lo, s.user_lo + cfg.num_users))
    humans = set(range(s.user_lo + 1, s.user_lo + cfg.num_users))
    o = analyse(f"gen_seed{seed}", src, cf, dv, us, ds, t, ty, cfg.train_frac, cfg.val_frac,
                user_filter=lambda u: u in humans)
    # guests separately
    gm = ~np.isin(us, list(reg))
    o["guest_event_share"] = round(float(gm.mean()), 4)
    o["guest_distinct"] = int(len(set(us[gm].tolist())))
    o["service_user0_share"] = round(float((us == s.user_lo).mean()), 4)
    # kill chain: attack events (etype 1,2,3,5) on the compromised machine's device between compromise and remediation
    # map by time windows; device slot may change by cookie wipe, so count by time window & device set
    comp = inc["compromise"]; rem = inc["remediate"]
    chains = []
    rem_by_m = defaultdict(list)
    for sc, tt, m in rem:
        rem_by_m[m].append((sc, tt))
    for sc, tt, m in comp:
        ends = [x for x in rem_by_m[m] if x[0] >= sc]
        if not ends:
            continue
        esc, ett = ends[0]
        chains.append((sc, esc, tt, ett))
    lens_ev = []; lens_t = []; steps = []
    kc = np.isin(ty, [1, 2, 3, 5])
    for sc, esc, tt, ett in chains:
        steps.append(esc - sc)
        lens_t.append((ett - tt) / 3600)
    o["intrusions"] = {"n_compromise": len(comp), "n_remediated": len(chains),
                       "chain_duration_hours": q(lens_t, (0.1, 0.5, 0.9, 1.0)),
                       "chain_duration_stream_events": q(steps, (0.1, 0.5, 0.9))}
    # theft incidents: group etype4 by (device, user)
    th = defaultdict(int)
    for d_, u_ in zip(dv[ty == 4].tolist(), us[ty == 4].tolist()):
        th[(d_, u_)] += 1
    o["theft_incidents_approx"] = {"n": len(th), "events_per_incident": q(list(th.values()), (0.1, 0.5, 0.9, 1.0))}
    # attack events per intrusion by class
    o["attack_events_per_intrusion"] = round(float(kc.sum() / max(1, len(comp))), 2)
    # per-day attack starts
    o["intrusions_per_day"] = round(len(comp) / o["span_days"], 3)
    o["thefts_per_day"] = round(len(th) / o["span_days"], 3)
    o["users"] = cfg.num_users; o["devices"] = cfg.num_devices
    return o


def run_pico():
    sys.path.insert(0, "/app/tests")
    from tests.datasets.picodomain import load_picodomain_stream
    d = load_picodomain_stream("/data/logs", "/data/Red Log.xlsx")
    A = lambda x: x.numpy()
    src, cf, dv, us, ds, t, ty = map(A, (d.source_nodes, d.config_nodes, d.device_nodes, d.user, d.dst, d.t, d.types))
    keys = d.keys
    real = lambda u: not str(keys[u]).startswith("usr:none")
    # t is relative to t0; first event epoch
    import glob as _g, json as _j
    from datetime import datetime
    o = analyse("picodomain", src, cf, dv, us, ds, t, ty, 0.7, 0.1, user_filter=real, t0_epoch=0)
    o["user_keys"] = sorted({str(keys[u]) for u in set(us.tolist())})[:40]
    o["cfg_keys_n_none"] = int(sum(str(keys[c]).startswith("cfg:none") for c in set(cf.tolist())))
    return o


if __name__ == "__main__":
    which = sys.argv[1:] or ["gen2000", "pico"]
    for w in which:
        o = run_pico() if w == "pico" else run_gen(int(w[3:]))
        path = f"/app/tasks/tmp/dist_check_{w}.json"
        with open(path, "w") as fh:
            json.dump(o, fh, indent=1, default=str)
        print(json.dumps(o, default=str)[:3000])
