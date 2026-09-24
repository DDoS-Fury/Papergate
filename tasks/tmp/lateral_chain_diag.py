"""Struttura delle campagne laterali sul seed 2000: quanti eventi per campagna, e
quanto tempo passa tra il recon e i laterali che lo seguono sulla stessa macchina.

Serve a dimensionare (a) una metrica a livello di campagna e (b) precursor_half_life.
"""
import numpy as np
from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import generate_streaming_data, stream_kwargs_from_cfg

cfg = TGNConfig(seed=2000)
s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))

N = s.t.numel()
train_end = int(N * cfg.train_frac)
val_end = train_end + int(N * cfg.val_frac)
t = s.t.numpy().astype(np.int64)
ty = s.types.numpy()
dev = s.device.numpy()

for name, lo, hi in (("val", train_end, val_end), ("test", val_end, N)):
    sl = slice(lo, hi)
    tt, tty, tdev = t[sl], ty[sl], dev[sl]
    lat = np.where(tty == 3)[0]
    print(f"\n=== finestra {name} ({hi-lo} eventi) ===")
    print(f"eventi laterali: {lat.size}")

    # Campagna = eventi laterali della STESSA macchina separati da meno di 7 giorni.
    # Va raggruppato per device e non per contiguita' nello stream: piu' macchine sono
    # compromesse insieme, quindi i loro eventi laterali si interlacciano.
    camps = []
    for m in np.unique(tdev[lat]):
        idx = lat[tdev[lat] == m]
        cur = [idx[0]]
        for i in idx[1:]:
            if tt[i] - tt[cur[-1]] < 7 * 86400:
                cur.append(i)
            else:
                camps.append(cur); cur = [i]
        camps.append(cur)
    sizes = np.array([len(c) for c in camps])
    print(f"campagne: {sizes.size} | eventi/campagna: min={sizes.min()} mediana={np.median(sizes):.0f} max={sizes.max()} media={sizes.mean():.1f}")

    # Durata della campagna e gap tra laterali consecutivi.
    dur = np.array([tt[c[-1]] - tt[c[0]] for c in camps]) / 3600.0
    gaps = np.concatenate([np.diff(tt[c]) for c in camps if len(c) > 1]) / 3600.0
    print(f"durata campagna (h): mediana={np.median(dur):.1f} p90={np.percentile(dur,90):.1f} max={dur.max():.1f}")
    if gaps.size:
        print(f"gap tra laterali consecutivi (h): mediana={np.median(gaps):.2f} p90={np.percentile(gaps,90):.2f}")

    # Δt dall'ultimo evento di recon (type 2) sulla stessa macchina al primo laterale.
    d_recon = []
    for c in camps:
        i0 = c[0]
        prev = np.where((tdev[:i0] == tdev[i0]) & (tty[:i0] == 2))[0]
        d_recon.append((tt[i0] - tt[prev[-1]]) / 3600.0 if prev.size else np.nan)
    d_recon = np.array(d_recon)
    ok = ~np.isnan(d_recon)
    print(f"campagne con recon precedente sulla stessa macchina: {ok.sum()}/{len(camps)}")
    if ok.any():
        print(f"  Δt recon→1o laterale (h): mediana={np.median(d_recon[ok]):.2f} p90={np.percentile(d_recon[ok],90):.2f}")
        for hl_h in (10 / 60, 1, 6, 24, 72):
            decay = 0.5 ** (d_recon[ok] / hl_h)
            print(f"  half_life={hl_h*60:7.0f} min -> boost mediano = {1 + 2.0*np.median(decay):.3f}")
