"""Diagnostic copy of tests/stream_client.py: logs (etype, score, threshold, verdict) per event.

--no-prefix keeps the trained entity keys (warm memory) instead of the prod_ cold-start prefix.
"""
import argparse, asyncio, json, sys, time
import aiohttp

sys.path.insert(0, "/app/tests")
from generator import event_generator


async def main(host, port, duration, prefix, out):
    gen = event_generator(seed=42)
    rows = []
    async with aiohttp.ClientSession() as s:
        t0 = time.time()
        while time.time() - t0 < duration:
            ev = await anext(gen)
            label, etype = ev.pop("label"), ev.pop("type")
            if prefix:
                for k in ("key_user", "key_device", "key_source", "key_config"):
                    if ev.get(k) is not None:
                        ev[k] = f"prod_{ev[k]}"
            async with s.post(f"http://{host}:{port}/infer", json=ev) as r:
                d = await r.json()
            rows.append({"etype": etype, "label": label, "score": d["anomaly_score"],
                         "thr": d["threshold"], "flag": d["is_anomaly"]})
            feats = ev.get("features", [])
            dirty = bool(feats and (feats[0] <= 0.5 or any(x > 0.5 for x in feats[1:4])))
            if etype not in (1, 6) and not dirty:
                async with s.post(f"http://{host}:{port}/update", json=ev) as r:
                    pass
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="serve-tgn"); p.add_argument("--port", type=int, default=8088)
    p.add_argument("--duration", type=int, default=120)
    p.add_argument("--no-prefix", action="store_true"); p.add_argument("--out", required=True)
    a = p.parse_args()
    asyncio.run(main(a.host, a.port, a.duration, not a.no_prefix, a.out))
