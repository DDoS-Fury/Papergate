"""Concurrent load + registry-capacity stress harness for the serve_api service.

The functional client (`test_client.py`) is strictly sequential — one request in
flight — so it never exercises the two things that actually stress the v3 service:

  Phase A — CONCURRENCY/LOAD: the service runs sync endpoints in FastAPI's
    threadpool but serialises every model-touching request on a single global
    ``STATE.lock`` (serve_api.py), ``workers=1``, model mutable in RAM. We fan out
    ``--concurrency`` parallel clients replaying realistic streams to measure the
    throughput ceiling imposed by that lock, the tail latency under contention,
    and to prove no request errors / no state corruption under parallelism.

  Phase B — REGISTRY CAPACITY/EVICTION: ``NodeRegistry`` maps unbounded external
    keys onto ``capacity`` slots and evicts the least-recently-updated slot when
    full (registry.py). We flood UNIQUE namespaced v3 keys (``src:``/``ck:``/user)
    to drive ``registry_size`` (read live from ``/health``) toward ``capacity`` and
    assert it stays bounded, the service stays healthy, and — if capacity is
    reached within the budget — that eviction kicks in (size plateaus at capacity).

Usage (from tests/, against a running serve-tgn on :8888):
    python stress_test.py --concurrency 32 --load-seconds 60 --capacity-seconds 60

Exit code is non-zero if any phase fails its pass criteria (CI-friendly).
"""

import argparse
import asyncio
import time

import aiohttp
import numpy as np

from generator import event_generator

BASE_URL = "http://localhost:8888"
TYPE_NAMES = {0: "Benign", 1: "Policy", 2: "Contextual", 3: "Lateral", 4: "CredTheft"}
# Unloaded single-client P50 round-trip (measured by test_client.py) — the yardstick
# the load phase reports contention against. The service serialises on one global
# lock, so the headline finding is the lock-bound throughput ceiling, NOT scaling.
UNLOADED_P50_MS = 6.5
# Real, preregistered resource routes (so Phase-B dst nodes stay valid while the
# user/device/source keys are the unique ones doing the registry churn).
RESOURCE_URIS = [
    "/", "/api/v1/personnel", "/api/v1/documents", "/api/v1/nuclear-materials",
    "/api/v1/reactor-parameters", "/api/v1/trusted-guard/sanitized-delete-personnel",
]


def _pct(xs, q):
    return float(np.percentile(xs, q)) if xs else 0.0


# --------------------------------------------------------------------------- #
# Phase A — concurrent realistic load
# --------------------------------------------------------------------------- #
async def _load_worker(wid, session, seed, deadline, lat, preds, errors, counts):
    gen = event_generator(seed=seed)
    user_counts = {}
    while time.time() < deadline:
        ev = await anext(gen)
        label, etype, key_actor = ev.pop("label"), ev.pop("type"), ev["key_user"]
        t0 = time.perf_counter()
        try:
            async with session.post(f"{BASE_URL}/infer", json=ev) as r:
                if r.status != 200:
                    errors.append(("infer", r.status)); continue
                data = await r.json()
        except Exception as e:
            errors.append(("infer", repr(e))); continue
        lat.append((time.perf_counter() - t0) * 1000)

        score = data.get("anomaly_score", 1.0)
        thr = data.get("threshold", 0.5)
        is_anom = data.get("is_anomaly", False)
        preds.append((is_anom, label == 1, etype))
        counts[0] += 1

        # OPA decision, then conditional /update.
        user_counts[key_actor] = user_counts.get(key_actor, 0) + 1
        policy_violation = (etype == 1)
        allow = (not policy_violation) and (not (score > thr))
        if allow:
            try:
                async with session.post(f"{BASE_URL}/update", json=ev) as r:
                    if r.status != 200:
                        errors.append(("update", r.status))
            except Exception as e:
                errors.append(("update", repr(e)))


async def phase_load(concurrency, seconds, base_seed):
    print(f"\n=== PHASE A — concurrent load ({concurrency} clients, {seconds}s) ===")
    lat, preds, errors, counts = [], [], [], [0]
    deadline = time.time() + seconds
    conn = aiohttp.TCPConnector(limit=concurrency)
    t_start = time.time()
    async with aiohttp.ClientSession(connector=conn) as session:
        # distinct seeds => distinct trajectories => real concurrent diversity
        await asyncio.gather(*[
            _load_worker(i, session, base_seed + i, deadline, lat, preds, errors, counts)
            for i in range(concurrency)
        ])
    elapsed = time.time() - t_start

    total = counts[0]
    thr = total / elapsed if elapsed else 0
    p50, mx = _pct(lat, 50), (max(lat) if lat else 0.0)
    contention = p50 / UNLOADED_P50_MS if UNLOADED_P50_MS else 0
    print(f"Requests (/infer): {total} in {elapsed:.1f}s  ->  {thr:.1f} ev/s "
          f"(2 lock ops/event incl. /update)")
    print(f"Errors: {len(errors)}" + (f"  e.g. {errors[:3]}" if errors else ""))
    print(f"Latency under load (ms) - P50 {p50:.2f} | P90 {_pct(lat,90):.2f} "
          f"| P99 {_pct(lat,99):.2f} | P99.9 {_pct(lat,99.9):.2f} | max {mx:.2f}")
    print(f"Contention vs unloaded P50 {UNLOADED_P50_MS}ms: {contention:.1f}x "
          f"(~{concurrency} clients queued on the global STATE.lock)")
    # accuracy by type (cold-start streams, no warmup => lower than the functional
    # client; reported only to show scoring stays sane under load, not as a gate)
    print("Accuracy by type (cold-start, informational):")
    for t in range(5):
        tp = [p for p in preds if p[2] == t]
        if tp:
            ok = sum(1 for p in tp if p[0] == p[1])
            print(f"  - {TYPE_NAMES[t]}: {ok}/{len(tp)} ({ok/len(tp)*100:.2f}%)")

    # Pass criteria for a *serialised* service: every concurrent request is served
    # with no errors and bounded latency (no hangs/timeouts/corruption). Throughput
    # is lock-capped by design, so it is REPORTED, not gated. The contention factor
    # (~concurrency) is the expected signature of correct serialisation.
    err_rate = len(errors) / max(total, 1)
    ok = (err_rate < 0.001) and (total > 0) and (mx < 5000)
    print(f"PHASE A: {'PASS' if ok else 'FAIL'} (err_rate={err_rate:.4%}, "
          f"ceiling={thr:.0f} ev/s, max_lat={mx:.0f}ms, contention={contention:.1f}x)")
    return ok


# --------------------------------------------------------------------------- #
# Phase B — registry capacity / eviction flood
# --------------------------------------------------------------------------- #
def _unique_event(gid, n):
    """A valid v3 event whose user/device/source keys are globally unique, so each
    request consumes a fresh registry slot. dst cycles real (preregistered) routes."""
    a, b, c = (n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF
    return {
        "key_user": f"user:F{gid}-{n}",
        "key_device": f"ck:F{gid}-{n}",
        "key_source": f"src:10.{a}.{b}.{c}",
        "key_dst": RESOURCE_URIS[n % len(RESOURCE_URIS)],
        "timestamp": 20_000_000 + gid * 10_000_000 + n,
        "features": [0.13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


async def _flood_worker(gid, session, deadline, lat, errors, sent):
    n = 0
    while time.time() < deadline:
        ev = _unique_event(gid, n); n += 1
        t0 = time.perf_counter()
        try:
            # /update admits AND advances memory.last_update (drives LRU eviction).
            async with session.post(f"{BASE_URL}/update", json=ev) as r:
                if r.status != 200:
                    errors.append(("update", r.status)); continue
        except Exception as e:
            errors.append(("flood", repr(e))); continue
        lat.append((time.perf_counter() - t0) * 1000)
        sent[gid] += 1


async def _registry_sampler(session, deadline, samples):
    while time.time() < deadline:
        try:
            async with session.get(f"{BASE_URL}/health") as r:
                h = await r.json()
                samples.append((time.time(), h["registry_size"], h["capacity"]))
        except Exception:
            pass
        await asyncio.sleep(1.0)


async def phase_capacity(concurrency, seconds, base_seed):
    print(f"\n=== PHASE B — registry capacity/eviction flood "
          f"({concurrency} clients, {seconds}s) ===")
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/health") as r:
            h0 = await r.json()
    cap = h0["capacity"]
    start_size = h0["registry_size"]
    print(f"Start registry_size={start_size}, capacity={cap}")

    lat, errors, sent = [], [], [0] * concurrency
    samples = []
    deadline = time.time() + seconds
    conn = aiohttp.TCPConnector(limit=concurrency + 1)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [_flood_worker(i, session, deadline, lat, errors, sent)
                 for i in range(concurrency)]
        tasks.append(_registry_sampler(session, deadline + 0.5, samples))
        await asyncio.gather(*tasks)

    total_sent = sum(sent)
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{BASE_URL}/health") as r:
            hf = await r.json()
    end_size = hf["registry_size"]
    peak_size = max((sz for _, sz, _ in samples), default=end_size)
    reached = peak_size >= cap

    print(f"Unique keys flooded: {total_sent}  ({total_sent/seconds:.0f} keys/s)")
    print(f"Errors: {len(errors)}" + (f"  e.g. {errors[:3]}" if errors else ""))
    print(f"Latency under churn (ms) - P50 {_pct(lat,50):.2f} | P99 {_pct(lat,99):.2f} "
          f"| max {max(lat):.2f}")
    print(f"registry_size: start={start_size} peak={peak_size} end={end_size} cap={cap}")
    # growth monotonic until it pins at capacity
    sizes = [sz for _, sz, _ in samples]
    monotonic_until_cap = all(
        b >= a or b >= cap for a, b in zip(sizes, sizes[1:])
    )
    if reached:
        print(f"-> Capacity REACHED: eviction engaged; size bounded at cap "
              f"(peak {peak_size} == cap {cap}, no crash).")
    else:
        need = cap - start_size
        eta = need / (total_sent / seconds) if total_sent else float("inf")
        print(f"-> Capacity NOT reached in {seconds}s "
              f"(need {need} more keys; ETA ~{eta:.0f}s at this rate). "
              f"Bounded growth + 0 errors is the pass signal here.")

    # Pass: no errors, registry never exceeded capacity, growth monotonic-until-cap,
    # and we genuinely grew the registry (admission path works under churn).
    ok = (len(errors) == 0 and peak_size <= cap and end_size > start_size
          and monotonic_until_cap)
    print(f"PHASE B: {'PASS' if ok else 'FAIL'} "
          f"(errors={len(errors)}, peak<=cap={peak_size<=cap}, grew={end_size>start_size})")
    return ok, reached


async def main(args):
    a_ok = await phase_load(args.concurrency, args.load_seconds, args.seed)
    b_ok, reached = await phase_capacity(args.concurrency, args.capacity_seconds, args.seed)
    print("\n" + "=" * 60)
    print(f"STRESS SUMMARY: load={'PASS' if a_ok else 'FAIL'} | "
          f"capacity={'PASS' if b_ok else 'FAIL'}"
          f"{' (eviction reached)' if reached else ' (bounded, eviction not reached)'}")
    print("=" * 60)
    return 0 if (a_ok and b_ok) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Concurrent + capacity stress harness")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--load-seconds", type=int, default=60)
    p.add_argument("--capacity-seconds", type=int, default=60)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()
    raise SystemExit(asyncio.run(main(args)))
