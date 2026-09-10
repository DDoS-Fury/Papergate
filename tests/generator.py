"""Async event generator for the live API test client (v4 schema).

Thin wrapper around :class:`graphagate.data.stream_synthetic.ZTAStreamSimulator` —
the SAME simulator the offline training stream is built from, so the live test
traffic follows the trained baseline by construction (no duplicated behaviour
model, as the previous copy of the generator logic was).

``warmup_steps`` replays the exact training-time event sequence (same seed) before
yielding, so the live stream continues seamlessly from where training stopped:
same clock, same kill-chain state, same device admission.
"""

import asyncio

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import ZTAStreamSimulator, stream_kwargs_from_cfg


async def event_generator(seed=None, warmup_steps=None, cfg: TGNConfig = TGNConfig(), omit_device: bool = False):
    # All generator parameters come from cfg via the shared mapping, so the live stream
    # is guaranteed to live in the same entity space as the trained checkpoint.
    sim = ZTAStreamSimulator(
        **stream_kwargs_from_cfg(cfg),
        admission_horizon=warmup_steps if warmup_steps else None,
        seed=seed,
    )
    if warmup_steps is None:
        warmup_steps = cfg.num_events if seed == cfg.seed else 0
    for _ in range(warmup_steps):
        sim.step()
    if warmup_steps:
        print(f"[Generator] Starting seamlessly at t={sim.t} (after {warmup_steps} warmup steps)")

    nf = sim.node_features
    while True:
        ev = sim.step()
        event_dict = {
            "key_user": ev["key_user"],
            "key_device": ev["key_device"],
            "key_source": ev["key_source"],
            "key_config": ev["key_config"],
            "key_dst": ev["key_dst"],
            "timestamp": int(ev["t"]),
            "features": [float(v) for v in ev["features"]],
            "user_feat": nf[ev["user"]].tolist(),
            "device_feat": nf[ev["device"]].tolist(),
            "dst_feat": nf[ev["dst"]].tolist(),
            "label": ev["label"],
            "type": ev["etype"],
        }
        if omit_device:
            event_dict.pop("key_device", None)
            
        yield event_dict
        await asyncio.sleep(0)
