"""Async event generator for the live API test client (v2 schema).

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
from graphagate.data.stream_synthetic import ZTAStreamSimulator


async def event_generator(seed=None, warmup_steps=None, cfg: TGNConfig = TGNConfig(), omit_device: bool = False):
    sim = ZTAStreamSimulator(
        num_users=cfg.num_users,
        num_devices=cfg.num_devices,
        num_sources=cfg.num_sources,
        num_resources=cfg.num_resources,
        num_wipe_slots=cfg.num_wipe_slots,
        num_theft_slots=cfg.num_theft_slots,
        benign_explore_prob=cfg.benign_explore_prob,
        p_roam=cfg.p_roam,
        p_shared_device=cfg.p_shared_device,
        p_cookie_wipe=cfg.p_cookie_wipe,
        p_cred_theft=cfg.p_cred_theft,
        admission_horizon=warmup_steps if warmup_steps else None,
        seed=seed,
        use_resource_risk=cfg.use_resource_risk,
        use_source_internal=cfg.use_source_internal,
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
            "key_dst": ev["key_dst"],
            "timestamp": int(ev["t"]),
            "features": [float(v) for v in ev["features"]],
            "src_feat": nf[ev["device"]].tolist(),
            "dst_feat": nf[ev["dst"]].tolist(),
            "label": ev["label"],
            "type": ev["etype"],
        }
        if omit_device:
            event_dict.pop("key_device", None)
            
        yield event_dict
        await asyncio.sleep(0)
