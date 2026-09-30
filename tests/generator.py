"""Async event generator for the live API test client.

Thin wrapper around :class:`graphagate.data.stream_synthetic.ZTAStreamSimulator`, the
same simulator the offline training stream is built from, so live traffic follows the
trained baseline by construction.

With the training seed, ``warmup_steps`` defaults to ``cfg.num_events``: the training
sequence is replayed before yielding, so the live stream continues where training stopped
(same clock, kill-chain state and device admission).
"""

import asyncio

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import ZTAStreamSimulator, stream_kwargs_from_cfg


def make_simulator(seed=None, cfg: TGNConfig | None = None, replay: bool = False) -> ZTAStreamSimulator:
    """Simulator in the checkpoint's entity space.

    ``replay=True`` uses the training admission horizon (as generate_streaming_data), so
    stepping it reproduces the training stream; otherwise every entity is admitted at once.
    """
    cfg = cfg or TGNConfig()
    kw = stream_kwargs_from_cfg(cfg)
    horizon = kw.pop("num_events")
    kw.update(admission_horizon=horizon if replay else None, seed=seed)
    return ZTAStreamSimulator(**kw)


async def event_generator(seed=None, warmup_steps=None, cfg: TGNConfig | None = None, omit_device: bool = False):
    """Yield API event dicts forever; ``warmup_steps`` events are generated and discarded first."""
    cfg = cfg or TGNConfig()
    if warmup_steps is None:
        warmup_steps = cfg.num_events if seed == cfg.seed else 0
    sim = make_simulator(seed, cfg, replay=bool(warmup_steps))
    for _ in range(warmup_steps):
        sim.step()
    if warmup_steps:
        print(f"[Generator] Resuming at t={sim.t} (after {warmup_steps} warmup steps)")

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
