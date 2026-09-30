"""The live event generator replays the training stream before yielding new events.

    pytest tests/test_generator_replay.py
"""

import asyncio
import dataclasses

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import (
    ZTAStreamSimulator,
    generate_streaming_data,
    stream_kwargs_from_cfg,
)
from generator import event_generator, make_simulator

CFG = dataclasses.replace(TGNConfig(), num_events=3000)


def _training_simulator(cfg):
    """Simulator exactly as generate_streaming_data builds it (reference, no helper)."""
    kw = stream_kwargs_from_cfg(cfg)
    return ZTAStreamSimulator(admission_horizon=kw.pop("num_events"), **kw)


def test_replay_simulator_reproduces_training_stream():
    stream = generate_streaming_data(**stream_kwargs_from_cfg(CFG))
    sim = make_simulator(CFG.seed, CFG, replay=True)
    events = [sim.step() for _ in range(CFG.num_events)]
    assert [e["t"] for e in events] == stream.t.tolist()
    assert [e["dst"] for e in events] == stream.dst.tolist()
    assert [e["etype"] for e in events] == stream.types.tolist()


def test_event_generator_resumes_where_training_stopped():
    ref = _training_simulator(CFG)
    for _ in range(CFG.num_events):
        ref.step()
    expected = ref.step()

    first = asyncio.run(event_generator(seed=CFG.seed, cfg=CFG).__anext__())
    assert first["timestamp"] == expected["t"]
    assert first["key_user"] == expected["key_user"]
    assert first["key_dst"] == expected["key_dst"]
    assert first["features"] == expected["features"]
