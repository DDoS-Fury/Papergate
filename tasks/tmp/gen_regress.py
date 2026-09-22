"""Digest of generated streams, to diff the generator before/after a change."""
import dataclasses, hashlib, json, sys

import torch

from graphagate.config import TGNConfig
from graphagate.data.stream_synthetic import V4_KNOBS, generate_streaming_data, stream_kwargs_from_cfg


def digest(s):
    h = hashlib.sha256()
    for name in ("source", "config", "device", "user", "dst", "t", "msg", "y", "types", "scenario"):
        h.update(getattr(s, name).numpy().tobytes())
    return h.hexdigest()


out = {}
for tag, extra in (("v5", {}), ("v4", V4_KNOBS)):
    for seed in (1000, 2000):
        cfg = dataclasses.replace(TGNConfig(), seed=seed, num_events=60000, **extra)
        s = generate_streaming_data(**stream_kwargs_from_cfg(cfg))
        out[f"{tag}_s{seed}"] = digest(s)
        print(tag, seed, out[f"{tag}_s{seed}"][:16], torch.bincount(s.types).tolist(), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
