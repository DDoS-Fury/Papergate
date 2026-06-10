"""Map the LANL 'Comprehensive Multi-Source Cyber-Security Events' auth logs to a ZTA stream.

External-validity test for the lateral-movement detector. LANL publishes 58 days of host
authentication events with a labelled **red-team** sub-stream — the canonical public ground
truth for lateral movement (an attacker reusing valid credentials to hop host→host).

Download (free, registration required):
  https://csr.lanl.gov/data/cyber1/   →   auth.txt.gz  (~12 GB) and redteam.txt  (~80 KB)

The files are NOT committed; pass their paths to :func:`load_lanl_stream`. The loader streams
``auth.txt.gz`` line-by-line (constant memory), keeps a time window around the red-team
activity, and down-samples benign traffic to a tractable size while keeping **every** red-team
event.

Schema mapping → :class:`graphagate.train_tgn.StreamData` (the synthetic stream's schema):
  auth.txt: time, srcUser@dom, dstUser@dom, srcComp, dstComp, authType, logonType, authOrient, success
  - src node  = source computer, dst node = destination computer (host-to-host = the lateral graph)
  - t         = integer second
  - msg[7]    = [ja3=1, snort=0, s1=0, s2=0, method, roleVal=0, clrVal=0] — the alarm columns are held CLEAN
                because LANL auth carries no TLS/IDS signal and red-team lateral movement is
                signal-clean by construction; only ``method`` carries auth metadata (the auth
                orientation code). This keeps the rule baseline correctly blind to lateral and
                makes the temporal/relational pattern the sole discriminator — the honest test.
  - y / types = 1 / 3 (lateral) iff (time,user,srcComp,dstComp) is a red-team event, else 0 / 0
  - node_features[16] = neutral (LANL has no roles/clearance): zeros with trust slot 14 = 1.0,
                so the model leans on memory + interaction history, not static priors.
  - structural-negative range = the whole computer id-space (dst is any computer).
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import torch

from graphagate.train_tgn import StreamData

# Auth-orientation → small integer code for the ``method`` slot (faithful, non-leaking).
_ORIENT_CODE = {"LogOn": 0, "LogOff": 1, "TGS": 2, "TGT": 3}


def _open(path: str):
    """Open a plain or gzipped text file transparently."""
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _load_redteam(redteam_path: str):
    """Return ``(set of (t,user,src,dst) keys, min_time, max_time)`` from redteam.txt."""
    keys = set()
    tmin, tmax = None, None
    with _open(redteam_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            t_str, user, src, dst = line.split(",")
            t = int(t_str)
            keys.add((t, user, src, dst))
            tmin = t if tmin is None else min(tmin, t)
            tmax = t if tmax is None else max(tmax, t)
    if not keys:
        raise RuntimeError(f"No red-team events parsed from {redteam_path}")
    return keys, tmin, tmax


def load_lanl_stream(
    auth_path: str,
    redteam_path: str,
    *,
    max_events: int = 200_000,
    benign_stride: int = 16,
    window_pad: int = 3600,
    window: tuple[int, int] | None = None,
) -> StreamData:
    """Stream-parse LANL auth into a :class:`StreamData` for ``train_tgn(dataset=...)``.

    ``max_events`` caps the kept events; ``benign_stride`` keeps 1-in-N benign auths (red-team
    events are always kept); the time ``window`` defaults to the red-team span padded by
    ``window_pad`` seconds on each side. Node indices are assigned in first-seen (time) order.
    """
    redteam, rt_min, rt_max = _load_redteam(redteam_path)
    if window is None:
        window = (rt_min - window_pad, rt_max + window_pad)
    w_lo, w_hi = window

    comp_to_idx: dict[str, int] = {}
    keys: list[str] = []

    def _idx(comp: str) -> int:
        i = comp_to_idx.get(comp)
        if i is None:
            i = len(keys)
            comp_to_idx[comp] = i
            keys.append(comp)
        return i

    src_l, dst_l, t_l, msg_l, y_l, ty_l = [], [], [], [], [], []
    benign_seen = 0
    n_lateral = 0

    with _open(auth_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            t = int(parts[0])
            if t < w_lo:
                continue
            if t > w_hi or len(t_l) >= max_events:
                break  # auth.txt is time-ordered: past the window/cap, we're done.

            user, src_comp, dst_comp = parts[1], parts[3], parts[4]
            orient, success = parts[7], parts[8]
            is_rt = (t, user, src_comp, dst_comp) in redteam
            if not is_rt:
                # Down-sample benign by stride to stay tractable; red-team is always kept.
                if benign_seen % benign_stride != 0:
                    benign_seen += 1
                    continue
                benign_seen += 1

            # Alarm columns held clean (signal-clean premise); auth orientation in ``method``.
            method = float(_ORIENT_CODE.get(orient, 4))
            msg_l.append([1.0, 0.0, 0.0, 0.0, method, 0.0, 0.0])
            src_l.append(_idx(src_comp))
            dst_l.append(_idx(dst_comp))
            t_l.append(t)
            y_l.append(1 if is_rt else 0)
            ty_l.append(3 if is_rt else 0)
            n_lateral += int(is_rt)

    if not t_l:
        raise RuntimeError(
            "No events parsed in the window — check the paths / window / that auth.txt is sorted."
        )
    if n_lateral == 0:
        raise RuntimeError("Window contains no red-team (lateral) events; widen window/max_events.")

    num_nodes = len(keys)
    node_features = torch.zeros(num_nodes, 16, dtype=torch.float)
    node_features[:, 14] = 1.0  # trust slot default (matches the synthetic generator)

    print(
        f"[lanl] events={len(t_l)} computers={num_nodes} lateral={n_lateral} "
        f"benign={len(t_l) - n_lateral} window={window} stride={benign_stride}"
    )
    return StreamData(
        src=torch.tensor(src_l, dtype=torch.long),
        dst=torch.tensor(dst_l, dtype=torch.long),
        t=torch.tensor(t_l, dtype=torch.long),
        msg=torch.tensor(msg_l, dtype=torch.float),
        y=torch.tensor(y_l, dtype=torch.long),
        types=torch.tensor(ty_l, dtype=torch.long),
        node_features=node_features,
        keys=keys,
        num_nodes=num_nodes,
        neg_lo=0,
        neg_num=num_nodes,
    )


if __name__ == "__main__":
    # Quick inspection: python tests/datasets/lanl_auth.py <auth_path> <redteam_path> [max_events]
    auth, red = sys.argv[1], sys.argv[2]
    max_ev = int(sys.argv[3]) if len(sys.argv) > 3 else 200_000
    data = load_lanl_stream(auth, red, max_events=max_ev)
    print("StreamData ready:", data.src.shape, "lateral frac =",
          float((data.types == 3).float().mean()))
