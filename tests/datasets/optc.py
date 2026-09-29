"""Map the LMDEval OpTC flow list (``scripts/optc_extract.py build``) to a ZTA stream.

Input: ``optc_flows.csv.gz`` with one deduplicated internal FLOW START per row, the LMDEval
columns (``timestamp, src, dst, src_port, dst_port, proto, label``) plus ``label_lm``,
``timestamp_abs`` and the extra eCAR fields.

Schema mapping -> :class:`graphagate.train_tgn.StreamData`:
  * ``nodes="enriched"`` (default): the 5-node chain the model is built for. Source = source IP,
    device = source host, user = eCAR ``principal``, config = ``image_path``, resource =
    destination host. Empty ``principal`` / ``image_path`` map to a per-host sentinel (never
    one shared "unknown" node).
  * ``nodes="lmdeval"``: only the information the LMDEval detectors get, for the equal-information
    comparison with Tab. 6. The source host is the actor (USER role), the destination host is
    the resource; no binding edges, as for LANL.
  * ``msg[7] = [ja3=1, 0, 0, 0, method, 0, 0]``: alarm columns clean (OpTC has no sensor
    signal, so the commit gate commits everything); ``method`` is a destination-port class.
  * ``types``: 3 = "Lateral movement", 7 = the other red-team events ("Other"), 0 = benign.
    ``y`` = 1 for both, so neither is trained on. LMDEval's LM-only metric counts Other as a
    negative: compute it from the raw scores, not from ``per_type``.
  * ``node_features``: zeros with the trust slot 14 = 1.0 (no roles or clearances in OpTC).

The split is by time: training ends at ``val_start``, validation (benign calibration) ends at
``test_start``. The returned fractions feed ``TGNConfig.train_frac`` / ``val_frac``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import torch

from graphagate.train_tgn import StreamData

T_BENIGN, T_LATERAL, T_OTHER = 0, 3, 7  # 7: not one of train_tgn's named types

# eCAR timestamps are seconds since 1970-01-01T00:00-04:00 (LMDEval's origin).
_ORIGIN = dt.datetime.fromisoformat("1970-01-01T00:00:00-04:00")

# Destination-port classes for the ``method`` slot (services seen on the internal flows).
_PORT_CLASS = {445: 1, 139: 2, 135: 3, 88: 4, 389: 5, 636: 5, 53: 6, 443: 7, 80: 8,
               3389: 9, 5985: 10, 5986: 10, 22: 11, 8530: 12, 137: 13, 138: 13}


def to_ts(iso: str) -> float:
    """ISO time (with offset) -> the ``timestamp_abs`` scale of the flow list."""
    return (dt.datetime.fromisoformat(iso) - _ORIGIN).total_seconds()


def port_class(port: int) -> float:
    if port in _PORT_CLASS:
        return float(_PORT_CLASS[port])
    return 14.0 if port >= 49152 else 15.0  # ephemeral / other registered


def pair_rarity(df) -> np.ndarray:
    """Baseline score 1 / (1 + earlier events with the same src -> dst pair), in stream order."""
    return 1.0 / (1.0 + df.groupby(["src", "dst"], sort=False).cumcount().to_numpy(dtype=np.float64))


def load_optc_stream(path: str, *, val_start: float, test_start: float, nodes: str = "enriched",
                     t_min: float | None = None, t_max: float | None = None):
    """Returns ``(StreamData, train_frac, val_frac, df)``; ``df`` is the kept rows, stream order."""
    import pandas as pd

    df = pd.read_csv(path, dtype={"principal": str, "image_path": str}, keep_default_na=False)
    if t_min is not None:
        df = df[df["timestamp_abs"] >= t_min]
    if t_max is not None:
        df = df[df["timestamp_abs"] < t_max]
    df = df.sort_values(["timestamp_abs", "id"], kind="stable").reset_index(drop=True)
    if not (df["timestamp_abs"].min() < val_start <= test_start < df["timestamp_abs"].max()):
        raise ValueError("val_start/test_start fuori dall'intervallo dei dati")

    n = len(df)
    n_train = int((df["timestamp_abs"] < val_start).sum())
    n_val = int((df["timestamp_abs"] < test_start).sum()) - n_train
    n_pos_fit = int(df["label"].iloc[:n_train + n_val].sum())
    if n_pos_fit:
        print(f"[optc] ATTENZIONE: {n_pos_fit} eventi red-team prima di test_start (esclusi dal training)")

    keys: list[str] = []
    index: dict[str, int] = {}

    def _range(names) -> tuple[int, int]:
        lo = len(keys)
        for k in names:
            if k not in index:
                index[k] = len(keys)
                keys.append(k)
        return lo, len(keys) - lo

    src_host, dst_host = df["src"].astype(str), df["dst"].astype(str)
    if nodes == "lmdeval":
        # One node per host, used as actor and as resource (the host graph of LANL/Euler).
        res_lo, res_num = _range(sorted("res:" + h for h in set(src_host) | set(dst_host)))
        usr_lo, usr_num = res_lo, res_num
        user = np.array([index["res:" + h] for h in src_host])
        device = source = config = None
        dev_lo = dev_num = cfg_lo = cfg_num = 0
    elif nodes == "enriched":
        princ = np.where(df["principal"] != "", "usr:" + df["principal"], "usr:none:" + src_host)
        image = np.where(df["image_path"] != "", "cfg:" + df["image_path"].str.lower(), "cfg:none:" + src_host)
        usr_lo, usr_num = _range(sorted(set(princ)))
        dev_lo, dev_num = _range(sorted("dev:" + h for h in set(src_host)))
        _range(sorted("src:" + ip for ip in set(df["src_ip"])))
        cfg_lo, cfg_num = _range(sorted(set(image)))
        res_lo, res_num = _range(sorted("res:" + h for h in set(dst_host)))
        user = np.array([index[k] for k in princ])
        device = torch.tensor([index["dev:" + h] for h in src_host], dtype=torch.long)
        source = torch.tensor([index["src:" + ip] for ip in df["src_ip"]], dtype=torch.long)
        config = torch.tensor([index[k] for k in image], dtype=torch.long)
        print(f"[optc] copertura principal={np.mean(df['principal'] != ''):.1%} "
              f"image_path={np.mean(df['image_path'] != ''):.1%}")
    else:
        raise ValueError(f"nodes={nodes!r}")
    dst = np.array([index["res:" + h] for h in dst_host])

    types = np.where(df["label_lm"] == 1, T_LATERAL, np.where(df["label"] == 1, T_OTHER, T_BENIGN))
    msg = np.zeros((n, 7), dtype=np.float32)
    msg[:, 0] = 1.0
    msg[:, 4] = [port_class(p) for p in df["dst_port"]]
    t0 = df["timestamp_abs"].iloc[0]

    num_nodes = len(keys)
    node_features = torch.zeros(num_nodes, 16, dtype=torch.float)
    node_features[:, 14] = 1.0

    print(f"[optc] nodes={nodes} eventi={n} (train={n_train} val={n_val} test={n - n_train - n_val}) "
          f"nodi={num_nodes} LM={int((types == T_LATERAL).sum())} Other={int((types == T_OTHER).sum())}")
    data = StreamData(
        user=torch.tensor(user, dtype=torch.long),
        dst=torch.tensor(dst, dtype=torch.long),
        t=torch.tensor(np.floor(df["timestamp_abs"].to_numpy() - t0), dtype=torch.long),
        msg=torch.from_numpy(msg),
        y=torch.tensor(df["label"].to_numpy(), dtype=torch.long),
        types=torch.tensor(types, dtype=torch.long),
        node_features=node_features,
        keys=keys,
        num_nodes=num_nodes,
        neg_lo=res_lo,
        neg_num=res_num,
        device_nodes=device,
        source_nodes=source,
        config_nodes=config,
        usr_lo=usr_lo,
        usr_num=usr_num,
        dev_lo=dev_lo,
        dev_num=dev_num,
        cfg_lo=cfg_lo,
        cfg_num=cfg_num,
    )
    # train_tgn slices with int(n * frac): the +0.5 makes that land exactly on n_train / n_val.
    return data, (n_train + 0.5) / n, (n_val + 0.5) / n, df
