"""Map the UWF-ZeekData24 network connection logs to the ZTA stream (5-node causal chain).

External-validity evaluation on the UWF-ZeekData24 cyber-range dataset.
The dataset provides enterprise network telemetry labeled by MITRE ATT&CK tactics:
  - Benign: normal network baseline
  - Credential_Access: credential dumping / authentication abuse (T_THEFT = 4)
  - Reconnaissance: network scanning and discovery (T_CONTEXTUAL = 2)
  - Exfiltration: large data transfers (T_EXFIL = 5)
  - Initial_Access / Defense_Evasion / Persistence / Privilege_Escalation: intrusion & lateral movement (T_LATERAL = 3)

The stream is chronologically structured for one-class training:
  - Train split (0.0 to train_frac): 100% benign traffic, allowing the TGN to learn
    habitual network topologies, entity interactions and normal timing baselines.
  - Val split (train_frac to train_frac + val_frac): held-out benign traffic for calibration.
  - Test split (remainder): held-out benign traffic merged with MITRE ATT&CK attack events.
"""

from __future__ import annotations

import glob
import math
import os

import pandas as pd
import torch

from graphagate.train_tgn import StreamData

# Event types matching src/data/stream_synthetic.py
T_BENIGN = 0
T_POLICY = 1
T_CONTEXTUAL = 2
T_LATERAL = 3
T_THEFT = 4
T_EXFIL = 5

_SERVICE_CODES = {
    "dns": 0.0,
    "ssl": 1.0,
    "http": 2.0,
    "smb": 3.0,
    "dce_rpc": 4.0,
    "ntlm": 5.0,
    "gssapi": 6.0,
    "dhcp": 7.0,
    "ntp": 8.0,
}


def _service_to_code(srv: str | float) -> float:
    if not isinstance(srv, str) or not srv or srv == "none":
        return 9.0  # other/unknown
    srv_lower = srv.lower()
    for k, v in _SERVICE_CODES.items():
        if k in srv_lower:
            return v
    return 9.0


def _is_rfc1918(ip: str) -> bool:
    if ip.startswith(("10.", "192.168.")):
        return True
    if ip.startswith("172."):
        parts = ip.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            val = int(parts[1])
            return 16 <= val <= 31
    # UWF cyber-range internal subnet: 143.88.x.x
    return ip.startswith("143.88.")


def load_uwf_stream(
    data_dir: str,
    *,
    max_benign_events: int = 50_000,
    max_attack_events_per_cat: int = 5_000,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> tuple[StreamData, float, float]:
    """Load and map UWF-ZeekData24 into a :class:`StreamData` object and return exact split fractions.

    Returns:
        (StreamData, actual_train_frac, actual_val_frac) where actual_train_frac and actual_val_frac
        guarantee that train_tgn's internal split partitions exactly into 100% benign train,
        100% benign val, and benign+attacks test.
    """
    data_dir = os.path.abspath(data_dir)
    benign_pattern = os.path.join(data_dir, "Benign", "*.csv")
    benign_files = glob.glob(benign_pattern)
    if not benign_files:
        raise FileNotFoundError(f"No benign CSV files found under {benign_pattern}")

    print(f"[uwf] Loading benign baseline from {len(benign_files)} file(s)...")
    df_benign = pd.concat([pd.read_csv(f) for f in benign_files], ignore_index=True)
    df_benign = df_benign.dropna(subset=["ts", "src_ip_zeek", "dest_ip_zeek"])
    df_benign = df_benign.sort_values("ts").reset_index(drop=True)

    if len(df_benign) > max_benign_events:
        print(f"[uwf] Down-sampling benign from {len(df_benign)} to {max_benign_events} events")
        df_benign = df_benign.iloc[:max_benign_events]

    # Load attack categories
    attack_map = {
        "Credential_Access": T_THEFT,
        "Reconnaissance": T_CONTEXTUAL,
        "Exfiltration": T_EXFIL,
        "Initial_Access": T_LATERAL,
        "Defense_Evasion": T_LATERAL,
        "Persistence": T_LATERAL,
        "Privilege_Escalation": T_LATERAL,
    }

    attack_dfs = []
    for cat_name, etype in attack_map.items():
        cat_files = glob.glob(os.path.join(data_dir, cat_name, "*.csv"))
        if not cat_files:
            continue
        df_cat = pd.concat([pd.read_csv(f) for f in cat_files], ignore_index=True)
        df_cat = df_cat.dropna(subset=["ts", "src_ip_zeek", "dest_ip_zeek"])
        df_cat = df_cat.sort_values("ts").reset_index(drop=True)
        if len(df_cat) > max_attack_events_per_cat:
            df_cat = df_cat.iloc[:max_attack_events_per_cat]
        df_cat["etype"] = etype
        df_cat["cat_name"] = cat_name
        attack_dfs.append(df_cat)
        print(f"[uwf] Loaded {len(df_cat):>5} attack events for {cat_name} (etype={etype})")

    df_benign["etype"] = T_BENIGN
    df_benign["cat_name"] = "Benign"

    # One-Class Chronological Partition:
    # 0 to train_end: 100% benign
    # train_end to val_end: 100% benign
    # val_end to end: remaining 20% benign + all attack events (interleaved in the test window)
    n_b = len(df_benign)
    n_train_b = int(n_b * train_frac)
    n_val_b = int(n_b * val_frac)

    df_train_b = df_benign.iloc[:n_train_b]
    df_val_b = df_benign.iloc[n_train_b : n_train_b + n_val_b]
    df_test_b = df_benign.iloc[n_train_b + n_val_b :]

    # Normalize benign timestamps to relative seconds from t0
    t0_b = df_benign["ts"].min()
    df_train_b = df_train_b.copy()
    df_val_b = df_val_b.copy()
    df_test_b = df_test_b.copy()

    df_train_b["rel_t"] = (df_train_b["ts"] - t0_b).astype(int)
    df_val_b["rel_t"] = (df_val_b["ts"] - t0_b).astype(int)
    df_test_b["rel_t"] = (df_test_b["ts"] - t0_b).astype(int)

    test_start_t = df_test_b["rel_t"].min() if len(df_test_b) > 0 else df_val_b["rel_t"].max() + 1
    test_end_t = df_test_b["rel_t"].max() if len(df_test_b) > 0 else test_start_t + 3600

    # Map attack events into the test window
    all_attacks = pd.concat(attack_dfs, ignore_index=True) if attack_dfs else pd.DataFrame()
    if not all_attacks.empty:
        att_t_min = all_attacks["ts"].min()
        att_t_max = all_attacks["ts"].max()
        span_att = max(att_t_max - att_t_min, 1.0)
        span_test = max(test_end_t - test_start_t, 3600)
        # Rescale attack timestamps smoothly across the test window
        all_attacks = all_attacks.copy()
        all_attacks["rel_t"] = (
            test_start_t + ((all_attacks["ts"] - att_t_min) / span_att) * span_test
        ).astype(int)

    df_test_all = pd.concat([df_test_b, all_attacks], ignore_index=True)
    df_test_all = df_test_all.sort_values("rel_t").reset_index(drop=True)

    # Full combined stream: Train (100% Benign) -> Val (100% Benign) -> Test (Benign + Attacks)
    df_all = pd.concat([df_train_b, df_val_b, df_test_all], ignore_index=True)
    print(
        f"[uwf] Full stream: {len(df_all)} events "
        f"(train_benign={len(df_train_b)}, val_benign={len(df_val_b)}, "
        f"test={len(df_test_all)} [{len(df_test_b)} benign + {len(all_attacks)} attacks])"
    )

    # Entity indexing across the 5 node types:
    # 1. users: usr:<src_ip>
    # 2. devices: dev:<src_ip>
    # 3. sources: src:<src_ip>
    # 4. configs: cfg:<service>
    # 5. resources: res:<dest_ip>:<dest_port>
    src_ips = sorted(df_all["src_ip_zeek"].dropna().unique())
    services = sorted(df_all["service"].fillna("none").unique())
    res_keys = sorted(
        (df_all["dest_ip_zeek"] + ":" + df_all["dest_port_zeek"].astype(str)).unique()
    )

    keys: list[str] = []
    index: dict[str, int] = {}

    def _get_idx(k: str) -> int:
        if k not in index:
            idx = len(keys)
            index[k] = idx
            keys.append(k)
            return idx
        return index[k]

    # Pre-register entity index blocks for contiguous negative sampling
    user_lo = len(keys)
    for ip in src_ips:
        _get_idx(f"usr:{ip}")
    user_num = len(src_ips)

    dev_lo = len(keys)
    for ip in src_ips:
        _get_idx(f"dev:{ip}")
    dev_num = len(src_ips)

    _src_lo = len(keys)
    for ip in src_ips:
        _get_idx(f"src:{ip}")
    src_num = len(src_ips)

    cfg_lo = len(keys)
    for srv in services:
        _get_idx(f"cfg:{srv}")
    cfg_num = len(services)

    res_lo = len(keys)
    for r in res_keys:
        _get_idx(f"res:{r}")
    res_num = len(res_keys)

    num_nodes = len(keys)

    # Build event tensors
    src_l, cfg_l, dev_l, usr_l, dst_l = [], [], [], [], []
    t_l, y_l, ty_l, msg_l = [], [], [], []
    last_user_t: dict[int, int] = {}

    for row in df_all.itertuples():
        s_ip = str(row.src_ip_zeek)
        d_ip = str(row.dest_ip_zeek)
        d_port = str(row.dest_port_zeek)
        srv = str(row.service) if pd.notna(row.service) else "none"
        rel_t = int(row.rel_t)
        etype = int(row.etype)

        u = _get_idx(f"usr:{s_ip}")
        d = _get_idx(f"dev:{s_ip}")
        s = _get_idx(f"src:{s_ip}")
        c = _get_idx(f"cfg:{srv}")
        r = _get_idx(f"res:{d_ip}:{d_port}")

        dt_user = rel_t - last_user_t.get(u, rel_t)
        last_user_t[u] = rel_t

        nbytes = float(row.orig_bytes) if (pd.notna(row.orig_bytes) and row.orig_bytes > 0) else 0.0
        method_code = _service_to_code(srv)

        # 10-dim edge message layout (same as synthetic and PicoDomain):
        # [ja3_valid=1.0, s1=0, s2=0, s3=0, method, roleVal=0, clrVal=0, bytes_in, bytes_out=0, dt_user]
        msg_vec = [
            1.0,  # ja3_valid
            0.0,  # s1
            0.0,  # s2
            0.0,  # s3
            method_code,
            0.0,  # roleVal
            0.0,  # clrVal
            float(math.log1p(nbytes)) / 10.0,
            0.0,  # bytes_out (zero response-side leakage)
            float(math.log1p(dt_user)) / 10.0,
        ]

        src_l.append(s)
        cfg_l.append(c)
        dev_l.append(d)
        usr_l.append(u)
        dst_l.append(r)
        t_l.append(rel_t)
        y_l.append(0 if etype == T_BENIGN else 1)
        ty_l.append(etype)
        msg_l.append(msg_vec)

    # Static node features (16-dim)
    node_features = torch.zeros(num_nodes, 16, dtype=torch.float)
    node_features[:, 14] = 1.0  # default trust slot
    for ip in src_ips:
        node_features[index[f"src:{ip}"], 5] = 1.0 if _is_rfc1918(ip) else 0.0

    types_t = torch.tensor(ty_l, dtype=torch.long)
    print(
        f"[uwf] Finished mapping: events={len(t_l)} nodes={num_nodes} "
        f"(users={user_num}, devices={dev_num}, sources={src_num}, configs={cfg_num}, resources={res_num})"
    )
    print(
        f"[uwf] Labels distribution: "
        f"benign={(types_t == T_BENIGN).sum().item()} "
        f"theft={(types_t == T_THEFT).sum().item()} "
        f"contextual={(types_t == T_CONTEXTUAL).sum().item()} "
        f"lateral={(types_t == T_LATERAL).sum().item()} "
        f"exfil={(types_t == T_EXFIL).sum().item()}"
    )
    actual_train_frac = float(len(df_train_b)) / float(len(df_all))
    actual_val_frac = float(len(df_val_b)) / float(len(df_all))

    stream_data = StreamData(
        user=torch.tensor(usr_l, dtype=torch.long),
        dst=torch.tensor(dst_l, dtype=torch.long),
        t=torch.tensor(t_l, dtype=torch.long),
        msg=torch.tensor(msg_l, dtype=torch.float),
        y=torch.tensor(y_l, dtype=torch.long),
        types=types_t,
        node_features=node_features,
        keys=keys,
        num_nodes=num_nodes,
        neg_lo=res_lo,
        neg_num=res_num,
        device_nodes=torch.tensor(dev_l, dtype=torch.long),
        source_nodes=torch.tensor(src_l, dtype=torch.long),
        config_nodes=torch.tensor(cfg_l, dtype=torch.long),
        usr_lo=user_lo,
        usr_num=user_num,
        dev_lo=dev_lo,
        dev_num=dev_num,
        cfg_lo=cfg_lo,
        cfg_num=cfg_num,
    )
    return stream_data, actual_train_frac, actual_val_frac

