"""Map the AIT Log Data Set (AIT-LDS 2023, Zenodo) to the ZTA 5-node stream.

External-validity evaluation on the AIT Log Data Set (AIT-LDS / AIT-ADS, Zenodo record 5789064).
Paper: Landauer et al., "Maintainable Log Datasets for Evaluation of Intrusion Detection Systems",
IEEE Trans. Dependable and Secure Computing (TDSC), 2023.

AIT-LDS simulates multi-host enterprise environments (workstations, DC, web server, mail, VPN)
with multi-step attacks (Reconnaissance, Credential Access, Lateral Movement, Exfiltration).
This loader supports both:
1. Native AIT-LDS Suricata EVE logs (``eve.json``) with TLS JA3/JA4 fingerprints, HTTP, DNS,
   flow records, combined with system authentication logs (``auth.log``) and ground-truth labels.
2. Zeek logs (``conn.log``, ``ssl.log`` with JA3, ``http.log``, etc.) if generated from AIT PCAPs.

Schema mapping onto the 5-node causal chain:
--------------------------------------------
  source   = client source IP (id.orig_h / src_ip).
  config   = client TLS JA3 fingerprint (from ssl.log:ja3 or eve.json tls.ja3.hash),
             falling back to application/service protocol fingerprint within bind_ttl.
  device   = host machine identity (hostname / machine account) associated with the source IP.
  user     = authenticated human principal (from auth.log SSH/PAM logins, Kerberos client),
             associated with the source IP within bind_ttl.
  resource = destination endpoint (id.resp_h:id.resp_p / dest_ip:dest_port / URI).

Strict One-Class Training:
--------------------------
To strictly enforce benign-only training:
  - 100% of benign baseline traffic up to train_frac (default 70%) is used for training.
  - The next val_frac (default 10%) of benign traffic is used for threshold calibration.
  - The remaining 20% of benign traffic plus all annotated attack events populate the test slice.
  - No attack event ever enters the training or validation splits.
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import os
import re
from datetime import datetime, timezone

import pandas as pd
import torch

from graphagate.train_tgn import StreamData

# Event types matching src/data/stream_synthetic.py and PicoDomain
T_BENIGN = 0
T_POLICY = 1
T_CONTEXTUAL = 2
T_LATERAL = 3
T_THEFT = 4
T_EXFIL = 5

_SERVICE_CODES = {
    "dns": 0.0,
    "ssl": 1.0,
    "tls": 1.0,
    "https": 1.0,
    "http": 2.0,
    "smb": 3.0,
    "dce_rpc": 4.0,
    "rpc": 4.0,
    "ntlm": 5.0,
    "ssh": 6.0,
    "dhcp": 7.0,
    "ntp": 8.0,
}

_SSH_AUTH_RE = re.compile(
    r"Accepted\s+(?:password|publickey|keyboard-interactive)\s+for\s+(?P<user>\S+)\s+from\s+(?P<ip>[\d\.]+)",
    re.IGNORECASE,
)
_PAM_SESSION_RE = re.compile(
    r"session\s+opened\s+for\s+user\s+(?P<user>\S+)",
    re.IGNORECASE,
)


def _service_to_code(srv: str | float) -> float:
    if not isinstance(srv, str) or not srv or srv.lower() in ("none", "unknown", "-"):
        return 9.0
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
    return False


def _parse_iso_or_epoch(val: str | float | int) -> float:
    """Parse timestamp into epoch seconds (float)."""
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return 0.0
    val_clean = val.strip()
    try:
        return float(val_clean)
    except ValueError:
        pass
    # ISO-8601 parsing
    try:
        # e.g. 2021-04-12T14:32:00.123456+0000 or Z
        val_iso = val_clean.replace("Z", "+00:00")
        dt = datetime.fromisoformat(val_iso)
        return dt.timestamp()
    except Exception:
        pass
    # Syslog format: "Apr 12 14:32:00"
    for fmt in ("%b %d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(val_clean[:19], fmt)
            if dt.year == 1900:
                dt = dt.replace(year=2023)
            return dt.replace(tzinfo=timezone.utc).timestamp()
        except Exception:
            continue
    return 0.0


def _map_attack_tag(tag: str) -> int:
    """Map AIT-LDS ground-truth attack tags to project event types."""
    t = tag.lower()
    if any(k in t for k in ("crack_password", "theft", "credential", "mimikatz", "dump", "shadow", "pass-the-hash")):
        return T_THEFT
    if any(k in t for k in ("escalat", "change_user", "lateral", "psexec", "wmi", "smb", "pivot", "remote_exec")):
        return T_LATERAL
    if any(k in t for k in ("dnsteal", "exfil", "data_transfer", "leak", "upload")):
        return T_EXFIL
    if any(k in t for k in ("scan", "nmap", "probe", "discovery", "dirb", "wpscan", "foothold", "webshell", "traceroute", "attacker_http", "attacker_vpn", "attacker")):
        return T_CONTEXTUAL
    if any(k in t for k in ("policy", "unauthorized", "clearance")):
        return T_POLICY
    return T_BENIGN


def _load_labels(data_dir: str) -> dict[str, list[tuple[float, float, int]]]:
    """Load ground-truth attack labels from labels/ directory and AIT attack signatures.

    Returns:
        dict mapping target (host or IP) -> list of (start_t, end_t, etype) intervals.
    """
    labels_by_target: dict[str, list[tuple[float, float, int]]] = {}

    # Attacker IPs specifically identified in AIT-LDS scenarios (e.g. fox VPN client and attacker subnet)
    attacker_ips = {"172.17.130.196", "192.168.128.4"}
    for ip in attacker_ips:
        # All actions originating from known attacker footholds/VPN IPs are marked as attack
        labels_by_target.setdefault(ip, []).append((0.0, float("inf"), T_CONTEXTUAL))

    # Parse openvpn labels to catch dynamically assigned attacker VPN IPs
    vpn_label_file = os.path.join(data_dir, "labels", "vpn", "logs", "openvpn.log")
    vpn_log_file = os.path.join(data_dir, "gather", "vpn", "logs", "openvpn.log")
    if os.path.exists(vpn_label_file) and os.path.exists(vpn_log_file):
        try:
            with open(vpn_log_file, errors="ignore") as fg:
                v_lines = fg.readlines()
            with open(vpn_label_file, errors="ignore") as fl:
                for line in fl:
                    if line.startswith("{"):
                        obj = json.loads(line)
                        l_no = obj.get("line")
                        if l_no and 1 <= l_no <= len(v_lines):
                            txt = v_lines[l_no - 1]
                            # Look for assigned IP, e.g. "172.17.x.x"
                            m_ip = re.search(r"172\.17\.\d+\.\d+", txt)
                            if m_ip:
                                a_ip = m_ip.group(0)
                                attacker_ips.add(a_ip)
                                labels_by_target.setdefault(a_ip, []).append((0.0, float("inf"), T_CONTEXTUAL))
        except Exception:
            pass

    # Parse auth.log labels (privilege escalation / lateral progression)
    auth_label_files = glob.glob(os.path.join(data_dir, "labels", "**", "auth.log"), recursive=True)
    for alf in auth_label_files:
        agf = alf.replace(os.path.sep + "labels" + os.path.sep, os.path.sep + "gather" + os.path.sep)
        if not os.path.exists(agf):
            continue
        try:
            with open(agf, errors="ignore") as fg:
                a_lines = fg.readlines()
            with open(alf, errors="ignore") as fl:
                for line in fl:
                    if line.startswith("{"):
                        obj = json.loads(line)
                        l_no = obj.get("line")
                        tags = obj.get("labels", [])
                        if l_no and 1 <= l_no <= len(a_lines):
                            log_line = a_lines[l_no - 1]
                            ts = _parse_iso_or_epoch(log_line[:15])
                            for tag in tags:
                                etype = _map_attack_tag(tag)
                                if etype != T_BENIGN and ts > 0:
                                    labels_by_target.setdefault("global", []).append((ts - 60.0, ts + 60.0, etype))
        except Exception:
            pass

    return labels_by_target


def load_ait_stream(
    data_dir: str,
    *,
    max_benign_events: int = 50_000,
    max_attack_events_per_cat: int = 5_000,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    bind_ttl: float = 3600.0,
) -> tuple[StreamData, float, float]:
    """Parse and map the AIT Log Data Set onto the 5-node ZTA causal chain.

    Ensures strict one-class partitioning:
      - Train slice (0.0 to actual_train_frac): 100% benign baseline traffic.
      - Val slice (actual_train_frac to actual_train_frac + actual_val_frac): 100% benign baseline traffic.
      - Test slice (remainder): held-out benign traffic + MITRE ATT&CK events.

    Returns:
        (StreamData, actual_train_frac, actual_val_frac)
    """
    data_dir = os.path.abspath(data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"AIT-LDS data directory not found: {data_dir}")

    print(f"[ait_lds] Scanning for logs under {data_dir}...")
    labels_map = _load_labels(data_dir)

    # 1. Parse network / access events (Suricata EVE JSON or Zeek conn/ssl/http)
    eve_files = glob.glob(os.path.join(data_dir, "**", "*eve*.json*"), recursive=True)
    zeek_files = glob.glob(os.path.join(data_dir, "**", "conn.log*"), recursive=True)
    auth_files = glob.glob(os.path.join(data_dir, "**", "*auth*.log*"), recursive=True) + \
                 glob.glob(os.path.join(data_dir, "**", "*secure*"), recursive=True)

    access_events: list[dict] = []
    bind_events: list[tuple[float, str, str, str]] = []  # (ts, ip, kind, value)

    # Harvest authentication bindings: IP -> User & IP -> Device
    print(f"[ait_lds] Found {len(auth_files)} authentication log(s)...")
    for af in auth_files:
        host = os.path.basename(os.path.dirname(os.path.dirname(af))) or "host"
        opener = gzip.open if af.endswith(".gz") else open
        try:
            with opener(af, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    m_ssh = _SSH_AUTH_RE.search(line)
                    if m_ssh:
                        u = m_ssh.group("user").strip()
                        ip = m_ssh.group("ip").strip()
                        ts = _parse_iso_or_epoch(line[:25])
                        bind_events.append((ts, ip, "user", u))
                        bind_events.append((ts, ip, "device", host))
                    else:
                        m_pam = _PAM_SESSION_RE.search(line)
                        if m_pam:
                            u = m_pam.group("user").strip()
                            ts = _parse_iso_or_epoch(line[:25])
                            bind_events.append((ts, "local", "user", u))
                            bind_events.append((ts, "local", "device", host))
        except Exception as e:
            print(f"[ait_lds] Note: error reading {af}: {e}")

    # Harvest network access events and TLS/JA3 fingerprints
    if eve_files:
        print(f"[ait_lds] Parsing {len(eve_files)} Suricata EVE JSON log(s)...")
        for ef in eve_files:
            opener = gzip.open if ef.endswith(".gz") else open
            try:
                with opener(ef, "rt", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line_s = line.strip()
                        if not line_s or not line_s.startswith("{"):
                            continue
                        try:
                            record = json.loads(line_s)
                        except Exception:
                            continue

                        ts = _parse_iso_or_epoch(record.get("timestamp", 0))
                        ev_type = record.get("event_type", "")
                        src_ip = record.get("src_ip", "")
                        dest_ip = record.get("dest_ip", "")
                        dest_port = record.get("dest_port", 0)
                        app_proto = record.get("app_proto", ev_type) or ev_type

                        if not src_ip or not dest_ip:
                            continue

                        # Extract TLS JA3 fingerprint
                        if ev_type == "tls":
                            tls_data = record.get("tls", {})
                            ja3 = tls_data.get("ja3", {}).get("hash") if isinstance(tls_data.get("ja3"), dict) else tls_data.get("ja3")
                            if ja3:
                                bind_events.append((ts, src_ip, "config", str(ja3)))

                        # Extract HTTP URI / resource details
                        uri = ""
                        nbytes = 0.0
                        if ev_type == "http":
                            http_data = record.get("http", {})
                            uri = http_data.get("url") or http_data.get("hostname", "")
                            nbytes = float(http_data.get("length", 0.0))
                        elif "flow" in record:
                            flow = record.get("flow", {})
                            nbytes = float(flow.get("bytes_toserver", 0.0))

                        # Target resource key
                        res_key = f"res:{dest_ip}:{dest_port}" + (f":{uri[:32]}" if uri else "")

                        # Determine attack label
                        etype = T_BENIGN
                        target_labels = labels_map.get(src_ip, []) + labels_map.get(dest_ip, []) + labels_map.get("global", [])
                        for st, et, tag_type in target_labels:
                            if st <= ts <= et:
                                etype = tag_type
                                break

                        # Specialize attack category based on target service/port
                        if etype != T_BENIGN:
                            if dest_port in (22, 135, 139, 445, 3389):
                                etype = T_LATERAL
                            elif dest_port == 53:
                                etype = T_EXFIL
                            elif etype == T_THEFT:
                                etype = T_THEFT
                            elif dest_port in (80, 443, 8080, 8443):
                                etype = T_CONTEXTUAL

                        access_events.append({
                            "ts": ts,
                            "src_ip": src_ip,
                            "dest_ip": dest_ip,
                            "dest_port": dest_port,
                            "service": str(app_proto),
                            "res": res_key,
                            "bytes": nbytes,
                            "etype": etype,
                        })
            except Exception as e:
                print(f"[ait_lds] Note: error reading {ef}: {e}")

    elif zeek_files:
        print(f"[ait_lds] Found Zeek logs; parsing conn.log and ssl.log...")
        # Fallback to standard Zeek parsing
        for zf in zeek_files:
            try:
                df_c = pd.read_csv(zf, sep="\t", comment="#", header=None, low_memory=False)
                # Zeek default: ts=0, uid=1, id.orig_h=2, id.orig_p=3, id.resp_h=4, id.resp_p=5, proto=6, service=7
                for _, r in df_c.iterrows():
                    ts = _parse_iso_or_epoch(r[0])
                    s_ip = str(r[2])
                    d_ip = str(r[4])
                    d_port = r[5]
                    srv = str(r[7]) if pd.notna(r[7]) else "none"
                    access_events.append({
                        "ts": ts,
                        "src_ip": s_ip,
                        "dest_ip": d_ip,
                        "dest_port": d_port,
                        "service": srv,
                        "res": f"res:{d_ip}:{d_port}",
                        "bytes": 0.0,
                        "etype": T_BENIGN,
                    })
            except Exception as e:
                print(f"[ait_lds] Note: error reading {zf}: {e}")

    if not access_events:
        raise RuntimeError(
            f"No valid network access events extracted from {data_dir}. "
            "Ensure the AIT-LDS directory contains either Suricata 'eve.json' logs or Zeek 'conn.log' files."
        )

    df_events = pd.DataFrame(access_events)
    df_events = df_events.sort_values("ts").reset_index(drop=True)

    # Split into Benign and Attacks
    df_benign = df_events[df_events["etype"] == T_BENIGN].copy().reset_index(drop=True)
    df_attack = df_events[df_events["etype"] != T_BENIGN].copy().reset_index(drop=True)

    print(f"[ait_lds] Total extracted raw events: {len(df_events)} (benign={len(df_benign)}, attack={len(df_attack)})")

    if len(df_benign) > max_benign_events:
        print(f"[ait_lds] Down-sampling benign from {len(df_benign)} to {max_benign_events} events")
        df_benign = df_benign.iloc[:max_benign_events].reset_index(drop=True)

    if len(df_attack) > 0:
        attack_dfs = []
        for cat_type in (T_POLICY, T_CONTEXTUAL, T_LATERAL, T_THEFT, T_EXFIL):
            df_cat = df_attack[df_attack["etype"] == cat_type]
            if len(df_cat) > max_attack_events_per_cat:
                df_cat = df_cat.iloc[:max_attack_events_per_cat]
            if len(df_cat) > 0:
                attack_dfs.append(df_cat)
        df_attack = pd.concat(attack_dfs, ignore_index=True) if attack_dfs else pd.DataFrame()

    # STRICT ONE-CLASS PARTITIONING:
    # Train = first train_frac of benign (100% BENIGN)
    # Val   = next val_frac of benign (100% BENIGN)
    # Test  = remaining benign + attacks
    n_b = len(df_benign)
    n_train_b = int(n_b * train_frac)
    n_val_b = int(n_b * val_frac)

    df_train_b = df_benign.iloc[:n_train_b].copy()
    df_val_b = df_benign.iloc[n_train_b : n_train_b + n_val_b].copy()
    df_test_b = df_benign.iloc[n_train_b + n_val_b :].copy()

    t0 = df_benign["ts"].min()
    df_train_b["rel_t"] = (df_train_b["ts"] - t0).astype(int)
    df_val_b["rel_t"] = (df_val_b["ts"] - t0).astype(int)
    df_test_b["rel_t"] = (df_test_b["ts"] - t0).astype(int)

    test_start_t = df_test_b["rel_t"].min() if len(df_test_b) > 0 else (df_val_b["rel_t"].max() + 1 if len(df_val_b) > 0 else 1)
    test_end_t = df_test_b["rel_t"].max() if len(df_test_b) > 0 else test_start_t + 3600

    if not df_attack.empty:
        att_t_min = df_attack["ts"].min()
        att_t_max = df_attack["ts"].max()
        span_att = max(att_t_max - att_t_min, 1.0)
        span_test = max(test_end_t - test_start_t, 3600)
        df_attack = df_attack.copy()
        df_attack["rel_t"] = (
            test_start_t + ((df_attack["ts"] - att_t_min) / span_att) * span_test
        ).astype(int)

    df_test_all = pd.concat([df_test_b, df_attack], ignore_index=True)
    df_test_all = df_test_all.sort_values("rel_t").reset_index(drop=True)

    df_all = pd.concat([df_train_b, df_val_b, df_test_all], ignore_index=True)

    # Sort session bindings chronologically
    bind_events.sort(key=lambda x: x[0])

    # Pre-register nodes for contiguous intervals
    bind_users = {v for _, _, k, v in bind_events if k == "user"}
    bind_devs = {v for _, _, k, v in bind_events if k == "device"}
    bind_cfgs = {v for _, _, k, v in bind_events if k == "config"}

    src_names = sorted(df_all["src_ip"].unique())
    res_names = sorted(df_all["res"].unique())

    # Include per-IP sentinels to prevent memory contamination
    user_names = sorted(bind_users) + [f"usr:none:{ip}" for ip in src_names]
    dev_names = sorted(bind_devs) + [f"dev:none:{ip}" for ip in src_names]
    cfg_names = sorted(bind_cfgs) + [f"cfg:none:{ip}" for ip in src_names]

    keys: list[str] = []
    index: dict[str, int] = {}

    def _idx(k: str) -> int:
        if k not in index:
            idx = len(keys)
            index[k] = idx
            keys.append(k)
            return idx
        return index[k]

    user_lo = 0
    for u in user_names:
        _idx(f"usr:{u}")
    dev_lo = len(keys)
    for d in dev_names:
        _idx(f"dev:{d}")
    _src_lo = len(keys)
    for s in src_names:
        _idx(f"src:{s}")
    cfg_lo = len(keys)
    for c in cfg_names:
        _idx(f"cfg:{c}")
    res_lo = len(keys)
    for r in res_names:
        _idx(r)
    num_nodes = len(keys)

    # Temporal binding reconstruction
    last_state: dict[tuple[str, str], tuple[float, str]] = {}
    bi = 0
    src_l, cfg_l, dev_l, usr_l, dst_l, t_l, msg_l, y_l, ty_l = [], [], [], [], [], [], [], [], []
    last_user_t: dict[int, int] = {}
    bound_counts = {"user": 0, "device": 0, "config": 0}

    for _, row in df_all.iterrows():
        when = row["ts"]
        ip = row["src_ip"]
        rel_t = int(row["rel_t"])
        res_key = row["res"]
        srv = row["service"]
        nbytes = float(row["bytes"])
        etype = int(row["etype"])

        while bi < len(bind_events) and bind_events[bi][0] <= when:
            bt, bip, bkind, bval = bind_events[bi]
            last_state[(bip, bkind)] = (bt, bval)
            bi += 1

        def _attr(kind: str) -> int:
            seen = last_state.get((ip, kind))
            if seen is not None and abs(when - seen[0]) <= bind_ttl:
                bound_counts[kind] += 1
                prefix = {"user": "usr:", "device": "dev:", "config": "cfg:"}[kind]
                return index[prefix + seen[1]]
            # Fallback sentinel
            prefix = {"user": "usr:usr:none:", "device": "dev:dev:none:", "config": "cfg:cfg:none:"}[kind]
            return index[prefix + ip]

        u = _attr("user")
        d = _attr("device")
        c = _attr("config")
        s = index[f"src:{ip}"]
        r = index[res_key]

        dt_user = rel_t - last_user_t.get(u, rel_t)
        last_user_t[u] = rel_t

        msg_vec = [
            1.0,  # ja3_valid
            0.0, 0.0, 0.0,  # sensor alarms
            _service_to_code(srv),
            0.0, 0.0,  # roleVal, clrVal
            float(math.log1p(nbytes)) / 10.0,
            0.0,  # bytes_resp (zero response-side leakage)
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

    node_features = torch.zeros(num_nodes, 16, dtype=torch.float)
    node_features[:, 14] = 1.0  # default trust score
    for ip in src_names:
        node_features[index[f"src:{ip}"], 5] = 1.0 if _is_rfc1918(ip) else 0.0

    types_t = torch.tensor(ty_l, dtype=torch.long)
    actual_train_frac = float(len(df_train_b)) / float(len(df_all))
    actual_val_frac = float(len(df_val_b)) / float(len(df_all))

    print(
        f"[ait_lds] Mapped Stream: events={len(t_l)} nodes={num_nodes} "
        f"(users={len(user_names)} devices={len(dev_names)} sources={len(src_names)} "
        f"configs={len(cfg_names)} resources={len(res_names)})"
    )
    print(
        "[ait_lds] Binding coverage: "
        + " ".join(f"{k}={bound_counts[k] / len(t_l):.1%}" for k in ("user", "device", "config"))
        + f" (TTL={bind_ttl:.0f}s)"
    )
    print(
        f"[ait_lds] Labels: benign={(types_t == T_BENIGN).sum().item()} "
        f"lateral={(types_t == T_LATERAL).sum().item()} "
        f"theft={(types_t == T_THEFT).sum().item()} "
        f"contextual={(types_t == T_CONTEXTUAL).sum().item()} "
        f"exfil={(types_t == T_EXFIL).sum().item()}"
    )
    print(
        f"[ait_lds] Strict One-Class Split: Train={len(df_train_b)} (100% Benign, frac={actual_train_frac:.3f}), "
        f"Val={len(df_val_b)} (100% Benign, frac={actual_val_frac:.3f}), Test={len(df_test_all)} (Benign + Attacks)\n"
    )

    return (
        StreamData(
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
            neg_num=len(res_names),
            device_nodes=torch.tensor(dev_l, dtype=torch.long),
            source_nodes=torch.tensor(src_l, dtype=torch.long),
            config_nodes=torch.tensor(cfg_l, dtype=torch.long),
            usr_lo=user_lo,
            usr_num=len(user_names),
            dev_lo=dev_lo,
            dev_num=len(dev_names),
            cfg_lo=cfg_lo,
            cfg_num=len(cfg_names),
        ),
        actual_train_frac,
        actual_val_frac,
    )
