"""Map the PicoDomain Zeek logs to the ZTA v4 stream (5-node causal chain).

External-validity case study on **real** network telemetry. PicoDomain is a compact
(16 MB, 3 days, 5 workstations + DC) capture of a realistic Windows-domain intrusion,
released as Zeek/Security Onion JSON logs plus a manually recorded red-team log.

Download (open, no registration):
  https://github.com/iHeartGraph/PicoDomain  →  ``Zeek_Logs.7z`` and ``Red Log.xlsx``
Paper: Laprade, Bowman, Huang, "PicoDomain: A Compact High-Fidelity Cybersecurity Dataset".

Neither file is committed; pass the extracted log directory and the xlsx to
:func:`load_picodomain_stream`.

Why this dataset
----------------
It is the only public corpus we found in which **all five nodes of the v4 chain are backed
by a real field** — including the ``config`` node: ``ssl.log`` carries the ``ja3`` column
(verified: 3433/3433 records, 12 distinct fingerprints). Public authentication-graph
datasets (LANL, OpTC) have no client fingerprint at all, so on them the config node
degenerates and the credential-theft class is unevaluable. See ``docs/datasets.md``.

Schema mapping → :class:`graphagate.train_tgn.StreamData`
--------------------------------------------------------
Access ("user → resource") events are drawn from the four Zeek logs that carry a request
against a named object: ``smb_mapping`` (share), ``smb_files`` (file), ``http`` (URI) and
``dce_rpc`` (RPC endpoint). The other four nodes are attributed to each access event by
**per-source-IP session reconstruction**, because Zeek gives no session join key across
logs (verified: the ``uid`` overlap between ``ssl`` and ``kerberos``/``ntlm`` is exactly 0
— the TLS handshake and the authentication happen on different connections):

  ``source``  = ``id.orig_h`` of the access event (verbatim).
  ``config``  = last ``ssl.log:ja3`` observed from that IP within ``bind_ttl`` seconds.
  ``device``  = last machine identity for that IP within ``bind_ttl``: the Kerberos machine
                account (``HR-WIN7-1$/G.LAB``) or ``ntlm.log:hostname``.
  ``user``    = last human identity for that IP within ``bind_ttl``: ``kerberos.log:client``
                (machine accounts excluded) or ``ntlm.log:username``, realm-normalised.
  ``dst``     = the requested object, namespaced by protocol.

Unbound nodes fall back to per-IP sentinels (``cfg:none:<ip>``, ``dev:none:<ip>``,
``usr:none:<ip>``) rather than to a single shared node, so an unattributed event never
inherits another entity's memory. The binding coverage is measured and printed — it is a
property of the data, not a tuning knob, and it belongs in any result reported on it.

Edge message (``msg_dim=10``, same layout as the synthetic generator)::

    [ ja3_valid=1, s1=0, s2=0, s3=0, method, roleVal=0, clrVal=0,
      bytes_req, bytes_resp=0, log1p(dt_user)/10 ]

The four alarm columns are held **clean**: PicoDomain ships no IDS alert stream, and the
red-team activity here is signal-clean by construction, so a rule baseline must stay blind
to it and the temporal/relational pattern remains the sole discriminator — the honest test.
``bytes_resp`` is held at 0 for the same reason ``http_status`` was removed from the
synthetic message (P0, ``tasks/todo.md``): a response size is not available at decision
time, and using it violates causality.

Static node features are neutral (PicoDomain has no roles, clearances or asset
classification): zeros with the trust slot 14 at 1.0, and slot 5 (source
internal/external, RFC1918) set on source nodes because it is derived from the IP itself.

Labels
------
``Red Log.xlsx`` records ``(timestamp, victim host, victim user, action)`` with the authors'
stated accuracy of ±1 minute. An access event is labelled when it falls within
``label_window`` seconds of a red-team row **and** matches that row's host or user. The
action text is mapped to the project's class taxonomy by the regex table
:data:`_ACTION_CLASS`; everything else stays benign. This is an approximate labelling of a
manually written log and must be reported as such.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone

import torch

from graphagate.train_tgn import StreamData

# --- event-type codes, matching src/data/stream_synthetic.py ---------------------------
T_BENIGN, T_POLICY, T_CONTEXTUAL, T_LATERAL, T_THEFT = 0, 1, 2, 3, 4

# Red-log action text → class. Ordered: the first match wins.
_ACTION_CLASS: list[tuple[re.Pattern, int]] = [
    # Remote execution against another host = lateral movement (the model-owned class).
    (re.compile(r"\b(dcom|wmic?\b|remote process creation|psexec|smbexec)", re.I), T_LATERAL),
    (re.compile(r"executing launcher on|execute beacon on", re.I), T_LATERAL),
    # Credential access, and the reuse that follows it.
    (re.compile(r"mimikatz|powerdump|kerberoast|hashdump", re.I), T_THEFT),
    (re.compile(r"beacon as |get .* beacon on", re.I), T_THEFT),
    # C2 / implant activity = contextual (an IDS or TLS-trust signal would carry it).
    (re.compile(r"callback|beacon|meterpreter|trojanized|persistence|listener", re.I), T_CONTEXTUAL),
    (re.compile(r"hot potato|bypass uac|powerup|elevate", re.I), T_CONTEXTUAL),
    # Domain enumeration.
    (re.compile(r"enumerate|get_(computer|user|gpo)", re.I), T_CONTEXTUAL),
]

# Zeek log → ``method`` code for the message. Keeps protocol identity without leaking.
_METHOD = {"smb_mapping": 0.0, "smb_files": 1.0, "http": 2.0, "dce_rpc": 3.0}

_EXCEL_EPOCH = datetime(1899, 12, 30, tzinfo=timezone.utc)
_XL_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


# --- Zeek log reading ------------------------------------------------------------------

def _read_zeek(log_dir: str, kind: str) -> list[dict]:
    """Read every ``<kind>.<hour>.log`` under ``log_dir``/*/ as JSON lines, time-sorted."""
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(log_dir, "*", f"{kind}.*.log"))):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "ts" in row:  # a handful of records ship without a timestamp
                    rows.append(row)
    rows.sort(key=lambda r: r["ts"])
    return rows


def _ts(row: dict) -> float:
    """Zeek ISO-8601 UTC timestamp → epoch seconds."""
    return datetime.strptime(row["ts"], "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()


def _norm_principal(client: str) -> tuple[str | None, str | None]:
    """``kerberos.log:client`` → ``(user, device)``, realm-normalised.

    ``HR-WIN7-1$/G.LAB`` is a *machine* account (device); ``jdoe/G.LAB`` is a user. The
    realm appears as ``G.LAB``, ``g.lab`` and truncated ``G`` in the same capture, so the
    principal is lower-cased and the realm dropped — without this, one entity occupies
    three memory slots.
    """
    name = client.split("/")[0].strip().lower()
    if not name:
        return None, None
    if name.endswith("$"):
        return None, name[:-1]
    return name, None


def _is_rfc1918(ip: str) -> bool:
    if ip.startswith("10.") or ip.startswith("192.168."):
        return True
    if ip.startswith("172."):
        try:
            return 16 <= int(ip.split(".")[1]) <= 31
        except (IndexError, ValueError):
            return False
    return False


# --- red-team ground truth --------------------------------------------------------------

def _read_red_log(xlsx_path: str) -> list[tuple[float, str, str, int]]:
    """Parse ``Red Log.xlsx`` → ``[(epoch, host, user, etype), ...]``.

    Read with the stdlib (zipfile + ElementTree) so the loader keeps no spreadsheet
    dependency. Date rows carry an integer Excel serial and reset the current day; the
    following rows carry a fraction-of-day.
    """
    with zipfile.ZipFile(xlsx_path) as zf:
        shared = [
            "".join(t.itertext())
            for t in ET.fromstring(zf.read("xl/sharedStrings.xml")).iter(_XL_NS + "si")
        ]
        sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    events: list[tuple[float, str, str, int]] = []
    day: datetime | None = None
    for row in sheet.iter(_XL_NS + "row"):
        cells: list[str] = []
        for c in row.iter(_XL_NS + "c"):
            v = c.find(_XL_NS + "v")
            if v is None or v.text is None:
                cells.append("")
            elif c.get("t") == "s":
                cells.append(shared[int(v.text)])
            else:
                cells.append(v.text)
        cells += [""] * (5 - len(cells))
        raw, host, user, _c2, action = cells[:5]
        if not raw:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue  # header row
        if val >= 1.0:  # a date row: an Excel day serial, no host/action
            day = _EXCEL_EPOCH + timedelta(days=int(val))
            continue
        if day is None or not action.strip():
            continue
        when = (day + timedelta(days=val)).timestamp()
        etype = T_CONTEXTUAL
        for pattern, code in _ACTION_CLASS:
            if pattern.search(action):
                etype = code
                break
        host_key = host.strip().lower()
        user_key = user.strip().lower().split("\\")[-1]
        events.append((when, host_key, user_key, etype))
    if not events:
        raise RuntimeError(f"No red-team rows parsed from {xlsx_path}")
    events.sort()
    return events


# --- the loader --------------------------------------------------------------------------

def load_picodomain_stream(
    log_dir: str,
    red_log_path: str,
    *,
    max_events: int = 200_000,
    bind_ttl: float = 36_000.0,
    label_window: float = 90.0,
    include_http: bool = True,
) -> StreamData:
    """Build a :class:`StreamData` from extracted PicoDomain Zeek logs.

    ``bind_ttl`` is how long a per-IP identity / fingerprint observation stays valid for
    attribution. The default is 10 hours — the default Kerberos TGT lifetime, i.e. the
    period over which the domain itself considers a logon session valid. It is chosen for
    that reason and not tuned against a metric, but coverage does depend on it and the
    dependence is reported: measured user / device / config coverage is 11.8 / 86.6 / 87.5 %
    at 900 s, 29.0 / 89.7 / 90.6 % at 3600 s and 90.2 / 97.5 / 94.8 % at 36000 s.

    ``label_window`` is the half-width, in seconds, of the match against a red-team row
    (the authors state ±1 minute of manual timestamp error).
    """
    red = _read_red_log(red_log_path)

    # --- attribution timeline: (t, ip, kind, value) from the identity-bearing logs ------
    binds: list[tuple[float, str, str, str]] = []
    for row in _read_zeek(log_dir, "ssl"):
        if row.get("ja3"):
            binds.append((_ts(row), row["id.orig_h"], "config", f"ja3:{row['ja3']}"))
    for row in _read_zeek(log_dir, "kerberos"):
        if row.get("client"):
            user, device = _norm_principal(row["client"])
            if user:
                binds.append((_ts(row), row["id.orig_h"], "user", user))
            if device:
                binds.append((_ts(row), row["id.orig_h"], "device", device))
    for row in _read_zeek(log_dir, "ntlm"):
        if row.get("username") and not row["username"].endswith("$"):
            binds.append((_ts(row), row["id.orig_h"], "user", row["username"].strip().lower()))
        if row.get("hostname"):
            binds.append((_ts(row), row["id.orig_h"], "device", row["hostname"].strip().lower()))
    binds.sort()

    # --- access events ------------------------------------------------------------------
    kinds = ["smb_mapping", "smb_files", "dce_rpc"] + (["http"] if include_http else [])
    access: list[tuple[float, str, str, str, float, float]] = []  # t, ip, kind, resource, bytes, _
    for kind in kinds:
        for row in _read_zeek(log_dir, kind):
            ip = row.get("id.orig_h")
            if not ip:
                continue
            if kind == "smb_mapping":
                res = f"smb:{row.get('path', '?')}"
                nbytes = 0.0
            elif kind == "smb_files":
                res = f"smb:{row.get('path', '?')}/{row.get('name', '?')}"
                nbytes = float(row.get("size") or 0.0)
            elif kind == "dce_rpc":
                res = f"rpc:{row.get('endpoint', '?')}.{row.get('operation', '?')}"
                nbytes = 0.0
            else:
                res = f"http:{row.get('host', '?')}{row.get('uri', '?')}"
                nbytes = float(row.get("request_body_len") or 0.0)
            access.append((_ts(row), ip, kind, res, nbytes, 0.0))
    access.sort(key=lambda e: e[0])
    if not access:
        raise RuntimeError(f"No access events found under {log_dir} — is it the extracted log dir?")
    if len(access) > max_events:
        access = access[:max_events]

    # --- node id space: users | devices | sources | configs | resources -------------------
    keys: list[str] = []
    index: dict[str, int] = {}

    def _idx(key: str) -> int:
        i = index.get(key)
        if i is None:
            i = len(keys)
            index[key] = i
            keys.append(key)
        return i

    # Reserve contiguous ranges by pre-registering every entity, in first-seen order per
    # group; the binding-edge negative sampler needs each group to be one interval.
    user_names, dev_names, cfg_names = [], [], []
    for _t, ip, kind, value in binds:
        bucket = {"user": user_names, "device": dev_names, "config": cfg_names}[kind]
        if value not in bucket:
            bucket.append(value)
    src_names = sorted({e[1] for e in access})
    res_names = sorted({e[3] for e in access})
    # Per-IP sentinels for unattributable events (never a single shared "unknown" node).
    user_names += [f"usr:none:{ip}" for ip in src_names]
    dev_names += [f"dev:none:{ip}" for ip in src_names]
    cfg_names += [f"cfg:none:{ip}" for ip in src_names]

    user_lo = 0
    for n in user_names:
        _idx(f"usr:{n}")
    dev_lo = len(keys)
    for n in dev_names:
        _idx(f"dev:{n}")
    src_lo = len(keys)
    for n in src_names:
        _idx(f"src:{n}")
    cfg_lo = len(keys)
    for n in cfg_names:
        _idx(f"cfg:{n}")
    res_lo = len(keys)
    for n in res_names:
        _idx(n)
    num_nodes = len(keys)

    # --- walk the merged timeline, attributing each access event --------------------------
    last: dict[tuple[str, str], tuple[float, str]] = {}  # (ip, kind) -> (t, value)
    bi = 0
    src_l, cfg_l, dev_l, usr_l, dst_l, t_l, msg_l, y_l, ty_l = [], [], [], [], [], [], [], [], []
    last_user_t: dict[int, float] = {}
    bound = {"config": 0, "device": 0, "user": 0}
    t0 = access[0][0]

    for when, ip, kind, res, nbytes, _ in access:
        while bi < len(binds) and binds[bi][0] <= when:
            bt, bip, bkind, bval = binds[bi]
            last[(bip, bkind)] = (bt, bval)
            bi += 1

        def _attr(what: str) -> int:
            seen = last.get((ip, what))
            if seen is not None and when - seen[0] <= bind_ttl:
                bound[what] += 1
                prefix = {"config": "cfg:", "device": "dev:", "user": "usr:"}[what]
                return index[prefix + seen[1]]
            prefix = {"config": "cfg:cfg:none:", "device": "dev:dev:none:", "user": "usr:usr:none:"}[what]
            return index[prefix + ip]

        u = _attr("user")
        d = _attr("device")
        c = _attr("config")
        s = index[f"src:{ip}"]
        r = index[res]

        # Label: a red-team row within ``label_window`` that names this host or this user.
        etype = T_BENIGN
        u_key = keys[u][4:]
        d_key = keys[d][4:]
        for r_t, r_host, r_user, r_type in red:
            if abs(r_t - when) > label_window:
                continue
            if (r_host and r_host == d_key) or (r_user and r_user == u_key):
                etype = r_type
                break

        dt_user = when - last_user_t.get(u, when)
        last_user_t[u] = when
        msg_l.append([
            1.0, 0.0, 0.0, 0.0, _METHOD[kind], 0.0, 0.0,
            float(torch.log1p(torch.tensor(nbytes)).item()) / 10.0,
            0.0,
            float(torch.log1p(torch.tensor(dt_user)).item()) / 10.0,
        ])
        src_l.append(s)
        cfg_l.append(c)
        dev_l.append(d)
        usr_l.append(u)
        dst_l.append(r)
        t_l.append(int(when - t0))
        y_l.append(0 if etype == T_BENIGN else 1)
        ty_l.append(etype)

    node_features = torch.zeros(num_nodes, 16, dtype=torch.float)
    node_features[:, 14] = 1.0  # trust slot default, as in the synthetic generator
    for ip in src_names:  # slot 5: RFC1918, derived from the IP itself (not a label)
        node_features[index[f"src:{ip}"], 5] = 1.0 if _is_rfc1918(ip) else 0.0

    n = len(t_l)
    types_t = torch.tensor(ty_l, dtype=torch.long)
    print(
        f"[picodomain] events={n} nodes={num_nodes} "
        f"(users={len(user_names)} devices={len(dev_names)} sources={len(src_names)} "
        f"configs={len(cfg_names)} resources={len(res_names)})"
    )
    print(
        "[picodomain] binding coverage: "
        + " ".join(f"{k}={bound[k] / n:.1%}" for k in ("user", "device", "config"))
        + f"  ttl={bind_ttl:.0f}s"
    )
    print(
        "[picodomain] labels: "
        + " ".join(
            f"{name}={int((types_t == code).sum())}"
            for name, code in (("benign", T_BENIGN), ("contextual", T_CONTEXTUAL),
                               ("lateral", T_LATERAL), ("theft", T_THEFT))
        )
        + f"  window=±{label_window:.0f}s"
    )

    return StreamData(
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
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspect the PicoDomain → ZTA v4 mapping.")
    p.add_argument("--log-dir", required=True, help="directory holding the extracted Zeek_Logs")
    p.add_argument("--red-log", required=True, help="path to 'Red Log.xlsx'")
    p.add_argument("--max-events", type=int, default=200_000)
    p.add_argument("--bind-ttl", type=float, default=36_000.0)
    p.add_argument("--label-window", type=float, default=90.0)
    a = p.parse_args()
    data = load_picodomain_stream(
        a.log_dir, a.red_log,
        max_events=a.max_events, bind_ttl=a.bind_ttl, label_window=a.label_window,
    )
    span = int(data.t.max() - data.t.min())
    print(f"StreamData ready: {tuple(data.msg.shape)} msg, span={span}s ({span / 86400:.2f} d), "
          f"anomalous fraction = {float(data.y.float().mean()):.4f}")
    sys.exit(0)
