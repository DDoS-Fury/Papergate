"""Synthetic benign IAM access graph generation and anomaly injection.

Implements a homogeneous attributed graph following the ZTA/IAM schema defined
in :mod:`graphagate.data.schema`. Heterogeneity is encoded via the node-type
one-hot in the feature vector; the graph itself is a single undirected edge set.

Public API
----------
- :func:`generate_benign_graph` — build a fully benign IAM graph.
- :func:`inject_anomalies` — overlay structural / attribute anomalies.
- :func:`to_dense` — pad to a fixed-size (bucket) dense representation.
"""

from __future__ import annotations

import argparse
import copy
import random
from typing import NamedTuple

import numpy as np
import torch
from torch_geometric.data import Data

from graphagate.config import DEFAULT_CONFIG, SyntheticConfig
from graphagate.data.schema import (
    FEATURE_DIM,
    NODE_TYPE_TO_IDX,
    NUMERIC_FEATURE_TO_IDX,
    NUM_NODE_TYPES,
    node_feature,
    stack_features,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _seed_all(seed: int) -> np.random.Generator:
    """Seed Python random, NumPy, and Torch for determinism; return a rng."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _clamp01(x: float) -> float:
    """Clamp a value to [0, 1]."""
    return float(max(0.0, min(1.0, x)))


def _daypart_benign(rng: np.random.Generator) -> dict[str, float]:
    """Return benign hour-fraction features (working-hours concentrated)."""
    # Base fractions: night tiny, morning/afternoon dominant, evening small
    night = rng.uniform(0.0, 0.05)
    evening = rng.uniform(0.02, 0.12)
    morning_frac = rng.uniform(0.35, 0.55)
    afternoon_frac = 1.0 - night - evening - morning_frac
    afternoon_frac = max(0.0, afternoon_frac)
    # Renormalise to exactly 1
    total = night + morning_frac + afternoon_frac + evening
    return {
        "hour_night": night / total,
        "hour_morning": morning_frac / total,
        "hour_afternoon": afternoon_frac / total,
        "hour_evening": evening / total,
    }


# ---------------------------------------------------------------------------
# Node-ID bookkeeping
# ---------------------------------------------------------------------------


class _NodeRanges(NamedTuple):
    """Global node-index offsets for each entity type."""

    user_start: int
    user_end: int        # exclusive
    device_start: int
    device_end: int
    ip_start: int
    ip_end: int
    resource_start: int
    resource_end: int
    role_start: int
    role_end: int
    session_start: int
    session_end: int

    @property
    def total(self) -> int:
        return self.session_end

    def user_ids(self) -> range:
        return range(self.user_start, self.user_end)

    def device_ids(self) -> range:
        return range(self.device_start, self.device_end)

    def ip_ids(self) -> range:
        return range(self.ip_start, self.ip_end)

    def resource_ids(self) -> range:
        return range(self.resource_start, self.resource_end)

    def role_ids(self) -> range:
        return range(self.role_start, self.role_end)

    def session_ids(self) -> range:
        return range(self.session_start, self.session_end)


def _build_ranges(cfg: SyntheticConfig, num_sessions: int) -> _NodeRanges:
    u = cfg.num_users
    d = cfg.num_devices
    ip = cfg.num_ip_subnets
    r = cfg.num_resources
    ro = cfg.num_roles
    s = num_sessions

    offsets: list[int] = []
    cursor = 0
    for count in (u, d, ip, r, ro, s):
        offsets.append(cursor)
        cursor += count

    return _NodeRanges(
        user_start=offsets[0],
        user_end=offsets[0] + u,
        device_start=offsets[1],
        device_end=offsets[1] + d,
        ip_start=offsets[2],
        ip_end=offsets[2] + ip,
        resource_start=offsets[3],
        resource_end=offsets[3] + r,
        role_start=offsets[4],
        role_end=offsets[4] + ro,
        session_start=offsets[5],
        session_end=offsets[5] + s,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_benign_graph(
    cfg: SyntheticConfig | None = None,
    seed: int | None = None,
) -> Data:
    """Generate a fully benign IAM access graph.

    Args:
        cfg: generation parameters; ``None`` falls back to
            ``DEFAULT_CONFIG.synthetic``.
        seed: overrides ``cfg.seed`` when provided.

    Returns:
        A PyG :class:`~torch_geometric.data.Data` with attributes ``x``
        (float32, ``[N, 14]``), ``edge_index`` (long, ``[2, E]``,
        undirected), ``node_type`` (long, ``[N]``), and ``y`` (long,
        ``[N]``, all zeros).
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.synthetic
    effective_seed = seed if seed is not None else cfg.seed
    rng = _seed_all(effective_seed)

    # -----------------------------------------------------------------------
    # 1. Decide entity counts and session assignment per user
    # -----------------------------------------------------------------------
    sessions_per_user = rng.integers(
        cfg.sessions_per_user[0],
        cfg.sessions_per_user[1] + 1,
        size=cfg.num_users,
    )
    total_sessions = int(sessions_per_user.sum())
    nr = _build_ranges(cfg, total_sessions)

    # -----------------------------------------------------------------------
    # 2. Assign roles to users (each user gets exactly one role)
    # -----------------------------------------------------------------------
    user_role = rng.integers(0, cfg.num_roles, size=cfg.num_users)  # local idx

    # -----------------------------------------------------------------------
    # 3. Assign resources to roles (role→resource grants)
    # -----------------------------------------------------------------------
    role_resources: list[list[int]] = []
    for _ in range(cfg.num_roles):
        n = int(rng.integers(cfg.resources_per_role[0], cfg.resources_per_role[1] + 1))
        chosen = rng.choice(cfg.num_resources, size=min(n, cfg.num_resources), replace=False).tolist()
        role_resources.append(chosen)

    # -----------------------------------------------------------------------
    # 4. Assign devices to users (user→device "uses" edges)
    # -----------------------------------------------------------------------
    user_devices: list[list[int]] = []
    for _ in range(cfg.num_users):
        n = int(rng.integers(cfg.devices_per_user[0], cfg.devices_per_user[1] + 1))
        chosen = rng.choice(cfg.num_devices, size=min(n, cfg.num_devices), replace=False).tolist()
        user_devices.append(chosen)

    # -----------------------------------------------------------------------
    # 5. Assign IP subnets (stable small set per user)
    # -----------------------------------------------------------------------
    user_ips: list[list[int]] = []
    for _ in range(cfg.num_users):
        n = int(rng.integers(1, 4))  # 1–3 subnets per user
        chosen = rng.choice(cfg.num_ip_subnets, size=min(n, cfg.num_ip_subnets), replace=False).tolist()
        user_ips.append(chosen)

    # -----------------------------------------------------------------------
    # 6. Build edge list (global node IDs)
    # -----------------------------------------------------------------------
    edges: list[tuple[int, int]] = []

    session_cursor = 0
    # Per-session metadata for feature computation
    session_user: list[int] = []          # which user owns this session (local idx)
    session_device: list[int] = []        # which device (local idx, -1 if none)
    session_ip: list[int] = []            # which ip_subnet (local idx)
    session_resources: list[list[int]] = []  # which resources (local idx)

    for u_local in range(cfg.num_users):
        u_global = nr.user_start + u_local
        role_local = int(user_role[u_local])
        role_global = nr.role_start + role_local

        # user —has→ role
        edges.append((u_global, role_global))

        # user —uses→ devices
        for d_local in user_devices[u_local]:
            d_global = nr.device_start + d_local
            edges.append((u_global, d_global))

        n_sess = int(sessions_per_user[u_local])
        for _ in range(n_sess):
            s_global = nr.session_start + session_cursor

            # user —owns→ session
            edges.append((u_global, s_global))

            # session uses one of the user's own devices
            dev_local = int(rng.choice(user_devices[u_local]))
            dev_global = nr.device_start + dev_local

            # session —from→ ip_subnet
            ip_local = int(rng.choice(user_ips[u_local]))
            ip_global = nr.ip_start + ip_local
            edges.append((s_global, ip_global))

            # session —accesses→ resources (only those granted by the user's role)
            granted = role_resources[role_local]
            n_access = int(rng.integers(1, min(len(granted), 4) + 1))
            accessed = rng.choice(granted, size=n_access, replace=False).tolist()
            for res_local in accessed:
                res_global = nr.resource_start + res_local
                edges.append((s_global, res_global))

            session_user.append(u_local)
            session_device.append(dev_local)
            session_ip.append(ip_local)
            session_resources.append(accessed)
            session_cursor += 1

    # role —grants→ resource
    for role_local, resources in enumerate(role_resources):
        role_global = nr.role_start + role_local
        for res_local in resources:
            res_global = nr.resource_start + res_local
            edges.append((role_global, res_global))

    # -----------------------------------------------------------------------
    # 7. Make edge_index undirected (both directions, dedup, no self-loops)
    # -----------------------------------------------------------------------
    edge_set: set[tuple[int, int]] = set()
    for a, b in edges:
        if a != b:
            edge_set.add((a, b))
            edge_set.add((b, a))
    src_list = [e[0] for e in edge_set]
    dst_list = [e[1] for e in edge_set]
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    # -----------------------------------------------------------------------
    # 8. Compute degree per node
    # -----------------------------------------------------------------------
    N = nr.total
    degree = np.zeros(N, dtype=np.float32)
    for nid in src_list:
        degree[nid] += 1
    max_deg = max(float(degree.max()), 1.0)

    # -----------------------------------------------------------------------
    # 9. Build per-node statistics for numeric features
    # -----------------------------------------------------------------------
    # activity_volume: count how many (directed) edge endpoints touch each node
    activity = np.zeros(N, dtype=np.float32)
    for nid in src_list:
        activity[nid] += 1
    max_activity = max(float(activity.max()), 1.0)

    # distinct_counterparts: #unique neighbours per node
    from collections import defaultdict
    neighbours: defaultdict[int, set[int]] = defaultdict(set)
    for a, b in edge_set:
        neighbours[a].add(b)
    # max distinct counterparts across all nodes
    max_counter = max((len(v) for v in neighbours.values()), default=1)

    # -----------------------------------------------------------------------
    # 10. Assign privilege to roles / resources
    # -----------------------------------------------------------------------
    # Deterministic within seed
    role_privilege = rng.uniform(0.2, 1.0, size=cfg.num_roles).astype(np.float32)
    resource_privilege = rng.uniform(0.1, 0.8, size=cfg.num_resources).astype(np.float32)

    # -----------------------------------------------------------------------
    # 11. Build feature matrix row by row
    # -----------------------------------------------------------------------
    features: list[np.ndarray] = []

    def _daypart() -> dict[str, float]:
        return _daypart_benign(rng)

    # --- users ---
    for u_local in range(cfg.num_users):
        uid = nr.user_start + u_local
        n_dev = len(user_devices[u_local])
        features.append(node_feature("user", {
            "degree": _clamp01(degree[uid] / max_deg),
            "distinct_counterparts": _clamp01(n_dev / cfg.devices_per_user[1]),
            "activity_volume": _clamp01(activity[uid] / max_activity),
            **_daypart(),
            "privilege": 0.0,
        }))

    # --- devices ---
    # distinct_counterparts = #users that use this device
    device_user_counts = np.zeros(cfg.num_devices, dtype=np.int32)
    for u_local in range(cfg.num_users):
        for d_local in user_devices[u_local]:
            device_user_counts[d_local] += 1
    max_dev_users = max(int(device_user_counts.max()), 1)

    for d_local in range(cfg.num_devices):
        did = nr.device_start + d_local
        features.append(node_feature("device", {
            "degree": _clamp01(degree[did] / max_deg),
            "distinct_counterparts": _clamp01(device_user_counts[d_local] / max_dev_users),
            "activity_volume": _clamp01(activity[did] / max_activity),
            **_daypart(),
            "privilege": 0.0,
        }))

    # --- ip_subnets ---
    for ip_local in range(cfg.num_ip_subnets):
        iid = nr.ip_start + ip_local
        features.append(node_feature("ip_subnet", {
            "degree": _clamp01(degree[iid] / max_deg),
            "distinct_counterparts": _clamp01(degree[iid] / max_deg),
            "activity_volume": _clamp01(activity[iid] / max_activity),
            **_daypart(),
            "privilege": 0.0,
        }))

    # --- resources ---
    for res_local in range(cfg.num_resources):
        rid = nr.resource_start + res_local
        features.append(node_feature("resource", {
            "degree": _clamp01(degree[rid] / max_deg),
            "distinct_counterparts": _clamp01(len(neighbours[rid]) / max(max_counter, 1)),
            "activity_volume": _clamp01(activity[rid] / max_activity),
            **_daypart(),
            "privilege": float(resource_privilege[res_local]),
        }))

    # --- roles ---
    for role_local in range(cfg.num_roles):
        roleid = nr.role_start + role_local
        n_res = len(role_resources[role_local])
        features.append(node_feature("role", {
            "degree": _clamp01(degree[roleid] / max_deg),
            "distinct_counterparts": _clamp01(n_res / cfg.resources_per_role[1]),
            "activity_volume": _clamp01(activity[roleid] / max_activity),
            **_daypart(),
            "privilege": float(role_privilege[role_local]),
        }))

    # --- sessions ---
    for s_idx in range(total_sessions):
        sid = nr.session_start + s_idx
        n_res = len(session_resources[s_idx])
        features.append(node_feature("session", {
            "degree": _clamp01(degree[sid] / max_deg),
            "distinct_counterparts": _clamp01(n_res / cfg.sessions_per_user[1]),
            "activity_volume": _clamp01(activity[sid] / max_activity),
            **_daypart(),
            "privilege": 0.0,
        }))

    # -----------------------------------------------------------------------
    # 12. Build node_type tensor
    # -----------------------------------------------------------------------
    type_labels: list[int] = (
        [NODE_TYPE_TO_IDX["user"]] * cfg.num_users
        + [NODE_TYPE_TO_IDX["device"]] * cfg.num_devices
        + [NODE_TYPE_TO_IDX["ip_subnet"]] * cfg.num_ip_subnets
        + [NODE_TYPE_TO_IDX["resource"]] * cfg.num_resources
        + [NODE_TYPE_TO_IDX["role"]] * cfg.num_roles
        + [NODE_TYPE_TO_IDX["session"]] * total_sessions
    )

    x_np = stack_features(features)
    x = torch.from_numpy(x_np)
    node_type_t = torch.tensor(type_labels, dtype=torch.long)
    y = torch.zeros(N, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        node_type=node_type_t,
        y=y,
    )
    # Store bookkeeping for anomaly injection
    data._nr = nr  # type: ignore[attr-defined]
    data._session_user = session_user  # type: ignore[attr-defined]
    data._session_resources = session_resources  # type: ignore[attr-defined]
    data._role_resources = role_resources  # type: ignore[attr-defined]
    data._user_role = user_role.tolist()  # type: ignore[attr-defined]
    data._user_devices = user_devices  # type: ignore[attr-defined]
    data._user_ips = user_ips  # type: ignore[attr-defined]
    return data


# ---------------------------------------------------------------------------


def inject_anomalies(
    data: Data,
    cfg: SyntheticConfig | None = None,
    seed: int | None = None,
) -> Data:
    """Inject structural and attribute anomalies into a benign graph.

    Returns a deep copy of *data* with ``y`` set to ``1`` on anomalous nodes.
    Does not mutate the input.

    Args:
        data: benign graph produced by :func:`generate_benign_graph`.
        cfg: generation config (``None`` → ``DEFAULT_CONFIG.synthetic``).
        seed: overrides ``cfg.seed`` when provided.

    Returns:
        New :class:`~torch_geometric.data.Data` with anomalies overlaid.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG.synthetic
    effective_seed = seed if seed is not None else cfg.seed
    rng = _seed_all(effective_seed + 99_999)  # different stream from generation

    data_copy = copy.deepcopy(data)

    N = data_copy.num_nodes
    n_anomalous = max(1, int(round(N * cfg.anomaly_fraction)))
    anomalous_nodes = rng.choice(N, size=n_anomalous, replace=False).tolist()
    anomalous_set = set(anomalous_nodes)

    # Build mutable adjacency for edge manipulation
    edge_index_np = data_copy.edge_index.numpy()
    extra_edges: list[tuple[int, int]] = []

    # Retrieve generation bookkeeping (may be absent if data was serialised)
    nr: _NodeRanges | None = getattr(data_copy, "_nr", None)
    role_resources: list[list[int]] | None = getattr(data_copy, "_role_resources", None)
    user_role: list[int] | None = getattr(data_copy, "_user_role", None)
    user_ips: list[list[int]] | None = getattr(data_copy, "_user_ips", None)
    user_devices: list[list[int]] | None = getattr(data_copy, "_user_devices", None)

    x_np: np.ndarray = data_copy.x.numpy().copy()

    for node_idx in anomalous_nodes:
        anomaly_type = int(rng.integers(0, 3))  # 0=structural, 1=attribute, 2=both

        # ------------------------------------------------------------------ #
        # STRUCTURAL anomaly                                                   #
        # ------------------------------------------------------------------ #
        if anomaly_type in (0, 2) and nr is not None:
            ntype_vec = x_np[node_idx, :NUM_NODE_TYPES]
            ntype_idx = int(ntype_vec.argmax())

            if ntype_idx == NODE_TYPE_TO_IDX["session"]:
                # Lateral movement: connect session to a resource NOT in its role
                s_local = node_idx - nr.session_start
                if 0 <= s_local < (nr.session_end - nr.session_start):
                    all_res = list(range(nr.resource_start, nr.resource_end))
                    granted = [nr.resource_start + r for r in (
                        role_resources[user_role[
                            getattr(data_copy, "_session_user", list(range(nr.user_end)))[s_local]
                        ]] if role_resources and user_role else []
                    )]
                    forbidden = [r for r in all_res if r not in granted]
                    if forbidden:
                        target = int(rng.choice(forbidden))
                        extra_edges.append((node_idx, target))
                        extra_edges.append((target, node_idx))

                # Foreign IP: connect to an ip not in user's stable set
                if user_ips and user_role:
                    sess_users = getattr(data_copy, "_session_user", None)
                    if sess_users:
                        u_local = sess_users[s_local]
                        own_ips = set(nr.ip_start + ip for ip in user_ips[u_local])
                        all_ips = list(range(nr.ip_start, nr.ip_end))
                        foreign_ips = [ip for ip in all_ips if ip not in own_ips]
                        if foreign_ips:
                            target_ip = int(rng.choice(foreign_ips))
                            extra_edges.append((node_idx, target_ip))
                            extra_edges.append((target_ip, node_idx))

            elif ntype_idx == NODE_TYPE_TO_IDX["user"]:
                # Privilege escalation: connect user to a role they do NOT have
                u_local = node_idx - nr.user_start
                if user_role and 0 <= u_local < len(user_role):
                    own_role = user_role[u_local]
                    other_roles = [
                        nr.role_start + r
                        for r in range(nr.role_end - nr.role_start)
                        if r != own_role
                    ]
                    if other_roles:
                        target_role = int(rng.choice(other_roles))
                        extra_edges.append((node_idx, target_role))
                        extra_edges.append((target_role, node_idx))

                # Also connect to a device belonging to another user
                if user_devices and 0 <= u_local < len(user_devices):
                    own_devs = set(nr.device_start + d for d in user_devices[u_local])
                    all_devs = list(range(nr.device_start, nr.device_end))
                    foreign_devs = [d for d in all_devs if d not in own_devs]
                    if foreign_devs:
                        target_dev = int(rng.choice(foreign_devs))
                        extra_edges.append((node_idx, target_dev))
                        extra_edges.append((target_dev, node_idx))

        # ------------------------------------------------------------------ #
        # ATTRIBUTE anomaly                                                    #
        # ------------------------------------------------------------------ #
        if anomaly_type in (1, 2):
            attr_kind = int(rng.integers(0, 3))

            if attr_kind == 0:
                # Shift activity to night
                total_h = float(
                    x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["hour_night"]]
                    + x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["hour_morning"]]
                    + x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["hour_afternoon"]]
                    + x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["hour_evening"]]
                )
                night_frac = float(rng.uniform(0.6, 0.9))
                rest = 1.0 - night_frac
                x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["hour_night"]] = night_frac
                x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["hour_morning"]] = rest * 0.3
                x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["hour_afternoon"]] = rest * 0.4
                x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["hour_evening"]] = rest * 0.3

            elif attr_kind == 1:
                # Inflate activity volume
                x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["activity_volume"]] = float(
                    rng.uniform(0.85, 1.0)
                )

            else:
                # Inflate privilege (even for types where it's normally 0 — anomaly signal)
                x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["privilege"]] = float(
                    rng.uniform(0.8, 1.0)
                )

    # ------------------------------------------------------------------ #
    # Merge extra edges                                                    #
    # ------------------------------------------------------------------ #
    if extra_edges:
        existing = set(zip(edge_index_np[0].tolist(), edge_index_np[1].tolist()))
        new_src, new_dst = [], []
        for a, b in extra_edges:
            if a != b and (a, b) not in existing:
                new_src.append(a)
                new_dst.append(b)
                existing.add((a, b))
        if new_src:
            extra_t = torch.tensor([new_src, new_dst], dtype=torch.long)
            data_copy.edge_index = torch.cat([data_copy.edge_index, extra_t], dim=1)

    # Recompute degree feature for affected nodes after edge changes
    new_ei = data_copy.edge_index.numpy()
    new_degree = np.zeros(N, dtype=np.float32)
    for nid in new_ei[0]:
        new_degree[nid] += 1
    new_max_deg = max(float(new_degree.max()), 1.0)
    for node_idx in anomalous_nodes:
        x_np[node_idx, NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX["degree"]] = _clamp01(
            new_degree[node_idx] / new_max_deg
        )

    data_copy.x = torch.from_numpy(x_np.astype(np.float32))

    # Mark anomalous labels
    y_new = torch.zeros(N, dtype=torch.long)
    for idx in anomalous_nodes:
        y_new[idx] = 1
    data_copy.y = y_new

    return data_copy


# ---------------------------------------------------------------------------


def to_dense(
    data: Data,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a graph to a fixed bucket size for the model.

    Args:
        data: PyG :class:`~torch_geometric.data.Data` graph.
        size: target number of nodes (bucket size). Must be ``>= data.num_nodes``.

    Returns:
        A tuple ``(x_pad, adj, mask)`` where:

        * ``x_pad`` — float32 ``[size, FEATURE_DIM]``, real nodes first, zero-padded.
        * ``adj`` — float32 ``[size, size]``, symmetric 0/1 adjacency, padding zeros.
        * ``mask`` — float32 ``[size]``, 1.0 for real nodes, 0.0 for padding.

    Raises:
        ValueError: if ``data.num_nodes > size``.
    """
    N = data.num_nodes
    if N > size:
        raise ValueError(
            f"data.num_nodes={N} exceeds bucket size={size}; choose a larger bucket."
        )

    # x_pad
    x_pad = torch.zeros(size, FEATURE_DIM, dtype=torch.float32)
    x_pad[:N] = data.x.float()

    # adj — symmetric, no self-loops
    adj = torch.zeros(size, size, dtype=torch.float32)
    if data.edge_index.numel() > 0:
        src = data.edge_index[0]
        dst = data.edge_index[1]
        # Filter out self-loops just in case
        mask_no_sl = src != dst
        src = src[mask_no_sl]
        dst = dst[mask_no_sl]
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0  # ensure symmetry (edge_index already undirected)

    # mask
    mask = torch.zeros(size, dtype=torch.float32)
    mask[:N] = 1.0

    return x_pad, adj, mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_stats(data: Data, cfg: SyntheticConfig, with_anomalies: bool) -> None:
    from graphagate.data.schema import NODE_TYPES, NODE_TYPE_TO_IDX

    N = data.num_nodes
    E_directed = data.edge_index.shape[1]
    E_undirected = E_directed // 2

    print(f"{'='*55}")
    print(f"  Graphagate synthetic IAM graph {'(+anomalies)' if with_anomalies else '(benign)'}")
    print(f"{'='*55}")
    print(f"  Total nodes : {N}")
    print(f"  Total edges : {E_undirected} undirected ({E_directed} directed)")
    print(f"  Feature dim : {data.x.shape[1]}")
    print()

    node_type_np = data.node_type.numpy()
    for t in NODE_TYPES:
        count = int((node_type_np == NODE_TYPE_TO_IDX[t]).sum())
        print(f"  {t:>12s} : {count:>4d} nodes")

    print()
    print(f"  x shape      : {list(data.x.shape)}  dtype={data.x.dtype}")
    print(f"  edge_index   : {list(data.edge_index.shape)}  dtype={data.edge_index.dtype}")
    print(f"  node_type    : {list(data.node_type.shape)}  dtype={data.node_type.dtype}")
    print(f"  y            : {list(data.y.shape)}  dtype={data.y.dtype}")

    if with_anomalies:
        n_anom = int(data.y.sum().item())
        print(f"\n  Anomalous nodes injected : {n_anom}  ({100*n_anom/N:.1f}% of total)")

    print(f"{'='*55}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry-point: ``python -m graphagate.data.synthetic --stats [--anomalies]``."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Graphagate IAM graph and print statistics."
    )
    parser.add_argument("--stats", action="store_true", default=True, help="Print node/edge statistics (default).")
    parser.add_argument("--anomalies", action="store_true", help="Inject anomalies and report them.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    args = parser.parse_args(argv)

    cfg = DEFAULT_CONFIG.synthetic
    data = generate_benign_graph(cfg=cfg, seed=args.seed)

    if args.anomalies:
        data = inject_anomalies(data, cfg=cfg, seed=args.seed)

    _print_stats(data, cfg, with_anomalies=args.anomalies)


if __name__ == "__main__":
    main()
