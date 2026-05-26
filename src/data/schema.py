"""IAM graph schema — the *shared contract* between the data layer and the model.

The architecture document models ZTA/IAM logs as a **heterogeneous** graph (users,
devices, IP subnets, resources, sessions, roles linked by access actions). For this
first version we render that heterogeneous graph as a **homogeneous attributed**
graph (DOMINANT-style): every node carries a fixed-size feature vector whose first
slots are a one-hot of its node type, followed by a shared block of behavioural
numeric features (zero where a feature does not apply to that type).

Keeping a single, fixed ``FEATURE_DIM`` for every node is what lets the encoder be a
plain dense GCN and the exported ONNX graph use fully static shapes (static bucketing
+ padding), as required by the document's low-latency / TensorRT constraints.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

# --- Node types -------------------------------------------------------------
# Order is part of the contract: it fixes the one-hot slot of each type.
NODE_TYPES: tuple[str, ...] = (
    "user",
    "device",
    "ip_subnet",
    "resource",
    "session",
    "role",
)
NODE_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(NODE_TYPES)}
NUM_NODE_TYPES: int = len(NODE_TYPES)

# --- Edge types (semantic) --------------------------------------------------
# The runtime graph is homogeneous (a single undirected edge set), but generation
# follows these semantic relations. Documented here so the data layer and any later
# heterogeneous extension share one vocabulary.
EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("user", "uses", "device"),
    ("user", "owns", "session"),
    ("user", "has", "role"),
    ("session", "from", "ip_subnet"),
    ("session", "accesses", "resource"),
    ("role", "grants", "resource"),
)

# --- Numeric feature block --------------------------------------------------
# Shared numeric slots, filled per node type (0.0 where not meaningful). All values
# are expected to be roughly normalised to [0, 1] by the data layer.
NUMERIC_FEATURES: tuple[str, ...] = (
    "degree",                # connectivity (normalised node degree)
    "distinct_counterparts",  # e.g. #distinct devices for a user, #users for a device
    "activity_volume",       # number of access events touching the node
    "hour_night",            # fraction of activity 00-06
    "hour_morning",          # fraction of activity 06-12
    "hour_afternoon",        # fraction of activity 12-18
    "hour_evening",          # fraction of activity 18-24
    "privilege",             # sensitivity / privilege level (role, resource)
)
NUM_NUMERIC_FEATURES: int = len(NUMERIC_FEATURES)
NUMERIC_FEATURE_TO_IDX: dict[str, int] = {f: i for i, f in enumerate(NUMERIC_FEATURES)}

# Total fixed feature dimension for every node.
FEATURE_DIM: int = NUM_NODE_TYPES + NUM_NUMERIC_FEATURES  # 6 + 8 = 14

# Slices into a feature vector.
TYPE_SLICE = slice(0, NUM_NODE_TYPES)
NUMERIC_SLICE = slice(NUM_NODE_TYPES, FEATURE_DIM)


def node_feature(node_type: str, numeric: Mapping[str, float] | None = None) -> np.ndarray:
    """Build a single fixed-size feature vector for a node.

    Args:
        node_type: one of :data:`NODE_TYPES`.
        numeric: optional mapping from names in :data:`NUMERIC_FEATURES` to values.

    Returns:
        ``float32`` array of length :data:`FEATURE_DIM`.
    """
    if node_type not in NODE_TYPE_TO_IDX:
        raise ValueError(f"unknown node_type {node_type!r}; expected one of {NODE_TYPES}")
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    vec[NODE_TYPE_TO_IDX[node_type]] = 1.0
    if numeric:
        for name, value in numeric.items():
            if name not in NUMERIC_FEATURE_TO_IDX:
                raise ValueError(f"unknown numeric feature {name!r}")
            vec[NUM_NODE_TYPES + NUMERIC_FEATURE_TO_IDX[name]] = float(value)
    return vec


def node_type_of(feature_matrix: np.ndarray) -> np.ndarray:
    """Recover the node-type index (argmax of the one-hot block) for each row."""
    return np.asarray(feature_matrix)[:, TYPE_SLICE].argmax(axis=1)


def stack_features(rows: Sequence[np.ndarray]) -> np.ndarray:
    """Stack per-node feature vectors into an ``[N, FEATURE_DIM]`` float32 matrix."""
    if not rows:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32)
    return np.vstack(rows).astype(np.float32)
