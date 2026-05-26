"""DOMINANT reconstruction losses and per-node anomaly scores.

Both functions support an optional boolean/float ``mask`` tensor ``[N]`` that
marks real nodes (1) vs. padding nodes (0).  When a mask is given, padding
nodes are excluded from all averages so they never inflate or deflate the loss.

References
----------
- Ding et al., "Deep Anomaly Detection on Attributed Networks" (SDM 2019).
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["reconstruction_loss", "node_anomaly_score"]

_EPS = 1e-8


def reconstruction_loss(
    adj: Tensor,
    a_hat: Tensor,
    x: Tensor,
    x_hat: Tensor,
    alpha: float,
    mask: Tensor | None = None,
    struct_pos_weight: float | None = None,
) -> Tensor:
    """DOMINANT combined reconstruction loss (scalar).

    .. math::

        \\mathcal{L} = \\alpha \\cdot \\mathcal{L}_{\\text{struct}}
                     + (1 - \\alpha) \\cdot \\mathcal{L}_{\\text{attr}}

    Structure error is a *positively-weighted* masked MSE over the real-node
    sub-block of the adjacency.  Attribute error is the masked MSE over
    real-node rows of the feature matrix.

    Positive-class weighting for structure
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Because real graphs are sparse, naively averaging ``(A - Â)²`` is
    dominated by the many zero entries and the model can achieve low loss by
    predicting near-zero everywhere.  A per-entry weight matrix

    .. math::

        W_{ij} = 1 + (\\theta - 1) \\cdot A_{ij}

    gives weight ``theta`` to real edges (``A==1``) and weight ``1`` to
    non-edges (``A==0``), then the structure error becomes

    .. math::

        \\mathcal{L}_{\\text{struct}}
            = \\frac{\\sum_{ij} W_{ij}\\,(A_{ij}-\\hat{A}_{ij})^2\\,m_i m_j}
                    {\\sum_{ij} W_{ij}\\,m_i m_j}

    where ``m_i`` is the mask (1 for real nodes, 0 for padding).

    Parameters
    ----------
    adj:
        Ground-truth 0/1 adjacency ``[N, N]``.
    a_hat:
        Reconstructed adjacency in ``[0, 1]``, ``[N, N]``.
    x:
        Node feature matrix ``[N, F]``.
    x_hat:
        Reconstructed node feature matrix ``[N, F]``.
    alpha:
        Trade-off coefficient (structure weight).  Must be in ``[0, 1]``.
    mask:
        Optional ``[N]`` float/bool tensor: 1 for real nodes, 0 for padding.
        When ``None`` all nodes contribute equally.
    struct_pos_weight:
        Positive-class weight ``theta`` for real edges in the structure term.

        * If a ``float`` is supplied, it is used directly as ``theta``.
        * If ``None`` (default), ``theta`` is computed automatically as
          ``(#neg) / (#pos)`` over the masked region (i.e. real-node×real-node
          sub-block when a mask is given, or the full matrix otherwise).
          ``theta`` is then clamped to ``[1.0, 50.0]``.  If there are no
          positive entries, ``theta = 1.0`` (falls back to plain MSE).

    Returns
    -------
    Tensor
        Scalar loss value.
    """
    struct_sq = (adj - a_hat) ** 2   # [N, N]
    attr_sq = (x - x_hat) ** 2       # [N, F]

    if mask is None:
        # --- compute theta ---
        if struct_pos_weight is not None:
            theta = float(struct_pos_weight)
        else:
            n_pos = adj.sum()
            n_neg = adj.numel() - n_pos
            if n_pos > 0:
                theta = float((n_neg / n_pos).clamp(1.0, 50.0))
            else:
                theta = 1.0

        # per-entry weight: 1 for non-edges, theta for edges
        W = 1.0 + (theta - 1.0) * adj                       # [N, N]
        struct_err = (W * struct_sq).sum() / W.sum().clamp(min=_EPS)
        attr_err = attr_sq.mean()
    else:
        m = mask.to(adj.dtype)                               # [N] float
        m_outer = m[:, None] * m[None, :]                   # [N, N]

        # --- compute theta restricted to the masked region ---
        if struct_pos_weight is not None:
            theta = float(struct_pos_weight)
        else:
            adj_masked = adj * m_outer
            n_pos = adj_masked.sum()
            n_neg = m_outer.sum() - n_pos
            if n_pos > 0:
                theta = float((n_neg / n_pos).clamp(1.0, 50.0))
            else:
                theta = 1.0

        # per-entry weight
        W = 1.0 + (theta - 1.0) * adj                       # [N, N]
        Wm = W * m_outer                                     # zero out padding pairs
        struct_err = (Wm * struct_sq).sum() / Wm.sum().clamp(min=_EPS)

        attr_weight = (m[:, None] * torch.ones_like(attr_sq)).sum().clamp(min=_EPS)
        attr_err = (attr_sq * m[:, None]).sum() / attr_weight

    return alpha * struct_err + (1.0 - alpha) * attr_err


def node_anomaly_score(
    adj: Tensor,
    a_hat: Tensor,
    x: Tensor,
    x_hat: Tensor,
    alpha: float,
    mask: Tensor | None = None,
) -> Tensor:
    """Per-node DOMINANT anomaly score (un-normalised).

    .. math::

        s_i = \\alpha \\cdot \\overline{(A_{i\\cdot} - \\hat{A}_{i\\cdot})^2}
            + (1 - \\alpha) \\cdot \\overline{(X_i - \\hat{X}_i)^2}

    Column-wise mean of the structure row error is masked by ``mask`` when
    given, so padding columns do not inflate ``s_i``.

    Parameters
    ----------
    adj:
        Ground-truth 0/1 adjacency ``[N, N]``.
    a_hat:
        Reconstructed adjacency ``[N, N]``.
    x:
        Node feature matrix ``[N, F]``.
    x_hat:
        Reconstructed node feature matrix ``[N, F]``.
    alpha:
        Trade-off coefficient (structure weight).
    mask:
        Optional ``[N]`` float/bool tensor marking real (1) vs. padding (0)
        nodes.  Only used to mask *columns* of the structure error row; the
        function returns a score for *every* node including padding nodes
        (callers should mask the output as needed).

    Returns
    -------
    Tensor
        Per-node anomaly score ``[N]`` (higher = more anomalous).
    """
    struct_sq = (adj - a_hat) ** 2   # [N, N]
    attr_sq = (x - x_hat) ** 2       # [N, F]

    if mask is None:
        struct_row_err = struct_sq.mean(dim=-1)    # [N]
    else:
        m = mask.to(adj.dtype)                     # [N]
        col_weight = m.sum().clamp(min=_EPS)
        struct_row_err = (struct_sq * m[None, :]).sum(dim=-1) / col_weight  # [N]

    attr_err_per_node = attr_sq.mean(dim=-1)       # [N]

    return alpha * struct_row_err + (1.0 - alpha) * attr_err_per_node
