"""Feature preprocessing utilities — z-score standardization of numeric features.

Standardizes ONLY the numeric feature block (schema.NUMERIC_SLICE) of the dense
node feature matrix [N, FEATURE_DIM], leaving the node-type one-hot block
(schema.TYPE_SLICE) unchanged.  Statistics are computed over real (non-padding)
nodes across a pool of training graphs.

Deployment note
---------------
This preprocessing step runs OUTSIDE the model (it is not part of the exported
ONNX graph).  At ONNX/Go deployment the same z-score transform must be applied
to every input feature matrix *before* passing it to the ONNX runtime.  The
required parameters (``feat_mean`` and ``feat_std``, each a list of
``NUM_NUMERIC_FEATURES`` floats) are stored in ``norm_stats.json`` under the
keys ``"feat_mean"`` and ``"feat_std"``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from graphagate.data.schema import NUM_NUMERIC_FEATURES, NUMERIC_SLICE

__all__ = ["compute_feature_stats", "standardize"]


def compute_feature_stats(
    x_list: list[Tensor],
    mask_list: list[Tensor],
) -> tuple[Tensor, Tensor]:
    """Compute per-numeric-feature mean and std over a pool of graphs.

    Only real nodes (mask == 1) contribute to the statistics.  Padding rows
    are excluded so they do not bias the estimates.

    Parameters
    ----------
    x_list:
        List of dense feature matrices, each ``[N_i, FEATURE_DIM]``.
    mask_list:
        Corresponding masks, each ``[N_i]`` with 1.0 for real nodes and 0.0
        for padding.

    Returns
    -------
    mean : Tensor
        Per-feature mean over all real nodes, shape ``[NUM_NUMERIC_FEATURES]``,
        dtype float32.
    std : Tensor
        Per-feature standard deviation, shape ``[NUM_NUMERIC_FEATURES]``,
        dtype float32.  Clamped to a minimum of 1e-6 so division is always safe.
    """
    # Collect numeric columns from all real nodes across all graphs.
    numeric_chunks: list[Tensor] = []
    for x, mask in zip(x_list, mask_list):
        # Boolean selection of real rows.
        real = mask.bool()                          # [N_i]
        numeric_chunks.append(x[real, NUMERIC_SLICE].float())  # [n_real, NUM_NUMERIC_FEATURES]

    # Concatenate all real-node numeric feature rows.
    all_numeric = torch.cat(numeric_chunks, dim=0)  # [total_real, NUM_NUMERIC_FEATURES]

    mean = all_numeric.mean(dim=0)                  # [NUM_NUMERIC_FEATURES]
    std = all_numeric.std(dim=0, unbiased=True)     # [NUM_NUMERIC_FEATURES]
    std = std.clamp(min=1e-6)

    return mean.float(), std.float()


def standardize(
    x: Tensor,
    mean: Tensor,
    std: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Apply z-score standardization to the numeric feature block of *x*.

    The one-hot node-type columns (schema.TYPE_SLICE) are left unchanged.
    This function returns a *copy* of *x*; the input is not mutated.

    Parameters
    ----------
    x:
        Dense feature matrix ``[N, FEATURE_DIM]``.
    mean:
        Per-numeric-feature mean ``[NUM_NUMERIC_FEATURES]``, from
        :func:`compute_feature_stats`.
    std:
        Per-numeric-feature std ``[NUM_NUMERIC_FEATURES]`` (already clamped),
        from :func:`compute_feature_stats`.
    mask:
        Optional ``[N]`` float/bool tensor (1.0 = real node, 0.0 = padding).
        When provided, padding rows are zeroed out in the returned tensor so
        that padding nodes remain neutral (all zeros) after standardization.

    Returns
    -------
    Tensor
        New ``[N, FEATURE_DIM]`` float32 tensor with standardized numeric block.
    """
    x_out = x.clone().float()

    # Z-score only the numeric feature block.
    x_out[:, NUMERIC_SLICE] = (x_out[:, NUMERIC_SLICE] - mean) / std

    # Zero out padding rows so they remain neutral in the loss / score.
    if mask is not None:
        m = mask.to(x_out.dtype)    # [N] float
        x_out = x_out * m[:, None]  # broadcast over feature dim

    return x_out
