"""DOMINANT-style dense Graph Autoencoder (GAE).

Dense formulation (no torch_geometric message-passing): every operation is a
plain matmul / elementwise op over fixed-size ``[N, N]`` / ``[N, F]`` tensors,
so the exported ONNX graph has fully static shapes and is TensorRT-friendly.

References
----------
- Ding et al., "Deep Anomaly Detection on Attributed Networks" (SDM 2019).
- Architecture document: "Architettura AI per ZTA e OPA".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from graphagate.config import ModelConfig

__all__ = ["DenseGCNLayer", "DominantGAE"]


class DenseGCNLayer(nn.Module):
    """Single dense graph convolution: H' = Â H W (+ bias).

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    out_dim:
        Output feature dimension.

    Notes
    -----
    ``a_hat`` passed to :meth:`forward` must already be the symmetric-normalised
    adjacency  D^{-1/2} (A+I) D^{-1/2}.  Normalisation is computed once per
    forward pass in :class:`DominantGAE` and shared across all layers.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, h: Tensor, a_hat_norm: Tensor) -> Tensor:
        """Apply one round of dense graph convolution.

        Parameters
        ----------
        h:
            Node feature matrix ``[N, in_dim]``.
        a_hat_norm:
            Pre-normalised adjacency ``[N, N]``.

        Returns
        -------
        Tensor
            Updated node features ``[N, out_dim]``.
        """
        # W applied first (cheaper): [N, out_dim]
        support = self.linear(h)
        # Graph aggregation: [N, out_dim]
        return torch.mm(a_hat_norm, support)


class DominantGAE(nn.Module):
    """DOMINANT Graph Autoencoder for unsupervised anomaly detection.

    Architecture
    ------------
    - **Encoder** – ``cfg.num_layers`` DenseGCN layers:
      ``feature_dim → hidden_dim → … → embed_dim``.
      ReLU + dropout between layers (not after the last).
    - **Structure decoder** – inner-product: ``sigmoid(Z Z^T)``.
    - **Attribute decoder** – two DenseGCN layers:
      ``embed_dim → hidden_dim → feature_dim``.

    DropEdge
    --------
    Applied **only during training**.  A symmetric Bernoulli mask (built from
    an upper-triangular random draw, symmetrised) multiplies the adjacency
    before normalisation.  In eval / export mode the adjacency is passed
    through unchanged, keeping the ONNX graph fully deterministic.

    Normalised adjacency
    --------------------
    Given (possibly DropEdge'd) adjacency ``A``:

    .. code-block:: text

        A_tilde = A + I
        d       = A_tilde.sum(-1)          # row-degree vector [N]
        D_inv_sqrt = diag(d^{-1/2})
        a_hat_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt

    Implemented without ``torch.diag`` calls (which fix N at trace time) by
    using broadcasting: ``d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]``.

    ONNX / TensorRT constraints
    ---------------------------
    - No data-dependent control flow; the DropEdge branch is gated by
      ``self.training`` (a Python bool — resolved at trace time).
    - No ``torch.where`` with boolean indexing, no ``.item()``, no dynamic
      shapes.
    - ``torch.eye`` is called with the static Python int ``N`` inferred from
      ``x.shape[0]`` — shape is symbolic but the op is supported.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # --- Encoder ---
        # Builds a sequence feature_dim -> hidden_dim -> ... -> embed_dim.
        # With num_layers=2: feature_dim -> hidden_dim -> embed_dim.
        enc_dims: list[int] = (
            [cfg.feature_dim]
            + [cfg.hidden_dim] * (cfg.num_layers - 1)
            + [cfg.embed_dim]
        )
        self.encoder_layers = nn.ModuleList(
            [DenseGCNLayer(enc_dims[i], enc_dims[i + 1]) for i in range(cfg.num_layers)]
        )

        # --- Attribute decoder ---
        # embed_dim -> hidden_dim -> feature_dim
        self.dec_attr_1 = DenseGCNLayer(cfg.embed_dim, cfg.hidden_dim)
        self.dec_attr_2 = DenseGCNLayer(cfg.hidden_dim, cfg.feature_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dropedge(self, adj: Tensor) -> Tensor:
        """Symmetric Bernoulli edge drop (training only).

        Builds an upper-triangular Bernoulli keep-mask, symmetrises it, and
        multiplies the adjacency element-wise.  The result preserves symmetry.
        Only called when ``self.training`` is ``True``.
        """
        N = adj.shape[0]
        # Upper-triangular keep mask (including diagonal)
        keep_prob = 1.0 - self.cfg.dropedge_p
        # Sample uniform [0,1) upper-triangular, threshold at keep_prob
        rand_upper = torch.rand(N, N, device=adj.device, dtype=adj.dtype)
        mask_upper = (rand_upper < keep_prob).to(adj.dtype)
        # Zero out lower triangle so we only work with upper
        triu_mask = torch.triu(mask_upper)
        # Symmetrise
        sym_mask = triu_mask + triu_mask.t() - torch.diag(torch.diag(triu_mask))
        return adj * sym_mask

    def _normalize_adj(self, adj: Tensor) -> Tensor:
        """Compute D^{-1/2} (A + I) D^{-1/2} via broadcasting.

        Parameters
        ----------
        adj:
            Raw 0/1 adjacency ``[N, N]``.

        Returns
        -------
        Tensor
            Symmetric-normalised adjacency ``[N, N]``.
        """
        N = adj.shape[0]
        eps = 1e-8
        a_tilde = adj + torch.eye(N, device=adj.device, dtype=adj.dtype)
        deg = a_tilde.sum(dim=-1)                          # [N]
        d_inv_sqrt = torch.rsqrt(deg.clamp(min=eps))      # [N]
        # Broadcast: D^{-1/2} @ A_tilde @ D^{-1/2}
        return d_inv_sqrt[:, None] * a_tilde * d_inv_sqrt[None, :]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, x: Tensor, adj: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode + decode.

        Parameters
        ----------
        x:
            Node feature matrix ``[N, feature_dim]``.
        adj:
            Symmetric 0/1 dense adjacency ``[N, N]``.

        Returns
        -------
        a_hat : Tensor ``[N, N]``
            Reconstructed adjacency in ``[0, 1]``.
        x_hat : Tensor ``[N, feature_dim]``
            Reconstructed node features.
        z : Tensor ``[N, embed_dim]``
            Node embeddings.
        """
        # 1. DropEdge (training only — Python bool, resolved at trace time)
        if self.training:
            adj_de = self._dropedge(adj)
        else:
            adj_de = adj

        # 2. Normalised adjacency
        a_hat_norm = self._normalize_adj(adj_de)

        # 3. Encoder
        h = x
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h, a_hat_norm)
            if i < len(self.encoder_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.cfg.dropout, training=self.training)
        z = h  # [N, embed_dim]

        # 4. Structure decoder: sigmoid(Z Z^T)
        a_hat = torch.sigmoid(torch.mm(z, z.t()))  # [N, N]

        # 5. Attribute decoder: two DenseGCN layers on Z
        x_dec = F.relu(self.dec_attr_1(z, a_hat_norm))
        x_hat = self.dec_attr_2(x_dec, a_hat_norm)  # [N, feature_dim]

        return a_hat, x_hat, z
