"""Scoring heads of the ZTA Temporal Graph Network
- :class:`LinkPredictor`
- :class:`StructuralProjector`
"""

import torch
from torch import nn
import torch.nn.functional as F


class LinkPredictor(nn.Module):
    """Feature head: MLP over both endpoints' embeddings and static features (+ hashed
    identity), the edge message, both recency encodings and the pair's history features."""

    def __init__(self, in_channels, msg_dim, node_feat_dim, hash_dim, time_dim, hist_feat_dim=0, hidden_layers=2, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(
            in_channels * 2 + msg_dim + (node_feat_dim + hash_dim) * 2 + time_dim * 2 + hist_feat_dim,
            in_channels,
        )

        # residual blocks
        num_blocks = max(1, hidden_layers-1)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(in_channels, in_channels),
            )
            for _ in range(num_blocks)
        ])
        self.norm=nn.LayerNorm(in_channels)
        self.lin2=nn.Linear(in_channels, 1)


    def forward(self, z, feat, src, dst, msg, recency_enc, src_recency_enc, hist_feats):
        """Logit of each ``src[i] -> dst[i]`` pair; ``z`` / ``feat`` hold one row per endpoint node.

        ``lin1`` is applied by column blocks of its input layout ``[z_src, z_dst, msg, feat_src,
        feat_dst, recency_enc, src_recency_enc, hist_feats]`` (the state_dict layout, unchanged):
        the endpoint blocks are projected once per node row and gathered per pair, instead of
        once per (pair × negative) row. Same value up to float summation order.
        """
        c, f, m = z.size(-1), feat.size(-1), msg.size(-1)
        w_zs, w_zd, w_msg, w_fs, w_fd, w_rest = self.lin1.weight.split(
            [c, c, m, f, f, self.lin1.in_features - 2 * c - m - 2 * f], dim=1
        )
        node_w = torch.cat([torch.cat([w_zs, w_fs], 1), torch.cat([w_zd, w_fd], 1)], 0)
        h_src, h_dst = F.linear(torch.cat([z, feat], dim=-1), node_w).chunk(2, dim=-1)
        edge = torch.cat([msg, recency_enc, src_recency_enc, hist_feats], dim=-1)

        # --- new combined SiLU ---
        h = F.silu(F.linear(edge, torch.cat([w_msg, w_rest], 1), self.lin1.bias) + h_src[src] + h_dst[dst])

        for block in self.blocks:
            h = h + block(h)

        h = self.norm(h)
        return self.lin2(h)


EDGE_ACCESS = "user>res"
EDGE_DEV_USER = "dev>user"
EDGE_CFG_USER = "cfg>user"
EDGE_CFG_DEV = "cfg>dev"
EDGE_SRC_CFG = "src>cfg"
EDGE_SRC_DEV = "src>dev"


class StructuralProjector(nn.Module):
    """Structural head: projection mapping node embeddings into a metric space for cosine similarity scoring.
    Includes non-affine batch normalization to prevent dimensional collapse onto a single direction vector.
    """
    def __init__(self, in_channels, hidden_layers=2, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 2, in_channels),
        )
        self.bn = nn.BatchNorm1d(in_channels, affine=False) # non-affine to prevent collapse

    def forward(self, x):
        h = self.net(x)
        if h.size(0) > 1 or not self.training:
            h = self.bn(h)
        return h