"""Temporal Graph Network embedding module
- :class:`GraphAttentionEmbedding`
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class GraphAttentionEmbedding(nn.Module):
    """``num_hops`` TransformerConv layers (LayerNorm, residual from the second) over
    historical edges whose attributes are ``[time_enc(Δt), msg]``."""

    def __init__(self, in_channels, out_channels, msg_dim, time_enc, num_hops=3, heads=4):
        super().__init__()
        self.time_enc = time_enc
        self.num_hops = num_hops
        edge_dim = msg_dim + time_enc.out_channels
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, out_channels, heads=heads, dropout=0.1, edge_dim=edge_dim, concat=False))
        for _ in range(num_hops - 1):
            self.convs.append(TransformerConv(out_channels, out_channels, heads=heads, dropout=0.1, edge_dim=edge_dim, concat=False))
        self.norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(num_hops)])

    def forward(self, x, last_update, edge_index, t, msg):
        """Embed ``x`` over edges ``edge_index`` with times ``t`` and messages ``msg``."""
        if edge_index.numel() == 0:
            edge_attr = torch.empty(0, msg.size(-1) + self.time_enc.out_channels, device=x.device)
        else:
            rel_t = last_update[edge_index[0]] - t
            rel_t_enc = self.time_enc(rel_t.to(x.dtype))
            edge_attr = torch.cat([rel_t_enc, msg], dim=-1)

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index, edge_attr)
            if i > 0:
                x = x + x_new  # Residual connection
            else:
                x = x_new
            x = self.norms[i](x)
            if i < len(self.convs) - 1:
                x = x.relu()
        return x
