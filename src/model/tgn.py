import torch
import torch.nn as nn
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
)
from torch_geometric.nn import TransformerConv

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=2,
                                    dropout=0.1, edge_dim=edge_dim, concat=False)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, msg_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_channels * 2 + msg_dim, in_channels)
        self.lin2 = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst, msg):
        h = torch.cat([z_src, z_dst, msg], dim=-1)
        h = self.lin1(h).relu()
        return self.lin2(h)

class ZTATemporalGraphNetwork(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, msg_dim, memory_dim=64, time_dim=32):
        super().__init__()
        
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        
        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=memory_dim,
            msg_dim=msg_dim,
            time_enc=self.memory.time_enc,
        )
        
        self.link_pred = LinkPredictor(in_channels=memory_dim, msg_dim=msg_dim)

    def forward(self, n_id, edge_index, t, msg):
        """
        Calculates node embeddings and link predictions.
        """
        # Get memory
        z, last_update = self.memory(n_id)
        
        # Calculate embedding
        z = self.gnn(z, last_update, edge_index, t, msg)
        
        # Predict links for the edges in the batch
        src, dst = edge_index
        pred = self.link_pred(z[src], z[dst], msg)
        return pred

