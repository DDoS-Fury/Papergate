import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    MeanAggregator,
)
from torch_geometric.nn import TransformerConv

from graphagate.model.neighbor import MessageNeighborLoader

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc, num_hops=3):
        super().__init__()
        self.time_enc = time_enc
        self.num_hops = num_hops
        edge_dim = msg_dim + time_enc.out_channels
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels, out_channels, heads=4, dropout=0.1, edge_dim=edge_dim, concat=False))
        for _ in range(num_hops - 1):
            self.convs.append(TransformerConv(out_channels, out_channels, heads=4, dropout=0.1, edge_dim=edge_dim, concat=False))
        self.norms = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(num_hops)])

    def forward(self, x, last_update, edge_index, t, msg):
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

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, msg_dim, node_feat_dim, hash_dim):
        super().__init__()
        # Static node attributes (role / clearance / device tier) are concatenated
        # for both endpoints: they carry the policy-relevant signal that separates a
        # policy-violation anomaly (same edge features as benign) from benign traffic.
        self.lin1 = nn.Linear(in_channels * 2 + msg_dim + (node_feat_dim + hash_dim) * 2, in_channels)
        self.lin2 = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst, msg, feat_src, feat_dst):
        h = torch.cat([z_src, z_dst, msg, feat_src, feat_dst], dim=-1)
        h = self.lin1(h).relu()
        return self.lin2(h)

class ZTATemporalGraphNetwork(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, msg_dim, memory_dim=64, time_dim=32, num_hops=2, hash_buckets=10000, hash_dim=16):
        super().__init__()

        self.num_hops = num_hops
        # Kept for (re)building the temporal neighbour loader, which is not an
        # nn.Module and so lives outside the state_dict (see init_neighbor_loader).
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim

        # Static per-node attributes (role / clearance / device tier), indexed by the
        # global node id. Populated from the data at train time and persisted in the
        # state_dict; dynamic entities admitted at serving time write their slot here.
        self.register_buffer("node_feat", torch.zeros(num_nodes, node_feat_dim))
        
        # Hashed Identity Trick buffer
        self.register_buffer("node_hash", torch.zeros(num_nodes, dtype=torch.long))
        self.hash_emb = nn.Embedding(hash_buckets, hash_dim)

        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(msg_dim, memory_dim, time_dim),
            aggregator_module=MeanAggregator(),
        )

        self.gnn = GraphAttentionEmbedding(
            in_channels=memory_dim + node_feat_dim + hash_dim,
            out_channels=memory_dim,
            msg_dim=msg_dim,
            time_enc=self.memory.time_enc,
            num_hops=num_hops,
        )

        self.link_pred = LinkPredictor(
            in_channels=memory_dim, msg_dim=msg_dim, node_feat_dim=node_feat_dim, hash_dim=hash_dim
        )

        # Dedicated structural-compatibility head: projects the (identity-aware)
        # embeddings and scores a src/dst pair by cosine similarity, scaled by a
        # learnable temperature. Unlike the concat-MLP feature head — which keys on the
        # edge message and static attributes — this head measures whether the pair
        # "belongs together" given the entities' history, the only signal that flags a
        # valid-but-non-habitual access (lateral movement).
        self.struct_proj = nn.Sequential(
            nn.Linear(memory_dim, memory_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim * 2, memory_dim)
        )
        self.struct_scale = nn.Parameter(torch.tensor(5.0))

    def init_neighbor_loader(self, size, device=None):
        """Create (or recreate) the bounded temporal neighbour loader on ``device``.

        Called after ``.to(device)`` because the loader holds plain tensors that
        ``nn.Module.to`` does not move. The loader is intentionally outside the
        state_dict; its buffers are persisted separately (see serve_tgn.save_model).
        """
        self.neighbor_loader = MessageNeighborLoader(
            num_nodes=self.num_nodes, size=size, msg_dim=self.msg_dim, device=device, k_hops=self.num_hops
        )
        return self.neighbor_loader

    def embed(self, n_id, edge_index, hist_t, hist_msg):
        """Node embeddings for ``n_id`` via attention over their temporal neighbours.

        ``edge_index`` is relabelled to local positions in ``n_id`` and
        ``hist_t`` / ``hist_msg`` are the corresponding *historical* edge attributes
        supplied by the neighbour loader — not the event currently being scored.
        """
        z, last_update = self.memory(n_id)
        nf = self.node_feat[n_id]
        h_idx = self.node_hash[n_id]
        he = self.hash_emb(h_idx)
        x = torch.cat([z, nf, he], dim=-1)  # identity-aware node features
        z = self.gnn(x, last_update, edge_index, hist_t, hist_msg)
        return z

    def score(self, z, nf, h_idx, src_local, dst_local, cur_msg):
        """Benign-vs-anomalous logit for ``src_local -> dst_local`` carrying ``cur_msg``.

        Sum of two complementary signals (shared by training and serving):
          * feature head — concat-MLP over embeddings, message and static attributes
            (catches policy / contextual anomalies);
          * structural head — scaled cosine compatibility of the projected embeddings
            (catches lateral movement: a valid-but-non-habitual src/dst pairing).
        """
        he = self.hash_emb(h_idx)
        feat_with_hash = torch.cat([nf, he], dim=-1)
        feat = self.link_pred(
            z[src_local], z[dst_local], cur_msg, feat_with_hash[src_local], feat_with_hash[dst_local]
        ).squeeze(-1)
        hs = F.normalize(self.struct_proj(z[src_local]), dim=-1)
        hd = F.normalize(self.struct_proj(z[dst_local]), dim=-1)
        struct = self.struct_scale * (hs * hd).sum(-1)
        return feat + struct

    def forward(self, n_id, edge_index, hist_t, hist_msg, src_local, dst_local, cur_msg):
        """Score the current edge(s) ``src_local -> dst_local`` carrying ``cur_msg``.

        Embeddings come from the historical neighbourhood (``edge_index`` / ``hist_*``);
        ``src_local`` / ``dst_local`` index the queried endpoints within ``n_id``, and
        ``cur_msg`` is the message of the event under evaluation.
        """
        z = self.embed(n_id, edge_index, hist_t, hist_msg)
        nf = self.node_feat[n_id]
        h_idx = self.node_hash[n_id]
        return self.score(z, nf, h_idx, src_local, dst_local, cur_msg)

