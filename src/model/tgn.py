import hashlib

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


def stable_hash(key, buckets: int) -> int:
    """Deterministic bucket for an entity ``key`` in ``[0, buckets)``.

    Uses BLAKE2b over the key's string form so the bucket assignment is identical
    across processes and machines. The builtin ``hash()`` is salted per process
    (``PYTHONHASHSEED``), which would give the *same* entity a *different* hashed
    identity across restarts / training runs — breaking both reproducibility and
    the inductive hashed-identity guarantee for entities admitted at serving time.
    """
    digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets

class GraphAttentionEmbedding(nn.Module):
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
    def __init__(self, in_channels, msg_dim, node_feat_dim, hash_dim, time_dim, hist_feat_dim=0, hidden_layers=2):
        super().__init__()
        # Static node attributes (role / clearance / device tier) are concatenated
        # for both endpoints: they carry the policy-relevant signal that separates a
        # policy-violation anomaly (same edge features as benign) from benign traffic.
        # ``hist_feat_dim`` adds the explicit interaction-history features for the scored
        # src→dst pair (how habitual this access is) — the runtime-derivable, non-circular
        # novelty signal that, combined with the temporal memory, flags lateral movement.
        self.lin1 = nn.Linear(
            in_channels * 2 + msg_dim + (node_feat_dim + hash_dim) * 2 + time_dim * 2 + hist_feat_dim,
            in_channels,
        )
        self.lin_mid = nn.Linear(in_channels, in_channels)
        # Extra hidden layers beyond the historical two (lin1 + lin_mid). Empty when
        # ``hidden_layers<=2`` so the state_dict keys stay identical to older checkpoints
        # (back-compat); ``hidden_layers=3`` inserts one extra Linear before the output.
        self.lin_extra = nn.ModuleList(
            nn.Linear(in_channels, in_channels) for _ in range(max(0, hidden_layers - 2))
        )
        self.lin2 = nn.Linear(in_channels, 1)

    def forward(self, z_src, z_dst, msg, feat_src, feat_dst, recency_enc, src_recency_enc, hist_feats):
        h = torch.cat(
            [z_src, z_dst, msg, feat_src, feat_dst, recency_enc, src_recency_enc, hist_feats], dim=-1
        )
        h = self.lin1(h).relu()
        h = self.lin_mid(h).relu()
        for layer in self.lin_extra:
            h = layer(h).relu()
        return self.lin2(h)

class ZTATemporalGraphNetwork(nn.Module):
    def __init__(self, num_nodes, node_feat_dim, msg_dim, memory_dim=64, time_dim=32, num_hops=2, hash_buckets=10000, hash_dim=16, hist_feat_dim=6, gnn_heads=4, link_pred_hidden_layers=2):
        super().__init__()

        self.num_hops = num_hops
        # Kept for (re)building the temporal neighbour loader, which is not an
        # nn.Module and so lives outside the state_dict (see init_neighbor_loader).
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.hist_feat_dim = hist_feat_dim

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
            heads=gnn_heads,
        )

        self.link_pred = LinkPredictor(
            in_channels=memory_dim, msg_dim=msg_dim, node_feat_dim=node_feat_dim, hash_dim=hash_dim,
            time_dim=time_dim, hist_feat_dim=hist_feat_dim, hidden_layers=link_pred_hidden_layers,
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
        self.last_contact = {}
        # Interaction-history counters (benign-gated, like last_contact): how many times
        # each src→dst pair and each src have been committed to memory. They feed
        # ``compute_hist_feats``; they are persisted alongside last_contact (not in the
        # state_dict) and reset/purged on the same events.
        self.pair_count = {}
        self.src_count = {}
        # Kill-chain precursor state: per-entity timestamp of the last alert (Snort /
        # detected anomaly). Lateral movement follows a recon alert on the SAME entity,
        # but the predict-then-update gate drops that precursor from the TGN memory — so
        # it is carried here and used as a time-decayed SERVING-TIME prior (see
        # serve_tgn.precursor_boost). It is NOT a trained input (benign-only training
        # would make it a dead feature). Persisted/purged like last_contact.
        self.recent_alert = {}
        self.precursor_half_life = 100000.0
        self.precursor_max_boost = 3.0

        # Ablation switches (runtime-only, not persisted): the normal model keeps them
        # ON. Used by the ablation driver to isolate the contribution of the structural
        # head, the hashed-identity embedding, the explicit history features and the
        # kill-chain precursor prior. Serving never toggles these.
        self.use_struct_head = True
        self.use_hash_identity = True
        self.use_hist_feats = True
        self.use_precursor = True

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
        if not self.use_hash_identity:
            he = torch.zeros_like(he)  # ablation: drop the hashed-identity signal
        x = torch.cat([z, nf, he], dim=-1)  # identity-aware node features
        z = self.gnn(x, last_update, edge_index, hist_t, hist_msg)
        return z

    def _hist_triplet(self, src_ids, dst_ids, device):
        """``[log1p(pair_count), log1p(src_count), pair_count/(src_count+1)]`` per pair."""
        pc = torch.tensor(
            [self.pair_count.get((int(s), int(d)), 0) for s, d in zip(src_ids, dst_ids)],
            dtype=torch.float, device=device,
        )
        sc = torch.tensor(
            [self.src_count.get(int(s), 0) for s in src_ids], dtype=torch.float, device=device,
        )
        return torch.stack([torch.log1p(pc), torch.log1p(sc), pc / (sc + 1.0)], dim=-1)

    def compute_hist_feats(self, src_ids, dst_ids, device, aux_src_ids=None):
        """Per-event interaction-history features for the directed ``src→dst`` pairs (6-dim).

        First triplet: how often this exact pair was seen, how active the src is, and the
        fraction of the src's traffic that habitually targets this dst. A never-seen pair
        from an active src (ratio≈0) is the novelty cue for lateral movement — but benign
        *exploration* shares it, so this signal is only discriminative in combination
        with the temporal memory / structural head (that is the honest, non-degenerate
        contribution). Counts come from benign-gated history only (anti-poisoning).

        Second triplet: the same statistics for the auxiliary ``aux_src→dst`` pairs.
        On the access edge (user→resource) the aux src is the DEVICE: its per-resource
        habituality counters replace the direct device→resource temporal edge of the
        v1 schema, so an infected machine's unusual reach stays visible without a 4th
        edge per request. ``aux_src_ids=None`` (binding edges, datasets without a
        device entity) zero-pads the second triplet.

        ``src_ids`` / ``dst_ids`` / ``aux_src_ids`` are iterables of *global* node ids.
        """
        base = self._hist_triplet(src_ids, dst_ids, device)
        if aux_src_ids is None:
            aux = torch.zeros_like(base)
        else:
            aux = self._hist_triplet(aux_src_ids, dst_ids, device)
        return torch.cat([base, aux], dim=-1)

    def score(self, z, nf, h_idx, src_local, dst_local, cur_msg, delta_t, delta_t_src, hist_feats):
        """Benign-vs-anomalous logit for ``src_local -> dst_local`` carrying ``cur_msg``.

        Sum of two complementary signals (shared by training and serving):
          * feature head — concat-MLP over embeddings, message, static attributes and the
            explicit interaction-history features (catches policy / contextual anomalies
            and supplies the novelty cue for lateral movement);
          * structural head — scaled cosine compatibility of the projected embeddings
            (catches lateral movement: a valid-but-non-habitual src/dst pairing).
        """
        he = self.hash_emb(h_idx)
        if not self.use_hash_identity:
            he = torch.zeros_like(he)  # ablation: drop the hashed-identity signal
        feat_with_hash = torch.cat([nf, he], dim=-1)
        recency_enc = self.memory.time_enc(delta_t)
        src_recency_enc = self.memory.time_enc(delta_t_src)
        if not self.use_hist_feats:
            hist_feats = torch.zeros_like(hist_feats)  # ablation: drop history features
        feat = self.link_pred(
            z[src_local], z[dst_local], cur_msg, feat_with_hash[src_local], feat_with_hash[dst_local],
            recency_enc, src_recency_enc, hist_feats,
        ).squeeze(-1)
        if not self.use_struct_head:
            return feat  # ablation: feature head only (no structural compatibility head)
        hs = F.normalize(self.struct_proj(z[src_local]), dim=-1)
        hd = F.normalize(self.struct_proj(z[dst_local]), dim=-1)
        struct = self.struct_scale * (hs * hd).sum(-1)
        return feat + struct

    def forward(self, n_id, edge_index, hist_t, hist_msg, src_local, dst_local, cur_msg, delta_t, delta_t_src, hist_feats):
        """Score the current edge(s) ``src_local -> dst_local`` carrying ``cur_msg``.

        Embeddings come from the historical neighbourhood (``edge_index`` / ``hist_*``);
        ``src_local`` / ``dst_local`` index the queried endpoints within ``n_id``,
        ``cur_msg`` is the message of the event under evaluation, and ``hist_feats`` are
        the precomputed interaction-history features for the scored pairs.
        """
        z = self.embed(n_id, edge_index, hist_t, hist_msg)
        nf = self.node_feat[n_id]
        h_idx = self.node_hash[n_id]
        return self.score(z, nf, h_idx, src_local, dst_local, cur_msg, delta_t, delta_t_src, hist_feats)

