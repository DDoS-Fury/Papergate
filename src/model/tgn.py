"""ZTA temporal graph network: TGN memory + graph attention over temporal neighbours.

- :func:`stable_hash`: process-independent hashed identity of an entity key.
- :class:`GraphAttentionEmbedding`: multi-hop attention over historical edges.
- :class:`LinkPredictor`: feature head of the edge score.
- :class:`ZTATemporalGraphNetwork`: the model, including the runtime state (recency,
  history counters, precursor alerts) kept outside the state_dict.
"""

import hashlib

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    MeanAggregator,
)

from graphagate.model.neighbor import MessageNeighborLoader
from graphagate.model.heads import LinkPredictor, StructuralProjector
from graphagate.model.gnn import GraphAttentionEmbedding
from graphagate.config import TGNConfig as _Cfg # unique source of config params

def stable_hash(key, buckets: int) -> int:
    """Deterministic bucket for an entity ``key`` in ``[0, buckets)``.

    BLAKE2b over ``str(key)``, not the per-process salted ``hash()``, so an entity keeps
    its bucket across processes, restarts and serving-time admissions.
    """
    digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


class ZTATemporalGraphNetwork(nn.Module):
    """Streaming TGN scoring directed edges of the ZTA access graph.

    Buffers (state_dict): ``node_feat`` static features, ``node_hash`` hashed identities,
    TGN ``memory``. Runtime state (plain attributes, benign-gated, persisted by
    ``serve_tgn.save_model``):
      * ``last_contact[(src, dst)]``: last commit time of the pair (recency input);
      * ``pair_count`` / ``src_count``: interaction-history counters (``compute_hist_feats``);
      * ``recent_alert[entity]``: last alert time (kill-chain precursor prior);
      * ``neighbor_loader``: bounded temporal neighbourhood (``init_neighbor_loader``).
    Serving knobs: ``precursor_half_life``, ``precursor_max_shift``, ``threshold_arm``,
    ``delta_t_cap``. The ``use_*`` switches (default on) exist for the ablations.
    """

    def __init__(self, num_nodes, node_feat_dim, msg_dim, memory_dim=64, time_dim=32, num_hops=2, hash_buckets=10000, hash_dim=16, hist_feat_dim=6, gnn_heads=4, link_pred_hidden_layers=2):
        super().__init__()

        self.num_hops = num_hops
        # Kept for (re)building the temporal neighbour loader, which is not an
        # nn.Module and so lives outside the state_dict (see init_neighbor_loader).
        self.num_nodes = num_nodes
        self.msg_dim = msg_dim
        self.hist_feat_dim = hist_feat_dim

        # Static node features by global node id; serving writes the slots it admits.
        _nf = torch.zeros(num_nodes, node_feat_dim)
        # Column 14 (trust) is a neutral 1.0 everywhere, headroom slots included; alarm
        # history lives in recent_alert.
        if node_feat_dim > 14:
            _nf[:, 14] = 1.0
        self.register_buffer("node_feat", _nf)

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

        # Structural head: scaled cosine compatibility of the projected embeddings, i.e.
        # whether the pair belongs together given history (valid-but-non-habitual access).
        self.struct_proj = StructuralProjector(
            in_channels=memory_dim,
            hidden_layers=link_pred_hidden_layers,
            dropout=0.1,
        )

        self.struct_scale = nn.Parameter(torch.tensor(5.0))

        # Runtime state (see the class docstring).
        self.last_contact = {}
        self.pair_count = {}
        self.src_count = {}
        self.recent_alert = {}

        self.precursor_half_life = _Cfg.precursor_half_life
        self.precursor_max_shift = _Cfg.precursor_max_shift
        # Arm threshold of serve_tgn.event_alarm, fitted at calibration (stats file);
        # None = arm on the decision alone.
        self.threshold_arm = None
        # Δt cap / never-seen sentinel (config.delta_t_cap). Not a buffer, so a checkpoint
        # cannot override it.
        self.delta_t_cap = _Cfg.delta_t_cap

        # Ablation switches (runtime only; the full model keeps them on).
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

        The GNN's relative time reads ``last_update`` from the memory buffer, not from
        ``TGNMemory.forward``: in train mode PyG returns 0 for nodes without pending
        messages, which would train on ``rel_t ≈ -1e7`` and serve on small values through a
        periodic encoding. The memory state ``z`` keeps PyG's train-mode update (standard TGN).
        """
        z, _last_update_train = self.memory(n_id)
        last_update = self.memory.last_update[n_id]
        nf = self.node_feat[n_id]
        h_idx = self.node_hash[n_id]
        he = self.hash_emb(h_idx)
        if not self.use_hash_identity:
            he = torch.zeros_like(he)  # ablation: drop the hashed-identity signal
        x = torch.cat([z, nf, he], dim=-1)  # identity-aware node features
        z = self.gnn(x, last_update, edge_index, hist_t, hist_msg)
        return z

    def pair_delta_t(self, src_ids, dst_ids, t_vals, device):
        """Recency of each ``src→dst`` pair, capped at ``delta_t_cap``; never-seen pairs get the cap.

        A constant sentinel (not the absolute clock) keeps train, val, test and long-running
        serving on the same encoding and gives InfoNCE no one-scalar shortcut (random
        negatives are never-seen by construction). Shared by training, replay and serving.
        """
        cap = float(self.delta_t_cap)
        out = []
        for s, d, t in zip(src_ids, dst_ids, t_vals):
            last = self.last_contact.get((int(s), int(d)))
            out.append(cap if last is None else min(max(float(t) - float(last), 0.0), cap))
        return torch.tensor(out, dtype=torch.float, device=device)

    def src_delta_t(self, src_ids, t_vals, device=None):
        """Recency of each src's memory, capped; ``last_update == 0`` (never committed) gets the cap."""
        cap = float(self.delta_t_cap)
        if not torch.is_tensor(src_ids):
            src_ids = torch.as_tensor(list(src_ids), dtype=torch.long,
                                      device=self.memory.last_update.device)
        if not torch.is_tensor(t_vals):
            t_vals = torch.as_tensor(list(t_vals), dtype=torch.float, device=src_ids.device)
        last = self.memory.last_update[src_ids].to(torch.float)
        d = (t_vals.to(torch.float).to(last.device) - last).clamp(min=0.0, max=cap)
        d = torch.where(last <= 0, torch.full_like(d, cap), d)
        return d if device is None else d.to(device)

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
        """6-dim interaction-history features of the directed ``src→dst`` pairs (global ids).

        First triplet: pair count, src activity and the share of the src's traffic going to
        this dst. A never-seen pair from an active src is the novelty cue for lateral
        movement; benign exploration shares it, so it helps only with memory and structure.
        Second triplet: the same for ``aux_src→dst``; on the access edge the aux src is the
        device (per-device habituality without a device→resource edge). ``aux_src_ids=None``
        zero-pads it. Counts are benign-gated.
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
        # Per-node work runs once per distinct endpoint; ``src`` / ``dst`` index those rows.
        uniq, inv = torch.unique(torch.cat([src_local, dst_local]), return_inverse=True)
        src, dst = inv[: src_local.numel()], inv[src_local.numel():]
        z_u = z[uniq]
        he = self.hash_emb(h_idx[uniq])
        if not self.use_hash_identity:
            he = torch.zeros_like(he)  # ablation: drop the hashed-identity signal
        feat_with_hash = torch.cat([nf[uniq], he], dim=-1)
        recency_enc = self.memory.time_enc(delta_t)
        src_recency_enc = self.memory.time_enc(delta_t_src)
        if not self.use_hist_feats:
            hist_feats = torch.zeros_like(hist_feats)  # ablation: drop history features
        feat = self.link_pred(
            z_u, feat_with_hash, src, dst, cur_msg, recency_enc, src_recency_enc, hist_feats,
        ).squeeze(-1)
        if not self.use_struct_head:
            return feat  # ablation: feature head only (no structural compatibility head)
        if self.training:
            # struct_proj has Dropout: keep one independent draw per scored row.
            hs = F.normalize(self.struct_proj(z_u[src]), dim=-1)
            hd = F.normalize(self.struct_proj(z_u[dst]), dim=-1)
        else:
            # Dropout is the identity in eval, so projecting each node once is exact.
            proj = F.normalize(self.struct_proj(z_u), dim=-1)
            hs, hd = proj[src], proj[dst]
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
