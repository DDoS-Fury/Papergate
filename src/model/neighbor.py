"""Bounded temporal neighbor loader for the streaming TGN.

PyG's :class:`LastNeighborLoader` keeps, for each node, only its last ``size``
interactions (neighbor id + a monotonically growing edge id). To run the temporal
attention layer one then needs each historical edge's *timestamp* and *raw message*.
The canonical PyG example looks these up in the full ``data.t`` / ``data.msg`` arrays
— fine for an offline, bounded dataset, but in a long-running streaming service that
per-edge store grows without bound (one entry per event ever seen).

:class:`MessageNeighborLoader` removes that unbounded store: it carries the timestamp
and raw message *alongside* the neighbor/edge-id ring buffers, in the same fixed-size
``[num_nodes, size]`` layout. Total state is O(num_nodes * size * msg_dim) and never
grows — safe for indefinite serving. The ring-buffer bookkeeping mirrors the parent's
``insert`` exactly so eviction semantics (keep the ``size`` most recent edges) are
preserved.
"""

from __future__ import annotations

import torch
from torch_geometric.nn.models.tgn import LastNeighborLoader


class MessageNeighborLoader(LastNeighborLoader):
    def __init__(self, num_nodes: int, size: int, msg_dim: int, device=None):
        # Allocate the parallel attribute buffers before super().__init__ (which
        # calls reset_state) so reset_state can clear them too.
        self.msg_dim = msg_dim
        self.last_t = torch.zeros((num_nodes, size), dtype=torch.long, device=device)
        self.last_msg = torch.zeros((num_nodes, size, msg_dim), dtype=torch.float,
                                    device=device)
        super().__init__(num_nodes, size, device=device)

    def reset_state(self):
        super().reset_state()
        # ``hasattr`` guard: super().__init__ calls this before our attrs in a plain
        # LastNeighborLoader, but here they are set first, so this always runs.
        if hasattr(self, "last_t"):
            self.last_t.zero_()
            self.last_msg.zero_()

    def insert(self, src, dst, t, msg):
        # Mirror of LastNeighborLoader.insert, threading (t, msg) through the same
        # sort / dense-placement / top-k bookkeeping so they stay aligned with e_id.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0),
                            device=src.device).repeat(2)
        e_t = torch.cat([t, t], dim=0)
        e_msg = torch.cat([msg, msg], dim=0)
        self.cur_e_id += src.numel()

        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]
        e_t, e_msg = e_t[perm], e_msg[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        dense_t = e_t.new_zeros(n_id.numel() * self.size)
        dense_t[dense_id] = e_t
        dense_t = dense_t.view(-1, self.size)

        dense_msg = e_msg.new_zeros(n_id.numel() * self.size, self.msg_dim)
        dense_msg[dense_id] = e_msg
        dense_msg = dense_msg.view(n_id.numel(), self.size, self.msg_dim)

        e_id_cat = torch.cat([self.e_id[n_id, :self.size], dense_e_id], dim=-1)
        neighbors_cat = torch.cat([self.neighbors[n_id, :self.size], dense_neighbors],
                                  dim=-1)
        t_cat = torch.cat([self.last_t[n_id, :self.size], dense_t], dim=-1)
        msg_cat = torch.cat([self.last_msg[n_id, :self.size], dense_msg], dim=1)

        e_id_top, perm = e_id_cat.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id_top
        self.neighbors[n_id] = torch.gather(neighbors_cat, 1, perm)
        self.last_t[n_id] = torch.gather(t_cat, 1, perm)
        self.last_msg[n_id] = torch.gather(
            msg_cat, 1, perm.unsqueeze(-1).expand(-1, -1, self.msg_dim))

    def __call__(self, n_id):
        """Expand ``n_id`` with its stored neighbors.

        Returns ``(out_n_id, edge_index, hist_t, hist_msg)`` where ``edge_index`` is
        relabelled to positions in ``out_n_id`` (source = neighbor, target = queried
        node) and ``hist_t`` / ``hist_msg`` are the matching historical edge attrs.
        ``self._assoc`` maps global ids to local positions in ``out_n_id``.
        """
        neighbors = self.neighbors[n_id]
        e_id = self.e_id[n_id]
        hist_t = self.last_t[n_id]
        hist_msg = self.last_msg[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)

        mask = e_id >= 0
        neighbors, nodes = neighbors[mask], nodes[mask]
        hist_t, hist_msg = hist_t[mask], hist_msg[mask]

        out_n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[out_n_id] = torch.arange(out_n_id.size(0), device=out_n_id.device)
        edge_index = torch.stack([self._assoc[neighbors], self._assoc[nodes]])
        return out_n_id, edge_index, hist_t, hist_msg

    def reset_node(self, idx: int) -> None:
        """Cold-start a reused slot after registry eviction.

        Clears the node's own ring buffer and scrubs any *inbound* references to the
        slot from other nodes' buffers, so a recycled index can't leak the previous
        entity's history. Called on the rare eviction path, where O(num_nodes*size)
        is acceptable.
        """
        self.e_id[idx].fill_(-1)
        self.neighbors[idx].fill_(0)
        self.last_t[idx].zero_()
        self.last_msg[idx].zero_()
        self.e_id[self.neighbors == idx] = -1

    # --- persistence (plain tensors, saved alongside the model checkpoint) ----
    def state(self) -> dict:
        return {
            "neighbors": self.neighbors,
            "e_id": self.e_id,
            "last_t": self.last_t,
            "last_msg": self.last_msg,
            "cur_e_id": int(self.cur_e_id),
        }

    def load_state(self, s: dict) -> None:
        self.neighbors = s["neighbors"]
        self.e_id = s["e_id"]
        self.last_t = s["last_t"]
        self.last_msg = s["last_msg"]
        self.cur_e_id = int(s["cur_e_id"])
