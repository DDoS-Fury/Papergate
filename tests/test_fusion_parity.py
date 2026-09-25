"""Exact-fusion parity: each fused path must equal its unfused reference (no checkpoint needed).

* ``MessageNeighborLoader.__call__`` (frontier without already-expanded nodes) against the
  re-expanding loader it replaced, kept verbatim below: same edge set, same embeddings.
* ``LinkPredictor.forward`` (``lin1`` applied by column blocks, endpoint blocks once per node)
  against ``lin1`` on the concatenated input: same logits and gradients, in train mode.
* ``score_event`` (one expansion + embed for the whole chain, via ``chain_logits``) against
  the max of the per-edge ``infer_logit`` reference, on a stream committed through the
  serving API: full chain, missing device, missing source, guest-device fallback.

The offline replay counterpart is ``tests/verify_replay_batching.py``.

    pytest tests/test_fusion_parity.py
"""

import random

import pytest
import torch

from graphagate.model.registry import NodeRegistry
from graphagate.model.tgn import LinkPredictor
from graphagate.netclass import to_guest_device
from graphagate.serve_tgn import (
    anomaly_score,
    build_model,
    commit_event,
    infer_logit,
    precursor_shift,
    score_event,
)

DEVICE = torch.device("cpu")
MSG_DIM = 7
HP = {
    "schema_version": 4,
    "capacity": 128,
    "node_feat_dim": 16,
    "msg_dim": MSG_DIM,
    "memory_dim": 32,
    "time_dim": 8,
    "num_hops": 3,
    "hash_buckets": 100,
    "hash_dim": 8,
    "hist_feat_dim": 6,
    "neighbor_size": 5,
}


def _reexpanding_call(self, n_id):
    """The loader before 2026-09-25, verbatim: re-expands already visited nodes."""
    nodes_list, neighbors_list, hist_t_list, hist_msg_list = [], [], [], []
    current_n_id = n_id
    for _ in range(self.k_hops):
        neighbors = self.neighbors[current_n_id]
        e_id = self.e_id[current_n_id]
        hist_t = self.last_t[current_n_id]
        hist_msg = self.last_msg[current_n_id]
        nodes = current_n_id.view(-1, 1).repeat(1, self.size)
        mask = e_id >= 0
        neighbors, nodes = neighbors[mask], nodes[mask]
        hist_t, hist_msg = hist_t[mask], hist_msg[mask]
        nodes_list.append(nodes)
        neighbors_list.append(neighbors)
        hist_t_list.append(hist_t)
        hist_msg_list.append(hist_msg)
        current_n_id = neighbors.unique()
        if current_n_id.numel() == 0:
            break
    all_nodes = torch.cat(nodes_list)
    all_neighbors = torch.cat(neighbors_list)
    all_hist_t = torch.cat(hist_t_list)
    all_hist_msg = torch.cat(hist_msg_list)
    out_n_id = torch.cat([n_id, all_neighbors]).unique()
    self._assoc[out_n_id] = torch.arange(out_n_id.size(0), device=out_n_id.device)
    edge_index = torch.stack([self._assoc[all_neighbors], self._assoc[all_nodes]])
    return out_n_id, edge_index, all_hist_t, all_hist_msg


def _stream(n, seed=0):
    rng = random.Random(seed)
    for i in range(n):
        u = rng.randrange(10)
        yield dict(
            key_user=f"u{u}",
            key_device=f"tpm:{u % 6}" if rng.random() < 0.9 else f"tpm:{rng.randrange(6)}",
            key_dst=f"/r/{rng.randrange(8)}",
            timestamp=100 + 37 * i,
            features=[rng.random() for _ in range(MSG_DIM)],
            key_source=f"src:10.0.0.{rng.randrange(4)}",
            key_config=f"conf:{u % 4}" if rng.random() < 0.9 else f"conf:{rng.randrange(4)}",
        )


def _warmed(n=200):
    torch.manual_seed(0)
    model = build_model(dict(HP), DEVICE)
    reg = NodeRegistry(capacity=HP["capacity"])
    for ev in _stream(n):
        commit_event(model, reg, device=DEVICE, **ev)
    return model, reg


def _edges(out):
    n_id, edge_index, hist_t, _ = out
    return list(zip(n_id[edge_index[0]].tolist(), n_id[edge_index[1]].tolist(), hist_t.tolist()))


def test_loader_dedup_same_edge_set_and_embeddings():
    model, reg = _warmed()
    model.eval()
    loader = model.neighbor_loader
    for keys in (["u0", "/r/1"], ["u1", "tpm:1", "conf:1", "src:10.0.0.2", "/r/3"]):
        q = torch.tensor(sorted(reg.get(k) for k in keys))
        new, old = loader(q), _reexpanding_call(loader, q)
        assert torch.equal(new[0], old[0])
        assert set(_edges(new)) == set(_edges(old))
        assert new[1].size(1) < old[1].size(1)  # the duplicates are gone
        with torch.no_grad():
            z_new, z_old = model.embed(*new), model.embed(*old)
        torch.testing.assert_close(z_new, z_old, atol=1e-5, rtol=1e-5)


def test_link_predictor_blocks_equal_concat_with_grads():
    torch.manual_seed(1)
    c, m, f, t, h = 32, MSG_DIM, 24, 8, 6
    lp = LinkPredictor(c, m, node_feat_dim=16, hash_dim=8, time_dim=t, hist_feat_dim=h).train()
    nodes, rows = 9, 40
    z = torch.randn(nodes, c, requires_grad=True)
    feat = torch.randn(nodes, f)
    src, dst = torch.randint(nodes, (rows,)), torch.randint(nodes, (rows,))
    msg, rec, srec, hist = (torch.randn(rows, d) for d in (m, t, t, h))

    out = lp(z, feat, src, dst, msg, rec, srec, hist)
    g_new = torch.autograd.grad(out.square().sum(), [z, lp.lin1.weight])

    x = torch.cat([z[src], z[dst], msg, feat[src], feat[dst], rec, srec, hist], dim=-1)
    ref = lp.lin1(x).relu()
    ref = lp.lin_mid(ref).relu()
    ref = lp.lin2(ref)
    g_ref = torch.autograd.grad(ref.square().sum(), [z, lp.lin1.weight])

    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    for a, b in zip(g_new, g_ref):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def _variant(ev, variant, i):
    """Exercise score_event's branches: missing device, missing source, guest devices."""
    ev = dict(ev)
    if variant == "no_device" and i % 2:
        ev["key_device"] = None
    elif variant == "no_source":
        ev["key_source"] = None
    elif variant == "guest" and i % 2:
        ev["key_device"] = f"ck:{i % 3}"  # non-TPM: collapses onto dev:guest
    return ev


@pytest.mark.parametrize("variant", ["full", "no_device", "no_source", "guest"])
def test_score_event_equals_per_edge_reference(variant):
    model, reg = _warmed()
    zeros = [0.0] * MSG_DIM
    guest = variant == "guest"
    for i, ev in enumerate(_stream(40, seed=1)):
        ev = _variant(ev, variant, i)
        fused, _, _ = score_event(model, reg, 2.0, device=DEVICE, update=False,
                                  guest_device_fallback=guest, **ev)
        key_dev = to_guest_device(ev["key_device"]) if guest else ev["key_device"]
        d = None if key_dev is None else reg.get(key_dev)
        s = None if ev["key_source"] is None else reg.get(ev["key_source"])
        u, r, c = (reg.get(ev[k]) for k in ("key_user", "key_dst", "key_config"))
        t = ev["timestamp"]
        per_edge = [
            infer_logit(model, u, r, t, ev["features"], DEVICE, aux_src_idx=d),
            infer_logit(model, c, u, t, zeros, DEVICE),
        ]
        if d is not None:
            per_edge += [infer_logit(model, c, d, t, zeros, DEVICE),
                         infer_logit(model, d, u, t, zeros, DEVICE)]
        if s is not None:
            per_edge.append(infer_logit(model, s, c, t, zeros, DEVICE))
        boost = d if d is not None else u
        ref = float(anomaly_score(max(per_edge) + precursor_shift(model, boost, t)))
        assert abs(fused - ref) <= 1e-6, (i, fused, ref)
        commit_event(model, reg, device=DEVICE, guest_device_fallback=guest, **ev)
