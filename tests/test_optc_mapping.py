"""Contract tests for the OpTC flow list -> StreamData mapping (tests/datasets/optc.py)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

pd = pytest.importorskip("pandas")
pytest.importorskip("torch")


def _flows(tmp_path):
    rows = []
    # hosts A..C, times 0..11 h; LM at 10 h, Other at 11 h (both in test)
    for i in range(12):
        src, dst = "ABC"[i % 3], "ABC"[(i + 1) % 3]
        rows.append({"timestamp": i * 3600, "src": src, "dst": dst, "src_port": 50000, "dst_port": 445 if i % 2 else 53,
                     "proto": 6, "label": int(i >= 10), "label_lm": int(i == 10), "timestamp_abs": 1.5e9 + i * 3600,
                     "id": f"e{i}", "hostname": src, "src_ip": f"10.0.0.{i % 3}", "dst_ip": "10.0.0.9",
                     "outbound": 1, "principal": "DOM\\u" if i % 4 == 0 else "", "image_path": "" if i % 2 else "C:\\X.EXE",
                     "actor_id": ""})
    path = tmp_path / "f.csv.gz"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("nodes", ["lmdeval", "enriched"])
def test_mapping_invariants(tmp_path, nodes):
    from datasets.optc import T_LATERAL, T_OTHER, load_optc_stream

    data, tf, vf, df = load_optc_stream(_flows(tmp_path), val_start=1.5e9 + 6 * 3600, test_start=1.5e9 + 8 * 3600,
                                        nodes=nodes)
    n = len(df)
    assert int(n * tf) == 6 and int(n * tf) + int(n * vf) == 8  # exact split at the time cuts
    assert data.types.tolist()[10:] == [T_LATERAL, T_OTHER] and data.y.tolist()[10:] == [1, 1]
    assert (data.t[1:] >= data.t[:-1]).all() and data.t[0] == 0
    assert data.msg[:, 1:4].abs().sum() == 0 and data.msg[:, 5:].abs().sum() == 0 and (data.msg[:, 0] == 1).all()
    assert (data.node_features[:, 14] == 1).all() and data.node_features[:, :14].abs().sum() == 0
    keys = data.keys
    assert all(keys[d].startswith("res:") for d in data.dst.tolist())
    assert data.neg_lo <= data.dst.min() and data.dst.max() < data.neg_lo + data.neg_num
    if nodes == "lmdeval":
        assert data.device_nodes is None and data.config_nodes is None
        assert keys[data.user[0]] == "res:A"  # same node as actor and as resource
    else:
        for arr, lo, num, prefix in ((data.user, data.usr_lo, data.usr_num, "usr:"),
                                     (data.device_nodes, data.dev_lo, data.dev_num, "dev:"),
                                     (data.config_nodes, data.cfg_lo, data.cfg_num, "cfg:")):
            assert lo <= arr.min() and arr.max() < lo + num
            assert all(keys[i].startswith(prefix) for i in range(lo, lo + num))
        # empty fields map to a per-host sentinel, never to one shared node
        assert keys[data.user[1]] == "usr:none:B" and keys[data.config_nodes[1]] == "cfg:none:B"
        assert keys[data.config_nodes[0]] == "cfg:c:\\x.exe"


def test_rejects_cuts_outside_data(tmp_path):
    from datasets.optc import load_optc_stream

    with pytest.raises(ValueError):
        load_optc_stream(_flows(tmp_path), val_start=1.0, test_start=2.0)
