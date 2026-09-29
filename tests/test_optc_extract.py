"""Unit tests for scripts/optc_extract.py (LMDEval semantics on synthetic eCAR lines)."""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "optc_extract.py"


@pytest.fixture(scope="module")
def ox():
    spec = importlib.util.spec_from_file_location("optc_extract", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _flow(eid, ts, src, dst, sport=50000, dport=445, direction="outbound", host="SysClient0201.systemia.com",
          action="START", obj="FLOW", spaced=False, principal="", image=""):
    evt = {"action": action, "actorID": "a-" + eid, "hostname": host, "id": eid, "object": obj,
           "principal": principal,
           "properties": {"dest_ip": dst, "dest_port": str(dport), "direction": direction,
                          "image_path": image, "l4protocol": "6", "src_ip": src, "src_port": str(sport)},
           "timestamp": ts}
    return json.dumps(evt, separators=(", ", ": ") if spaced else (",", ":"))


def test_include_ip(ox):
    assert ox.include_ip("10.20.1.5") and ox.include_ip("142.20.56.10") and ox.include_ip("fe80::1")
    assert not ox.include_ip("10.20.1.255") and not ox.include_ip("142.20.59.255")
    assert not ox.include_ip("8.8.8.8") and not ox.include_ip("ff02::1:3")


def test_flow_start_compact_and_spaced(ox):
    ts = "2019-09-23T10:00:00.000-04:00"
    assert ox.is_flow_start(_flow("a", ts, "10.0.0.1", "10.0.0.2"))
    assert ox.is_flow_start(_flow("a", ts, "10.0.0.1", "10.0.0.2", spaced=True))
    assert not ox.is_flow_start(_flow("a", ts, "10.0.0.1", "10.0.0.2", action="END"))
    assert not ox.is_flow_start(_flow("a", ts, "10.0.0.1", "10.0.0.2", obj="PROCESS"))
    # LMDEval requires "action" to be the first key; so do we
    moved = json.dumps({"timestamp": ts, "action": "START", "object": "FLOW"})
    assert not ox.is_flow_start(moved)


def test_extract_keeps_internal_starts_only(ox, tmp_path):
    ts = "2019-09-23T10:00:00.500-04:00"
    lines = [
        _flow("in", ts, "142.20.56.1", "142.20.56.2", principal="DOM\\u1", image="C:\\x.exe"),
        _flow("ext", ts, "142.20.56.1", "8.8.8.8"),
        _flow("bcast", ts, "142.20.56.1", "142.20.59.255"),
        _flow("sp", ts, "fe80::1", "10.1.1.1", spaced=True, direction="inbound"),
        '{"action":"START","object":"FLOW", broken',
    ]
    raw = io.BytesIO(gzip.compress(("\n".join(lines) + "\n").encode()))
    out = tmp_path / "x.csv.gz"
    c = ox.extract_gzip_stream(raw, str(out))
    assert c == {"lines": 5, "flow_start": 5, "kept": 2, "bad": 1}
    rows = list(ox.read_raw(str(out)))
    assert [r[0] for r in rows] == ["in", "sp"]
    r = rows[0]
    assert r[1] == ox.make_timestamp(ts) and r[5:9] == [50000, 445, 6, True]
    assert r[9] == "DOM\\u1" and r[10] == "C:\\x.exe" and rows[1][8] is False
    assert not (tmp_path / "x.csv.gz.part").exists()


def _row(eid, ts, src, dst, outbound, sport=1, host="h"):
    return [eid, ts, host, src, dst, sport, 445, 6, outbound, "", "", ""]


def test_deduplicate_rules(ox):
    flows = {
        "both": [_row("o", 1, "a", "b", True), _row("i", 1, "a", "b", False)],
        "one_side": [_row("i1", 1, "a", "b", False), _row("i2", 2, "a", "b", False)],
        "unbalanced": [_row("o1", 1, "a", "b", True), _row("o2", 2, "a", "b", True), _row("i", 1, "a", "b", False)],
    }
    ox.deduplicate(flows)
    assert [e[0] for e in flows["both"]] == ["o"]
    assert [e[0] for e in flows["one_side"]] == ["i1", "i2"]
    assert [e[0] for e in flows["unbalanced"]] == ["i"]


def test_build_labels_hosts_and_time(ox, tmp_path):
    pytest.importorskip("pandas")
    t = "2019-09-23T10:00:0{}.700-04:00"
    lines = [
        _flow("lm", t.format(2), "142.20.56.1", "142.20.56.2", host="A"),       # outbound: .1 -> A
        _flow("oth", t.format(1), "142.20.56.3", "142.20.56.1", sport=2, direction="inbound", host="A"),
        _flow("ben", t.format(1), "142.20.56.3", "142.20.56.2", sport=3, host="C"),
    ]
    raw = io.BytesIO(gzip.compress("\n".join(lines).encode()))
    ox.extract_gzip_stream(raw, str(tmp_path / "r" / "f.csv.gz"))
    known = {"142.20.56.1": "OLD", "142.20.56.2": "B"}
    df, learned = ox.build(ox._raw_files(str(tmp_path / "r")), known, {"lm", "oth"}, {"lm"})
    assert list(df["id"]) == ["ben", "oth", "lm"]  # time order, ties broken by id
    assert list(df["timestamp"]) == [0, 0, 1]
    assert list(df["label"]) == [0, 1, 1] and list(df["label_lm"]) == [0, 0, 1]
    # learned mapping overrides the known file, as in LMDEval
    assert learned == {"142.20.56.1": "A", "142.20.56.3": "C"}
    assert list(df["src"]) == ["C", "C", "A"] and list(df["dst"]) == ["B", "A", "B"]


def test_select_members(ox):
    idx = [("2019-09-23/AIA-201-225/AIA-201-225.ecar-2019-09-23-sysclient0201.json.gz", 512, 10),
           ("2019-09-23/AIA-201-225/AIA-201-225.ecar-2019-09-23-sysclient0202.json.gz", 1024, 10),
           ("2019-09-23/AIA-401-425/AIA-401-425.ecar-2019-09-23-sysclient0402.json.gz", 2048, 10),
           ("2019-09-23/AIA-401-425/README.txt", 4096, 10)]
    assert len(ox.select_members(idx)) == 3
    assert [m[1] for m in ox.select_members(idx, groups=["AIA-201-225"])] == [512, 1024]
    assert [m[1] for m in ox.select_members(idx, hosts=["SysClient0402", "sysclient0201"])] == [512, 2048]
    assert [m[1] for m in ox.select_members(idx, groups=["AIA-201-225"], hosts=["sysclient0402"])] == []
    assert ox.select_members(idx, hosts=["sysclient020"]) == []  # no prefix matches
