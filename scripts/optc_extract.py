"""Streaming extraction of internal OpTC flows with the LMDEval semantics.

LMDEval (https://github.com/cl-anssi/LMDEval, extract_optc.py) reads the whole eCAR tree from
disk (~1 TB). This script produces the same flow list without storing the raw logs:

  1. ``extract``: stream each raw ``.json.gz`` (local file or a member of the corrected per-day
     release, doi:10.57745/UXCWOC, read with HTTP range requests), keep only FLOW START events
     whose two endpoints pass ``include_ip``, and write them to one small ``.csv.gz`` per raw file.
  2. ``build``: merge the per-file outputs, learn the address -> host map, deduplicate flows seen
     on both endpoints and attach the labels, exactly as ``extract_optc_dataset`` does.

Differences from LMDEval, all additive:
  * the FLOW START test also accepts the ``", "`` separators of the corrected release (the
    original prefix test ``line[1:17] == '"action":"START"'`` matches 0 lines there);
  * the output keeps the absolute timestamp (needed to cut the test period at a calendar day),
    both label sets (``label`` = all red-team ids, ``label_lm`` = what ``--lm-only`` gives),
    and ``id``, ``hostname``, ``principal``, ``image_path``, ``actor_id`` of the kept event;
  * ties in time are ordered by the absolute timestamp, then by id (LMDEval sorts by the
    truncated integer second with an unstable sort).

``include_ip``, ``deduplicate`` and the address learning are adapted from LMDEval,
Copyright (c) 2026, cl-anssi, BSD 2-Clause License (text in data/optc/meta/LMDEval/LICENSE).

Usage:
    python scripts/optc_extract.py index 2019-09-23.tar                       # member list
    python scripts/optc_extract.py extract --day 2019-09-23 --group AIA-201-225 --out data/optc/flows
    python scripts/optc_extract.py extract-local data/optc/ecar --out data/optc/flows_local
    python scripts/optc_extract.py build data/optc/flows --meta data/optc/meta/LMDEval \
        --out data/optc/optc_flows.csv.gz
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import gzip
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

ORIGIN = dt.datetime.fromisoformat("1970-01-01T00:00:00.000-04:00")  # as in LMDEval
API = "https://entrepot.recherche.data.gouv.fr/api"
DOI = "doi:10.57745/UXCWOC"
RETRIES = 4

RAW_COLS = ["id", "ts", "hostname", "src_ip", "dst_ip", "src_port", "dst_port", "proto",
            "outbound", "principal", "image_path", "actor_id"]

_START_PREFIXES = ('"action":"START"', '"action": "START"')


# ---------------------------------------------------------------------------------------------
# Stage 1: raw eCAR lines -> internal FLOW START rows
# ---------------------------------------------------------------------------------------------


def include_ip(addr: str) -> bool:
    """LMDEval's filter: internal unicast addresses only."""
    if addr.startswith("10.") and not addr.endswith(".255"):
        return True
    if addr.startswith("142.") and addr != "142.20.59.255":
        return True
    return addr.startswith("fe80:")


def make_timestamp(ts: str) -> float:
    return (dt.datetime.fromisoformat(ts) - ORIGIN).total_seconds()


def is_flow_start(line: str) -> bool:
    """LMDEval's fast test, extended to the spaced JSON of the corrected release."""
    return (line[1:17] == _START_PREFIXES[0] or line[1:18] == _START_PREFIXES[1]) and (
        '"object":"FLOW"' in line or '"object": "FLOW"' in line)


def parse_flow(line: str) -> list | None:
    """One FLOW START line -> a RAW_COLS row, or None if an endpoint is not internal."""
    evt = json.loads(line)
    p = evt["properties"]
    src_ip, dst_ip = p["src_ip"], p["dest_ip"]
    if not include_ip(src_ip) or not include_ip(dst_ip):
        return None
    return [evt["id"], repr(make_timestamp(evt["timestamp"])), evt["hostname"], src_ip, dst_ip,
            int(p["src_port"]), int(p["dest_port"]), int(p["l4protocol"]),
            int(p["direction"] == "outbound"), evt.get("principal") or "", p.get("image_path") or "",
            evt.get("actorID") or ""]


def extract_lines(lines, out_path: str) -> dict:
    """Filter an iterable of raw lines into ``out_path`` (.csv.gz), atomically. Returns counts."""
    part = out_path + ".part"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n_lines = n_start = n_kept = n_bad = 0
    with gzip.open(part, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(RAW_COLS)
        for line in lines:
            n_lines += 1
            if not is_flow_start(line):
                continue
            n_start += 1
            try:
                row = parse_flow(line)
            except (ValueError, KeyError) as e:  # truncated or malformed record
                n_bad += 1
                if n_bad <= 5:
                    print(f"    [!] riga non valida ({type(e).__name__}): {line[:120]!r}", file=sys.stderr)
                continue
            if row is not None:
                w.writerow(row)
                n_kept += 1
    os.replace(part, out_path)
    return {"lines": n_lines, "flow_start": n_start, "kept": n_kept, "bad": n_bad}


def extract_gzip_stream(raw, out_path: str) -> dict:
    """``raw`` is a binary file-like object holding a gzip stream."""
    with gzip.GzipFile(fileobj=raw) as gz:
        return extract_lines(io.TextIOWrapper(gz, encoding="utf-8", errors="replace"), out_path)


# ---------------------------------------------------------------------------------------------
# Corrected release (data.gouv): tar members read with HTTP range requests
# ---------------------------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _urlopen(req, tries: int = RETRIES):
    for attempt in range(tries):
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
        except OSError:
            if attempt == tries - 1:
                raise
        time.sleep(2 ** attempt * 2)


def release_files() -> dict[str, int]:
    """File name -> data.gouv datafile id."""
    with _urlopen(f"{API}/datasets/:persistentId/?persistentId={DOI}") as r:
        files = json.load(r)["data"]["latestVersion"]["files"]
    return {f["dataFile"]["filename"]: f["dataFile"]["id"] for f in files}


def signed_url(file_id: int) -> str:
    """The access endpoint redirects to a short-lived signed URL that honours Range."""
    try:
        urllib.request.build_opener(_NoRedirect).open(f"{API}/access/datafile/{file_id}", timeout=60)
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307) and e.headers.get("Location"):
            return e.headers["Location"]
        raise
    raise RuntimeError(f"datafile {file_id}: nessun redirect verso l'URL firmato")


def _range(url: str, start: int, end: int):
    return _urlopen(urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"}))


def tar_index(file_id: int) -> list[tuple[str, int, int]]:
    """(member path, data offset, size) of every regular file, reading tar headers only."""
    url = signed_url(file_id)
    out, off = [], 0
    while True:
        with _range(url, off, off + 511) as r:
            h = r.read()
        if len(h) < 512 or h == b"\0" * 512:
            return out
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        typ = chr(h[156]) if h[156] else "0"
        prefix = h[345:500].rstrip(b"\0").decode()
        name = (prefix + "/" if prefix else "") + h[0:100].rstrip(b"\0").decode()
        if typ == "L":  # GNU long name: the next header describes the file
            with _range(url, off + 512, off + 512 + size - 1) as r:
                name = r.read().rstrip(b"\0").decode()
            off += 512 + (size + 511) // 512 * 512
            with _range(url, off, off + 511) as r:
                h = r.read()
            size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
            typ = chr(h[156]) if h[156] else "0"
        if typ in ("0", "\0"):
            out.append((name, off + 512, size))
        off += 512 + (size + 511) // 512 * 512


def cached_index(file_id: int, tar: str, out_dir: str) -> list[tuple[str, int, int]]:
    """``tar_index`` saved as ``<out_dir>/<tar>.index.tsv``: the scan costs ~1000 range requests."""
    path = os.path.join(out_dir, tar + ".index.tsv")
    if os.path.exists(path):
        with open(path) as f:
            return [(n, int(o), int(s)) for n, o, s in (line.rstrip("\n").split("\t") for line in f)]
    idx = tar_index(file_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(path + ".part", "w") as f:
        f.writelines(f"{n}\t{o}\t{s}\n" for n, o, s in idx)
    os.replace(path + ".part", path)
    return idx


def _member_out(out_dir: str, member: str) -> str:
    return os.path.join(out_dir, member.removesuffix(".json.gz") + ".csv.gz")


def extract_member(file_id: int, member: str, offset: int, size: int, out_dir: str) -> tuple[str, dict]:
    """Stream one tar member, filter it, write its .csv.gz. Restarts the member on errors."""
    out_path = _member_out(out_dir, member)
    for attempt in range(1, RETRIES + 1):
        try:
            with _range(signed_url(file_id), offset, offset + size - 1) as r:
                return member, extract_gzip_stream(r, out_path)
        except (OSError, EOFError) as e:
            if attempt == RETRIES:
                raise
            print(f"    [!] {member}: {type(e).__name__}: {e}; ricomincio ({attempt + 1}/{RETRIES})", file=sys.stderr)
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------------------------
# Stage 2: per-file rows -> deduplicated, labelled flow list
# ---------------------------------------------------------------------------------------------


def deduplicate(flows: dict) -> dict:
    """LMDEval's rule. ``flows``: 5-tuple -> list of rows whose index 8 is ``outbound``."""
    for key, events in flows.items():
        balance = sum(1 if e[8] else -1 for e in events)
        if balance == 0:
            flows[key] = [e for e in events if e[8]]
        elif abs(balance) == len(events):
            continue
        else:
            flows[key] = [e for e in events if not e[8]]
    return flows


def read_raw(path: str):
    with gzip.open(path, "rt", newline="") as fh:
        r = csv.reader(fh)
        header = next(r)
        if header != RAW_COLS:
            raise ValueError(f"{path}: intestazione inattesa {header}")
        for row in r:
            yield [row[0], float(row[1]), row[2], row[3], row[4], int(row[5]), int(row[6]), int(row[7]),
                   row[8] == "1", row[9], row[10], row[11]]


def build(raw_paths: list[str], addr_to_host: dict, all_ids: set, lm_ids: set):
    """Returns the flow DataFrame (pandas) and the learned address -> host map."""
    import pandas as pd

    flows: dict = defaultdict(list)
    learned: dict = {}
    for path in sorted(raw_paths):
        for row in read_raw(path):
            flows[(row[3], row[4], row[5], row[6], row[7])].append(row)
            learned[row[3] if row[8] else row[4]] = row[2]
    addr_to_host = {**addr_to_host, **learned}
    deduplicate(flows)
    rows = [e for events in flows.values() for e in events]
    df = pd.DataFrame(rows, columns=["id", "timestamp_abs", "hostname", "src_ip", "dst_ip", "src_port",
                                     "dst_port", "proto", "outbound", "principal", "image_path", "actor_id"])
    df["timestamp"] = (df["timestamp_abs"] - df["timestamp_abs"].min()).astype(int)
    df["src"] = df["src_ip"].map(lambda a: addr_to_host.get(a, a))
    df["dst"] = df["dst_ip"].map(lambda a: addr_to_host.get(a, a))
    df["label"] = df["id"].isin(all_ids).astype(int)
    df["label_lm"] = df["id"].isin(lm_ids).astype(int)
    df = df.sort_values(["timestamp_abs", "id"], kind="stable").reset_index(drop=True)
    cols = ["timestamp", "src", "dst", "src_port", "dst_port", "proto", "label", "label_lm",
            "timestamp_abs", "id", "hostname", "src_ip", "dst_ip", "outbound", "principal",
            "image_path", "actor_id"]
    return df[cols], learned


def load_meta(meta_dir: str):
    import pandas as pd

    red = pd.read_csv(os.path.join(meta_dir, "optc_redteam.csv"))
    with open(os.path.join(meta_dir, "optc_known_addresses.json")) as f:
        known = json.load(f)
    return known, set(red["id"]), set(red.loc[red["label"] == "Lateral movement", "id"])


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _raw_files(out_dir: str) -> list[str]:
    return [os.path.join(d, f) for d, _, fs in os.walk(out_dir) for f in fs if f.endswith(".csv.gz")]


def cmd_index(args) -> int:
    fid = release_files()[args.tar]
    for name, off, size in tar_index(fid):
        print(name, off, size)
    return 0


def cmd_extract(args) -> int:
    tar = f"{args.day}.tar"
    fid = release_files()[tar]
    print(f"-> indice di {tar} (solo intestazioni)...")
    members = [m for m in cached_index(fid, tar, args.out) if m[0].endswith(".json.gz")
               and (not args.group or any(f"/{g}/" in m[0] for g in args.group))]
    todo = [m for m in members if not os.path.exists(_member_out(args.out, m[0]))]
    total = sum(m[2] for m in todo)
    print(f"   membri selezionati: {len(members)} | da estrarre: {len(todo)} ({total / 1e9:.2f} GB grezzi)")
    if not todo:
        return 0
    t0, done, failed = time.monotonic(), 0, []
    with cf.ProcessPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(extract_member, fid, n, off, sz, args.out): (n, sz) for n, off, sz in todo}
        for fut in cf.as_completed(futs):
            name, sz = futs[fut]
            try:
                _, c = fut.result()
            except Exception as e:  # noqa: BLE001 - report and continue with the other members
                failed.append((name, e))
                print(f"   [ERRORE] {name}: {e}", file=sys.stderr)
                continue
            done += sz
            el = time.monotonic() - t0
            print(f"   {os.path.basename(name)}: righe={c['lines']} start={c['flow_start']} tenuti={c['kept']} "
                  f"invalidi={c['bad']} | {done / 1e9:.2f}/{total / 1e9:.2f} GB, {done / 1e6 / el:.1f} MB/s", flush=True)
    if failed:
        print(f"[!] {len(failed)} membri falliti: rilancia lo stesso comando per riprenderli", file=sys.stderr)
        return 1
    return 0


def cmd_extract_local(args) -> int:
    paths = [os.path.join(d, f) for d, _, fs in os.walk(args.input_dir) for f in fs if f.endswith(".json.gz")]
    rels = [os.path.relpath(p, args.input_dir) for p in sorted(paths)]
    with cf.ProcessPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(_extract_file, os.path.join(args.input_dir, r), _member_out(args.out, r)): r for r in rels}
        for fut in cf.as_completed(futs):
            c = fut.result()
            print(f"   {futs[fut]}: righe={c['lines']} start={c['flow_start']} tenuti={c['kept']} invalidi={c['bad']}",
                  flush=True)
    return 0


def _extract_file(path: str, out_path: str) -> dict:
    with open(path, "rb") as raw:
        return extract_gzip_stream(raw, out_path)


def cmd_build(args) -> int:
    known, all_ids, lm_ids = load_meta(args.meta)
    paths = _raw_files(args.raw_dir)
    df, learned = build(paths, known, all_ids, lm_ids)
    df.to_csv(args.out, index=False)
    t0 = ORIGIN + dt.timedelta(seconds=float(df["timestamp_abs"].min()))
    print(f"file grezzi={len(paths)} flussi={len(df)} host/IP distinti={len(set(df['src']) | set(df['dst']))} "
          f"positivi tutti={int(df['label'].sum())} LM={int(df['label_lm'].sum())} "
          f"indirizzi appresi={len(learned)} inizio={t0.isoformat()} -> {args.out}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index", help="list the members of a per-day tar")
    s.add_argument("tar")
    s.set_defaults(fn=cmd_index)
    s = sub.add_parser("extract", help="stream-extract members of the corrected release")
    s.add_argument("--day", required=True, help="YYYY-MM-DD (one tar per day)")
    s.add_argument("--group", action="append", default=[], help="host group, e.g. AIA-201-225 (repeatable)")
    s.add_argument("--out", required=True)
    s.add_argument("--jobs", type=int, default=4, help="members streamed in parallel")
    s.set_defaults(fn=cmd_extract)
    s = sub.add_parser("extract-local", help="extract local .json.gz files (original release layout)")
    s.add_argument("input_dir")
    s.add_argument("--out", required=True)
    s.add_argument("--jobs", type=int, default=os.cpu_count(), help="files processed in parallel")
    s.set_defaults(fn=cmd_extract_local)
    s = sub.add_parser("build", help="merge, deduplicate and label")
    s.add_argument("raw_dir")
    s.add_argument("--meta", required=True, help="directory with optc_redteam.csv and optc_known_addresses.json")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_build)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
