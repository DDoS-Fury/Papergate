"""Download the DARPA OpTC dataset from its public Google Drive release.

The release (~1 TB, https://github.com/FiveDirections/OpTC-data) has three sub-trees:

  ecar-bro/  eCAR FLOW events annotated with the bro uid        ~7 GB   (default)
  ecar/      full endpoint telemetry, host buckets of 25 hosts  ~1 TB
  bro/       network sensor logs by date                        (see caveats)

WARNING: ecar-bro is NOT usable for lateral-movement evaluation. Measured on all 80 files of
ecar-bro/evaluation (8.68 M rows): every row is an outbound flow to an external address (ports
80/443), none has both endpoints internal, and none of the 1359 event ids of LMDEval's
optc_redteam.csv appears. Internal flows and the labelled ids exist only in ecar/ (see
docs/external_dataset_optc.md §10).

Everything lands under --dest mirroring the Drive tree, so ``<dest>/ecar`` is the directory
LMDEval's extract_optc.py expects. ``<dest>/meta`` gets the ground-truth PDF, the eCAR docs and
the LMDEval label files, pinned by commit.

Usage:
    python scripts/download_optc.py --dry-run                 # list what would be fetched
    python scripts/download_optc.py                           # ecar-bro (~7 GB)
    python scripts/download_optc.py --subset ecar-bro ecar --path evaluation/23Sep19-red/AIA-201-225
    python scripts/download_optc.py --subset ecar --yes       # everything in ecar (~1 TB)

Caveats, all observed on the live folder (2026-09-21):
  * Drive's folder page lists at most 50 children and carries no page token. A listing that
    reaches 50 entries is treated as truncated and aborts: a silently partial corpus is worse
    than no corpus. bro/<date>/ is affected (needs rclone or the Drive API).
  * Two siblings differ only by case (evaluation/23Sep-night vs 23Sep-Night) with different
    content: on a case-insensitive filesystem they would merge, so colliding names get an id
    suffix.
  * Files over ~100 MB go through Drive's virus-scan interstitial (a form to re-submit).
  * Drive's per-file daily quota answers with an HTML page; it is reported, never saved as data.

The download mechanism (re-submitting the interstitial form, ``.part`` + atomic rename, gzip
verification) is modelled on the Go tool datalog/examples/optc/fetch from
https://github.com/swdunlop/pkg (MIT); the code here is independent.
"""

from __future__ import annotations

import argparse
import gzip
import html
import http.client
import http.cookiejar
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import defaultdict
from dataclasses import dataclass

ROOT_FOLDER_ID = "1n3kkS3KR31KUegn42yk3-e6JkZvf0Caa"
FOLDER_URL = "https://drive.google.com/drive/folders/{id}?hl=en"
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={id}"
FOLDER_MIME = "application/vnd.google-apps.folder"

PAGE_CAP = 50  # Drive's folder page lists at most this many children
SUBSETS = ("ecar-bro", "ecar", "bro")
LARGE_BYTES = 50 * 10**9  # abort without --yes above this (all of ecar is ~1 TB)
LIST_DELAY = 0.3  # seconds between folder-listing requests
RETRIES = 4

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = os.path.join(PROJECT_ROOT, "data", "optc")

# Companion files, pinned by commit so labels and docs cannot change under us.
_OPTC_DATA = "https://raw.githubusercontent.com/FiveDirections/OpTC-data/5b108604f11f767aa11ea79ff827595f3fad15fd/"
_LMDEVAL = "https://raw.githubusercontent.com/cl-anssi/LMDEval/2b95a68387fa59853426c671c5ef3fced6cb5dd0/"
META_FILES = [
    ("OpTC-data", n, _OPTC_DATA + n) for n in ("README.md", "ecar.md", "errata.md", "OpTCRedTeamGroundTruth.pdf")
] + [
    ("LMDEval", n, _LMDEVAL + n) for n in ("optc_redteam.csv", "optc_known_addresses.json", "LICENSE")
]

_QUOTA_RE = re.compile(r"too many users|download quota|can.t view or download this file", re.I)


class ListingTruncated(RuntimeError):
    """A Drive folder listing hit the page cap, so it may be incomplete."""


class QuotaExceeded(RuntimeError):
    """Drive refused the download because of its per-file quota."""


@dataclass(frozen=True)
class Entry:
    id: str
    name: str
    is_dir: bool
    size: int


@dataclass(frozen=True)
class RemoteFile:
    id: str
    drive_path: str  # "<subset>/<Drive names>"
    local_path: str  # relative to --dest; case-collisions disambiguated
    size: int


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------


def _make_opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    return opener


def _open(opener, request, tries: int = RETRIES):
    """opener.open with exponential backoff on 429/5xx and on network errors."""
    for attempt in range(tries):
        try:
            return opener.open(request, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                raise
            e.close()
        except (OSError, http.client.HTTPException):
            if attempt == tries - 1:
                raise
        time.sleep(2**attempt * 2)


# ---------------------------------------------------------------------------------------------
# Listing (Drive folder page, anonymous)
# ---------------------------------------------------------------------------------------------


def _unescape_js(s: str) -> str:
    r"""Decode the body of a single-quoted JS string literal (\xHH, \uHHHH, \/, \\, \')."""

    def rep(m: re.Match) -> str:
        t = m.group(1)
        if len(t) > 1:  # \xHH or \uHHHH
            return chr(int(t[1:], 16))
        return {"n": "\n", "t": "\t", "r": "\r"}.get(t, t)

    return re.sub(r"\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)", rep, s, flags=re.S)


def _parse_folder_page(page: str) -> list[Entry]:
    """Children of a Drive folder from the ``_DRIVE_ivd`` blob embedded in its HTML page."""
    for script in re.findall(r"<script[^>]*>(.*?)</script>", page, re.S):
        if "_DRIVE_ivd" not in script:
            continue
        literals = re.findall(r"'((?:[^'\\]|\\.)*)'", script)  # 1st is the key, 2nd the payload
        if len(literals) < 2:
            continue
        rows = json.loads(_unescape_js(literals[1]))[0] or []
        # row layout: [id, [parents], name, mimeType, ..., size at index 13]
        return [Entry(r[0], r[2], r[3] == FOLDER_MIME, int(r[13] or 0) if len(r) > 13 else 0) for r in rows]
    raise RuntimeError("elenco non trovato nella pagina Drive (formato cambiato?)")


def list_folder(opener, folder_id: str, where: str) -> list[Entry]:
    for attempt in range(1, RETRIES + 1):
        with _open(opener, FOLDER_URL.format(id=folder_id)) as resp:
            page = resp.read().decode("utf-8", "replace")
        try:
            entries = _parse_folder_page(page)
            break
        except (ValueError, RuntimeError, IndexError, TypeError) as e:
            # Under load Drive occasionally serves a page without the listing (seen once, while three
            # jobs hit it at once; not reproducible sequentially): retry like a 5xx instead of crashing
            if attempt == RETRIES:
                raise RuntimeError(f"{where}: pagina Drive non interpretabile dopo {RETRIES} tentativi ({e})") from e
            time.sleep(2**attempt)
    if len(entries) >= PAGE_CAP:
        raise ListingTruncated(
            f"{where}: {len(entries)} voci, Drive tronca l'elenco a {PAGE_CAP} e senza credenziali non c'è "
            "paginazione: il dataset risulterebbe incompleto. Usa rclone (docs/external_dataset_optc.md §10) "
            "o restringi --path a cartelle più piccole."
        )
    time.sleep(LIST_DELAY)
    return entries


def _safe_name(name: str) -> str:
    """Drive allows '/' and '..' in names; a remote name must never leave --dest."""
    name = name.replace("/", "_").replace("\\", "_").strip()
    return "_" if name in ("", ".", "..") else name


def _local_names(children: list[Entry], where: str) -> dict[str, str]:
    """Drive id -> local name, unique also on case-insensitive filesystems.

    Every member of a colliding group gets the suffix (not just the later ones), so the layout
    does not depend on the order Drive lists siblings in.
    """
    groups: dict[str, list[Entry]] = defaultdict(list)
    for c in children:
        groups[_safe_name(c.name).casefold()].append(c)
    out = {}
    for group in groups.values():
        if len(group) > 1:
            print(f"    [!] {where}: nomi uguali a meno delle maiuscole ({', '.join(c.name for c in group)}): "
                  "salvati con suffisso __<id>")
        for c in group:
            out[c.id] = _safe_name(c.name) if len(group) == 1 else f"{_safe_name(c.name)}__{c.id[:6]}"
    return out


def _wanted(rel: str, prefixes: list[str]) -> bool:
    """True if the subset-relative path is inside, or on the way to, one of the --path prefixes."""
    return not prefixes or any(rel == p or rel.startswith(p + "/") or p.startswith(rel + "/") for p in prefixes)


def walk(opener, folder_id, subset, rel, local, prefixes, out: list[RemoteFile]) -> None:
    where = f"{subset}/{rel}" if rel else subset
    children = list_folder(opener, folder_id, where)
    local_names = _local_names(children, where)
    for c in children:
        child_rel = f"{rel}/{c.name}" if rel else c.name
        if not _wanted(child_rel, prefixes):
            continue
        child_local = os.path.join(local, local_names[c.id])
        if c.is_dir:
            walk(opener, c.id, subset, child_rel, child_local, prefixes, out)
        else:
            out.append(RemoteFile(c.id, f"{subset}/{child_rel}", child_local, c.size))


# ---------------------------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------------------------


def _is_html(resp) -> bool:
    return resp.headers.get_content_type() == "text/html"


def _read_html(resp) -> str:
    with resp:
        page = resp.read(1 << 20).decode("utf-8", "replace")
    if _QUOTA_RE.search(page):
        raise QuotaExceeded(
            "quota giornaliera di Drive esaurita per questo file. Riprova più tardi (fino a 24 h) oppure "
            "copia la cartella nel tuo Drive e scarica con rclone; lo script non aggira il limite."
        )
    return page


def _confirm_url(page: str) -> str:
    """Rebuild Drive's virus-scan confirmation form (method GET) as a URL."""
    action = re.search(r'<form[^>]*\baction="([^"]+)"', page, re.I)
    if not action:
        raise RuntimeError("form di conferma non trovato nella pagina intermedia di Drive (formato cambiato?)")
    params = {}
    for tag in re.findall(r"<input\b[^>]*>", page, re.I):
        attrs = dict(re.findall(r'([\w-]+)="([^"]*)"', tag))
        if attrs.get("type", "").lower() == "hidden" and "name" in attrs:
            params[html.unescape(attrs["name"])] = html.unescape(attrs.get("value", ""))
    url = html.unescape(action.group(1))
    return url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)


def _open_file(opener, file_id: str, offset: int):
    """Open the byte stream of a Drive file, going through the interstitial if one is served."""

    def request(url: str) -> urllib.request.Request:
        req = urllib.request.Request(url)
        if offset:
            req.add_header("Range", f"bytes={offset}-")
        return req

    resp = _open(opener, request(DOWNLOAD_URL.format(id=file_id)))
    if _is_html(resp):
        resp = _open(opener, request(_confirm_url(_read_html(resp))))
        if _is_html(resp):
            _read_html(resp)  # raises QuotaExceeded if that is what it is
            raise RuntimeError("Drive ha risposto con una pagina HTML invece del file")
    return resp


def _stream(resp, out, done: int, total: int) -> None:
    last = 0.0
    while True:
        chunk = resp.read(1 << 20)
        if not chunk:
            break
        out.write(chunk)
        done += len(chunk)
        if total and time.monotonic() - last >= 0.5:
            sys.stdout.write(f"\r    [{done * 100 / total:5.1f}%] {done / 1e6:.1f} / {total / 1e6:.1f} MB")
            sys.stdout.flush()
            last = time.monotonic()
    if total:
        sys.stdout.write(f"\r    [{done * 100 / total:5.1f}%] {done / 1e6:.1f} / {total / 1e6:.1f} MB\n")
        sys.stdout.flush()


def _gzip_ok(path: str) -> bool:
    try:
        with gzip.open(path, "rb") as g:
            while g.read(1 << 20):
                pass
        return True
    except (OSError, EOFError, zlib.error):
        return False


def _is_present(dest_root: str, f: RemoteFile) -> bool:
    p = os.path.join(dest_root, f.local_path)
    return os.path.isfile(p) and (not f.size or os.path.getsize(p) == f.size)


def download_file(opener, f: RemoteFile, dest_root: str) -> None:
    """Fetch one file to ``<dest>/<local_path>``.

    Data goes to ``.part`` (resumed with Range across attempts and reruns) and is renamed only
    after the size matches the listing and the gzip stream is intact, so a file at its final
    name is always complete.
    """
    dest = os.path.join(dest_root, f.local_path)
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, RETRIES + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if f.size and have > f.size:
            os.remove(part)
            have = 0
        if not f.size or have < f.size:
            try:
                resp = _open_file(opener, f.id, have)
                with resp:
                    if have and resp.status != 206:  # server ignored Range: start over
                        have = 0
                    with open(part, "ab" if have else "wb") as out:
                        _stream(resp, out, have, f.size)
            except (QuotaExceeded, urllib.error.HTTPError):
                raise
            except (OSError, http.client.HTTPException) as e:
                if attempt == RETRIES:
                    raise
                print(f"\n    [!] {type(e).__name__}: {e}; nuovo tentativo ({attempt + 1}/{RETRIES})")
                time.sleep(2**attempt)
                continue
        have = os.path.getsize(part)
        if not f.size or have == f.size:
            break
        if attempt == RETRIES:
            raise RuntimeError(f"download incompleto: {have} di {f.size} byte")
    if f.local_path.endswith((".gz", ".tgz")) and not _gzip_ok(part):
        os.remove(part)
        raise RuntimeError("il file scaricato non è un gzip valido (pagina d'errore o dati corrotti): scartato")
    os.replace(part, dest)


def fetch_meta(opener, dest_root: str) -> None:
    for sub, name, url in META_FILES:
        dest = os.path.join(dest_root, "meta", sub, name)
        if os.path.exists(dest):
            print(f"   [GIA' PRESENTE] meta/{sub}/{name}")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with _open(opener, url) as resp, open(dest + ".part", "wb") as out:
            shutil.copyfileobj(resp, out)
        os.replace(dest + ".part", dest)
        print(f"   meta/{sub}/{name} ({os.path.getsize(dest) / 1e3:.0f} KB)")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def _free_bytes(path: str) -> int:
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Download the DARPA OpTC dataset (Google Drive release).")
    p.add_argument("--subset", nargs="+", choices=SUBSETS, default=["ecar-bro"],
                   help="sub-trees to fetch (default: ecar-bro, ~7 GB; ecar is ~1 TB)")
    p.add_argument("--path", action="append", default=[], metavar="PREFIX",
                   help="restrict to a path inside each selected subset, e.g. evaluation/23Sep19-red/AIA-201-225 "
                        "(repeatable; Drive names, case-sensitive)")
    p.add_argument("--dest", default=DEFAULT_DEST, help=f"target directory (default: {DEFAULT_DEST})")
    p.add_argument("--dry-run", action="store_true", help="list files and sizes, download nothing")
    p.add_argument("--yes", action="store_true", help=f"allow downloads above {_human(LARGE_BYTES)}")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    dest = os.path.abspath(args.dest)
    prefixes = [p.strip("/") for p in args.path]
    subsets = list(dict.fromkeys(args.subset))

    print("=== OpTC Downloader ===")
    print(f"Destinazione: {dest}")
    print(f"Subset: {', '.join(subsets)}" + (f" | path: {', '.join(prefixes)}" if prefixes else "") + "\n")

    opener = _make_opener()
    files: list[RemoteFile] = []
    try:
        root = list_folder(opener, ROOT_FOLDER_ID, "radice")
        for subset in subsets:
            node = next((c for c in root if c.is_dir and c.name == subset), None)
            if node is None:
                print(f"[ERRORE] '{subset}' non trovata nella radice della release", file=sys.stderr)
                return 2
            print(f"-> Elenco {subset}...")
            walk(opener, node.id, subset, "", subset, prefixes, files)
    except (RuntimeError, OSError, http.client.HTTPException) as e:
        print(f"[ERRORE] {e}", file=sys.stderr)
        return 2
    if not files:
        print("[ERRORE] nessun file selezionato (controlla --subset / --path)", file=sys.stderr)
        return 2

    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for f in files:
        g = groups["/".join(f.drive_path.split("/")[:2])]
        g[0] += 1
        g[1] += f.size
    print()
    for name in sorted(groups):
        print(f"   {name:<32} {groups[name][0]:>5} file  {_human(groups[name][1]):>10}")
    todo = [f for f in files if not _is_present(dest, f)]
    need = sum(f.size for f in todo)
    print(f"\nSelezionati: {len(files)} file, {_human(sum(f.size for f in files))} | "
          f"già presenti: {len(files) - len(todo)} | da scaricare: {len(todo)} ({_human(need)})")
    print(f"Spazio libero: {_human(_free_bytes(dest))}")

    if args.dry_run:
        print(f"\n[DRY-RUN] nessun download. Metadati ({len(META_FILES)} file da GitHub, commit pinnati) non scaricati.")
        if need > LARGE_BYTES:
            print(f"    Il download reale di questa selezione richiede --yes (oltre {_human(LARGE_BYTES)}).")
        return 0
    if need > _free_bytes(dest):
        print(f"[ERRORE] spazio insufficiente: servono {_human(need)}", file=sys.stderr)
        return 2
    if need > LARGE_BYTES and not args.yes:
        print(f"[ERRORE] {_human(need)} da scaricare, oltre {_human(LARGE_BYTES)}: rilancia con --yes "
              "(oppure restringi con --path)", file=sys.stderr)
        return 2

    os.makedirs(dest, exist_ok=True)
    failed: list[tuple[RemoteFile, Exception]] = []
    try:
        print("\n-> Metadati (GitHub, commit pinnati)")
        fetch_meta(opener, dest)
        for i, f in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {f.drive_path} ({_human(f.size)})")
            try:
                download_file(opener, f, dest)
            except (RuntimeError, OSError, http.client.HTTPException) as e:
                print(f"    [ERRORE] {e}", file=sys.stderr)
                failed.append((f, e))
    except KeyboardInterrupt:
        print("\n[!] Interrotto: rilancia lo stesso comando per riprendere (i .part vengono ripresi).")
        return 130

    if failed:
        print(f"\n[!] {len(failed)} file non scaricati:", file=sys.stderr)
        for f, e in failed:
            print(f"    {f.drive_path}: {e}", file=sys.stderr)
        print("    Rilancia lo stesso comando: i file completi vengono saltati, i .part ripresi.", file=sys.stderr)
        return 1
    print("\n=== Download completato! ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
