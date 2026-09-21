"""Offline tests for scripts/download_optc.py (no network, no dataset needed).

The downloader talks to Google Drive, whose failure modes are exactly the ones a naive
downloader gets wrong silently: a folder listing truncated at 50 entries, two sibling folders
that differ only by case, the virus-scan interstitial on large files, a quota page saved as if
it were data, a connection cut mid-file. Each of them is reproduced here, the download ones
against a local ``http.server``.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import os
import re
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "download_optc.py")
_FOLDER = "application/vnd.google-apps.folder"
_QUOTA_HTML = (
    "<p>Sorry, you can't view or download this file at this time. "
    "Too many users have viewed or downloaded this file recently.</p>"
)


@pytest.fixture(scope="module")
def _module():
    spec = importlib.util.spec_from_file_location("download_optc", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # @dataclass with postponed annotations looks its module up here
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


@pytest.fixture
def dl(_module, monkeypatch):
    """The module under test, with backoff/listing delays removed."""
    monkeypatch.setattr(_module.time, "sleep", lambda s: None)
    return _module


# ---------------------------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------------------------


def _js_escape(s: str) -> str:
    """What Drive does to its JSON payload before embedding it in a single-quoted JS string."""
    return s.replace("\\", "\\\\").replace("'", "\\x27").replace('"', "\\x22").replace("/", "\\/")


def _row(id_: str, name: str, folder: bool = False, size: int = 0) -> list:
    row = [None] * 20
    row[0], row[1], row[2] = id_, ["parent"], name
    row[3] = _FOLDER if folder else "application/gzip"
    row[13] = size or None
    return row


def _page(rows) -> str:
    payload = json.dumps([rows, None, None, None, [[1]], 1])
    return f"<html><script>var x = 1;</script><script>window['_DRIVE_ivd'] = '{_js_escape(payload)}';</script></html>"


class _Resp:
    """Minimal stand-in for an urlopen response."""

    def __init__(self, body: str, ctype: str = "text/html"):
        self._body = body.encode()
        self.headers = type("H", (), {"get_content_type": lambda s: ctype})()

    def read(self, n=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_parse_folder_page_decodes_js_escapes(dl):
    names = ["plain.json.gz", "caf\u00e9 \u4e2d", 'we"ird\'name', "back\\slash", "a/b"]
    rows = [_row(f"id{i}", n, size=100 + i) for i, n in enumerate(names)] + [_row("dir", "sub", folder=True)]
    entries = dl._parse_folder_page(_page(rows))
    assert [e.name for e in entries] == names + ["sub"]
    assert [e.size for e in entries] == [100, 101, 102, 103, 104, 0]  # folders carry no size
    assert [e.is_dir for e in entries] == [False] * 5 + [True]
    assert entries[0].id == "id0"


def test_parse_empty_folder_and_missing_blob(dl):
    assert dl._parse_folder_page(f"<script>window['_DRIVE_ivd'] = '{_js_escape(json.dumps([None, 1]))}';</script>") == []
    with pytest.raises(RuntimeError):
        dl._parse_folder_page("<html>nothing here</html>")


def test_list_folder_fails_closed_at_page_cap(dl, monkeypatch):
    def listing(n):
        page = _page([_row(f"id{i}", f"f{i}.gz", size=1) for i in range(n)])
        monkeypatch.setattr(dl, "_open", lambda opener, req, tries=0: _Resp(page))
        return dl.list_folder(None, "folder", "bro/2019-09-23")

    assert len(listing(dl.PAGE_CAP - 1)) == dl.PAGE_CAP - 1
    with pytest.raises(dl.ListingTruncated, match="bro/2019-09-23"):
        listing(dl.PAGE_CAP)


_EMPTY_BLOB = "<script>window['_DRIVE_ivd'] = '';</script>"  # the listing is missing: json.loads('') fails


def test_list_folder_retries_a_page_without_listing(dl, monkeypatch):
    pages = iter(["<html>unusual traffic</html>", _EMPTY_BLOB, _page([_row("id0", "a.gz", size=1)])])
    monkeypatch.setattr(dl, "_open", lambda opener, req, tries=0: _Resp(next(pages)))
    assert [e.name for e in dl.list_folder(None, "f", "ecar/benign")] == ["a.gz"]


def test_list_folder_raises_a_clean_error_when_the_page_never_parses(dl, monkeypatch):
    calls = []
    monkeypatch.setattr(dl, "_open", lambda opener, req, tries=0: calls.append(1) or _Resp(_EMPTY_BLOB))
    with pytest.raises(RuntimeError, match="non interpretabile") as exc:  # not a bare JSONDecodeError
        dl.list_folder(None, "f", "ecar/benign")
    assert len(calls) == dl.RETRIES and "ecar/benign" in str(exc.value)


def test_truncated_listing_is_not_retried(dl, monkeypatch):
    calls = []
    page = _page([_row(f"id{i}", f"f{i}.gz") for i in range(dl.PAGE_CAP)])
    monkeypatch.setattr(dl, "_open", lambda opener, req, tries=0: calls.append(1) or _Resp(page))
    with pytest.raises(dl.ListingTruncated):
        dl.list_folder(None, "f", "bro/2019-09-23")
    assert len(calls) == 1  # a full page is a fact about the folder, not a glitch


# ---------------------------------------------------------------------------------------------
# Interstitial / quota
# ---------------------------------------------------------------------------------------------

_INTERSTITIAL = (
    '<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">'
    '<input type="hidden" name="id" value="FILEID"><input type="hidden" name="export" value="download">'
    '<input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="5a11-b2">'
    "</form>"
)


def test_confirm_url_rebuilds_the_get_form(dl):
    url = urllib.parse.urlparse(dl._confirm_url(_INTERSTITIAL))
    assert f"{url.scheme}://{url.netloc}{url.path}" == "https://drive.usercontent.google.com/download"
    assert urllib.parse.parse_qs(url.query) == {
        "id": ["FILEID"], "export": ["download"], "confirm": ["t"], "uuid": ["5a11-b2"],
    }


def test_confirm_url_tolerates_entities_and_attribute_order(dl):
    page = ('<form action="https://x.test/dl?a=1&amp;b=2"><input name="k" value="v&amp;w" type="hidden">'
            '<input type="submit" name="go" value="Download"></form>')
    url = urllib.parse.urlparse(dl._confirm_url(page))
    assert urllib.parse.parse_qs(url.query) == {"a": ["1"], "b": ["2"], "k": ["v&w"]}  # submit input ignored


def test_confirm_url_without_form_raises(dl):
    with pytest.raises(RuntimeError, match="form di conferma"):
        dl._confirm_url("<html>quota? no, just nothing</html>")


def test_quota_page_is_recognised_not_saved(dl):
    with pytest.raises(dl.QuotaExceeded):
        dl._read_html(_Resp(_QUOTA_HTML))
    assert "form" in dl._read_html(_Resp(_INTERSTITIAL))


# ---------------------------------------------------------------------------------------------
# Selection: case collisions, --path, walk
# ---------------------------------------------------------------------------------------------


def test_case_colliding_siblings_get_distinct_names_in_any_order(dl):
    a = dl.Entry("1nTeKNDA", "23Sep-night", True, 0)
    b = dl.Entry("1oaP8KQZ", "23Sep-Night", True, 0)
    other = dl.Entry("1XOGWT8R", "23Sep19-red", True, 0)
    m1 = dl._local_names([a, b, other], "ecar-bro/evaluation")
    m2 = dl._local_names([b, other, a], "ecar-bro/evaluation")
    assert m1 == m2  # independent of the order Drive lists siblings in
    assert m1["1nTeKNDA"] == "23Sep-night__1nTeKN" and m1["1oaP8KQZ"] == "23Sep-Night__1oaP8K"
    assert m1["1XOGWT8R"] == "23Sep19-red"  # non-colliding names are untouched
    assert len({v.casefold() for v in m1.values()}) == 3  # unique on a case-insensitive filesystem


@pytest.mark.parametrize("raw, safe", [("a/b", "a_b"), ("..", "_"), ("", "_"), (" x ", "x"), ("ok.gz", "ok.gz")])
def test_safe_name_keeps_remote_names_inside_dest(dl, raw, safe):
    assert dl._safe_name(raw) == safe


@pytest.mark.parametrize("rel, want", [
    ("evaluation", True),  # ancestor of the prefix: must descend
    ("evaluation/23Sep19-red", True),
    ("evaluation/23Sep19-red/AIA-201-225", True),  # the prefix itself
    ("evaluation/23Sep19-red/AIA-201-225/ecarbro.json.gz", True),  # below it
    ("evaluation/23Sep19-red/AIA-1-25", False),  # sibling bucket
    ("benign", False),
    ("evaluation/23Sep19-redX", False),  # shared string prefix is not a path prefix
])
def test_path_filter(dl, rel, want):
    assert dl._wanted(rel, ["evaluation/23Sep19-red/AIA-201-225"]) is want
    assert dl._wanted(rel, []) is True


def _tree(dl):
    E = dl.Entry
    return {
        "root": [E("eb", "ecar-bro", True, 0), E("e", "ecar", True, 0), E("b", "bro", True, 0)],
        "eb": [E("ben", "benign", True, 0), E("ev", "evaluation", True, 0)],
        "ben": [E("f1", "a.json.gz", False, 100)],
        "ev": [E("d1", "23Sep-night", True, 0), E("d2", "23Sep-Night", True, 0)],
        "d1": [E("f2", "x.json.gz", False, 50)],
        "d2": [E("f3", "x.json.gz", False, 70)],
        "e": [E("ee", "evaluation", True, 0)],
        "ee": [E("f4", "big.json.gz", False, 5000)],
    }


@pytest.fixture
def fake_drive(dl, monkeypatch):
    tree = _tree(dl)

    def fake_list(opener, folder_id, where):
        if folder_id == "b":
            raise dl.ListingTruncated(f"{where}: 50 voci")
        return tree[folder_id]

    monkeypatch.setattr(dl, "ROOT_FOLDER_ID", "root")
    monkeypatch.setattr(dl, "list_folder", fake_list)
    monkeypatch.setattr(dl, "fetch_meta", lambda opener, dest: None)
    return tree


def test_walk_keeps_case_colliding_folders_apart(dl, fake_drive):
    out: list = []
    dl.walk(None, "eb", "ecar-bro", "", "ecar-bro", [], out)
    locals_ = sorted(f.local_path for f in out)
    assert locals_ == sorted([
        os.path.join("ecar-bro", "benign", "a.json.gz"),
        os.path.join("ecar-bro", "evaluation", "23Sep-night__d1", "x.json.gz"),
        os.path.join("ecar-bro", "evaluation", "23Sep-Night__d2", "x.json.gz"),
    ])
    assert sum(f.size for f in out) == 220  # both x.json.gz survive: nothing merged


def test_walk_prunes_to_the_requested_path(dl, fake_drive):
    out: list = []
    dl.walk(None, "eb", "ecar-bro", "", "ecar-bro", ["evaluation/23Sep-Night"], out)
    assert [f.drive_path for f in out] == ["ecar-bro/evaluation/23Sep-Night/x.json.gz"]


# ---------------------------------------------------------------------------------------------
# main(): dry-run, guards, failure modes
# ---------------------------------------------------------------------------------------------


def test_dry_run_downloads_and_creates_nothing(dl, fake_drive, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(dl, "download_file", lambda *a: pytest.fail("dry-run must not download"))
    dest = tmp_path / "optc"
    assert dl.main(["--dry-run", "--dest", str(dest)]) == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out and "Selezionati: 3 file" in out
    assert not dest.exists()


def test_default_subset_is_ecar_bro_only(dl, fake_drive, tmp_path, capsys):
    dl.main(["--dry-run", "--dest", str(tmp_path / "d")])
    out = capsys.readouterr().out
    assert "ecar-bro/benign" in out and "big.json.gz" not in out and "ecar/evaluation" not in out


def test_large_selection_needs_yes(dl, fake_drive, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(dl, "LARGE_BYTES", 100)
    got = []
    monkeypatch.setattr(dl, "download_file", lambda opener, f, dest: got.append(f.drive_path))
    dest = str(tmp_path / "d")
    assert dl.main(["--dest", dest]) == 2  # 220 bytes > 100
    assert "--yes" in capsys.readouterr().err and got == []
    assert dl.main(["--dest", dest, "--yes"]) == 0
    assert len(got) == 3


def test_present_files_are_skipped_only_if_size_matches(dl, fake_drive, tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(dl, "download_file", lambda opener, f, dest: got.append(f.drive_path))
    dest = tmp_path / "d"
    ok = dest / "ecar-bro" / "benign" / "a.json.gz"
    ok.parent.mkdir(parents=True)
    ok.write_bytes(b"x" * 100)  # matches the listing
    bad = dest / "ecar-bro" / "evaluation" / "23Sep-night__d1" / "x.json.gz"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"x" * 7)  # wrong size: re-download
    assert dl.main(["--dest", str(dest)]) == 0
    assert "ecar-bro/benign/a.json.gz" not in got and "ecar-bro/evaluation/23Sep-night/x.json.gz" in got


def test_truncated_listing_aborts_with_error(dl, fake_drive, tmp_path, capsys):
    assert dl.main(["--subset", "bro", "--dry-run", "--dest", str(tmp_path / "d")]) == 2
    assert "50 voci" in capsys.readouterr().err


def test_path_matching_nothing_is_an_error(dl, fake_drive, tmp_path, capsys):
    assert dl.main(["--path", "evaluation/typo", "--dry-run", "--dest", str(tmp_path / "d")]) == 2
    assert "nessun file" in capsys.readouterr().err


def test_one_failed_file_does_not_stop_the_rest(dl, fake_drive, tmp_path, capsys, monkeypatch):
    def fake_download(opener, f, dest):
        if f.drive_path.endswith("a.json.gz"):
            raise dl.QuotaExceeded("quota")

    monkeypatch.setattr(dl, "download_file", fake_download)
    calls = []
    monkeypatch.setattr(dl, "_is_present", lambda d, f: calls.append(f.drive_path) or False)
    assert dl.main(["--dest", str(tmp_path / "d")]) == 1  # non-zero: the corpus is incomplete
    assert "1 file non scaricati" in capsys.readouterr().err
    assert len(calls) >= 3  # the other files were still attempted


# ---------------------------------------------------------------------------------------------
# download_file against a local server that behaves like Drive
# ---------------------------------------------------------------------------------------------

PAYLOAD = gzip.compress(os.urandom(300_000))  # incompressible: a few hundred KB of gzip


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls: list = []

    def log_message(self, *a):
        pass

    def _reply(self, body: bytes, ctype: str, status: int = 200, headers=None, send: int | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body if send is None else body[:send])
        self.wfile.flush()
        if send is not None:  # promised len(body) bytes, delivered fewer: cut the connection
            self.close_connection = True

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(url.query)
        mode = q.get("id", [""])[0]
        rng = self.headers.get("Range")
        _Handler.calls.append((url.path, mode, rng))
        base = f"http://127.0.0.1:{self.server.server_port}"

        if url.path == "/uc" and mode == "interstitial":
            form = (f'<form id="download-form" action="{base}/confirm" method="get">'
                    '<input type="hidden" name="id" value="direct"><input type="hidden" name="confirm" value="t">'
                    '<input type="hidden" name="uuid" value="u-1"></form>')
            return self._reply(form.encode(), "text/html; charset=utf-8")
        if url.path == "/confirm":
            if q.get("confirm") != ["t"] or q.get("uuid") != ["u-1"]:
                return self._reply(b"bad request", "text/plain", 400)
            mode = "direct"
        if mode == "quota":
            return self._reply(_QUOTA_HTML.encode(), "text/html")
        if mode == "badgz":
            return self._reply(b"<html>not a gzip</html>", "application/octet-stream")
        if mode == "empty":
            return self._reply(b"", "application/octet-stream")
        if mode == "dead":  # promises the file, sends nothing
            return self._reply(PAYLOAD, "application/octet-stream", send=0)
        if mode == "norange":  # ignores Range and always replies 200 with everything
            return self._reply(PAYLOAD, "application/octet-stream")

        start = int(re.match(r"bytes=(\d+)-", rng).group(1)) if rng else 0
        if start >= len(PAYLOAD):
            return self._reply(b"", "text/plain", 416)
        body = PAYLOAD[start:]
        cut = len(body) // 2 if (mode == "flaky" and rng is None) else None
        if rng:
            hdr = {"Content-Range": f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}"}
            return self._reply(body, "application/octet-stream", 206, hdr)
        return self._reply(body, "application/octet-stream", send=cut)


@pytest.fixture
def drive(dl, monkeypatch, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Handler.calls = []
    monkeypatch.setattr(dl, "DOWNLOAD_URL", f"http://127.0.0.1:{server.server_port}/uc?id={{id}}")
    yield tmp_path
    server.shutdown()
    server.server_close()


def _file(dl, id_: str, size: int = len(PAYLOAD)):
    return dl.RemoteFile(id_, f"ecar-bro/x/{id_}.json.gz", os.path.join("ecar-bro", "x", f"{id_}.json.gz"), size)


def _dest(root, f) -> str:
    return os.path.join(str(root), f.local_path)


def test_download_direct_file(dl, drive):
    f = _file(dl, "direct")
    dl.download_file(dl._make_opener(), f, str(drive))
    assert open(_dest(drive, f), "rb").read() == PAYLOAD
    assert not os.path.exists(_dest(drive, f) + ".part")


def test_download_goes_through_the_interstitial(dl, drive):
    f = _file(dl, "interstitial")
    dl.download_file(dl._make_opener(), f, str(drive))
    assert open(_dest(drive, f), "rb").read() == PAYLOAD
    assert [c[0] for c in _Handler.calls] == ["/uc", "/confirm"]


def test_quota_page_is_never_saved_as_data(dl, drive):
    f = _file(dl, "quota", size=123)
    with pytest.raises(dl.QuotaExceeded):
        dl.download_file(dl._make_opener(), f, str(drive))
    assert not os.path.exists(_dest(drive, f)) and not os.path.exists(_dest(drive, f) + ".part")


def test_html_disguised_as_gzip_is_discarded(dl, drive):
    body = b"<html>not a gzip</html>"
    f = _file(dl, "badgz", size=len(body))  # size matches, so only the gzip check can catch it
    with pytest.raises(RuntimeError, match="gzip"):
        dl.download_file(dl._make_opener(), f, str(drive))
    assert not os.path.exists(_dest(drive, f)) and not os.path.exists(_dest(drive, f) + ".part")


def test_unknown_size_html_disguised_as_gzip_is_still_rejected(dl, drive):
    f = _file(dl, "badgz", size=0)  # no size to compare against: the gzip check is the only defence
    with pytest.raises(RuntimeError, match="gzip"):
        dl.download_file(dl._make_opener(), f, str(drive))
    assert not os.path.exists(_dest(drive, f))


def test_empty_remote_file_is_saved_as_an_empty_file(dl, drive):
    f = _file(dl, "empty", size=0)  # gzip reads a 0-byte file as an empty stream: must not be rejected on every rerun
    dl.download_file(dl._make_opener(), f, str(drive))
    assert os.path.getsize(_dest(drive, f)) == 0


def test_size_mismatch_is_an_error_not_a_file(dl, drive):
    f = _file(dl, "direct", size=len(PAYLOAD) - 10)  # listing says smaller than what is served
    with pytest.raises(RuntimeError, match="incompleto"):
        dl.download_file(dl._make_opener(), f, str(drive))
    assert not os.path.exists(_dest(drive, f))


def test_cut_connection_resumes_with_range(dl, drive):
    f = _file(dl, "flaky")
    dl.download_file(dl._make_opener(), f, str(drive))
    assert open(_dest(drive, f), "rb").read() == PAYLOAD
    assert _Handler.calls[0][2] is None and _Handler.calls[1][2].startswith("bytes=")


def test_existing_part_file_is_resumed(dl, drive):
    f = _file(dl, "direct")
    os.makedirs(os.path.dirname(_dest(drive, f)))
    with open(_dest(drive, f) + ".part", "wb") as p:
        p.write(PAYLOAD[:1000])
    dl.download_file(dl._make_opener(), f, str(drive))
    assert open(_dest(drive, f), "rb").read() == PAYLOAD
    assert _Handler.calls[0][2] == "bytes=1000-"


def test_server_ignoring_range_restarts_instead_of_corrupting(dl, drive):
    f = _file(dl, "norange")
    os.makedirs(os.path.dirname(_dest(drive, f)))
    with open(_dest(drive, f) + ".part", "wb") as p:
        p.write(b"x" * 1000)  # junk prefix: appending the 200 body to it would corrupt the file
    dl.download_file(dl._make_opener(), f, str(drive))
    assert open(_dest(drive, f), "rb").read() == PAYLOAD
    # restarted in place on the 200: without the 206 check the junk would be appended first and only
    # the size guard would notice, costing a second full download
    assert len(_Handler.calls) == 1


def test_dead_server_gives_up_after_retries(dl, drive):
    f = _file(dl, "dead")
    with pytest.raises(RuntimeError, match="incompleto"):
        dl.download_file(dl._make_opener(), f, str(drive))
    assert not os.path.exists(_dest(drive, f))
