"""Selective downloader for AIT Log Data Set (AIT-LDS 2023, Zenodo).

Uses HTTP Range requests with a block cache to read the remote Zenodo ZIP archive (15 GB)
and extracts ONLY the essential network and host log files (Suricata EVE, Auth logs, Labels),
skipping the massive 14.7 GB raw PCAP files.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile


class BufferedRemoteStream(io.RawIOBase):
    """Buffered seekable stream over an HTTP resource using 32MB cached blocks."""

    def __init__(self, url: str, block_size: int = 32 * 1024 * 1024):
        self.url = url
        self.block_size = block_size
        req = urllib.request.Request(url, method="HEAD")
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req) as resp:
                    self.size = int(resp.headers.get("Content-Length", 0))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 4:
                    time.sleep(5 * (attempt + 1))
                else:
                    raise
        self.pos = 0
        self.buf = b""
        self.buf_start = -1

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        buf_end = self.buf_start + len(self.buf)
        if not (self.buf_start <= self.pos < buf_end):
            fetch_size = max(len(b), self.block_size)
            end = min(self.size - 1, self.pos + fetch_size - 1)
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req) as resp:
                        self.buf = resp.read()
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 429 and attempt < 4:
                        time.sleep(5 * (attempt + 1))
                    else:
                        raise
            self.buf_start = self.pos
            buf_end = self.buf_start + len(self.buf)

        offset = self.pos - self.buf_start
        avail = min(len(b), len(self.buf) - offset)
        b[:avail] = self.buf[offset : offset + avail]
        self.pos += avail
        return avail


def download_ait_scenario(
    scenario: str = "fox",
    target_dir: str = "data/ait_lds",
    zenodo_record: str = "5789064",
) -> None:
    url = f"https://zenodo.org/records/{zenodo_record}/files/{scenario}.zip?download=1"
    out_dir = os.path.abspath(os.path.join(target_dir, scenario))
    os.makedirs(out_dir, exist_ok=True)

    print("================================================================================")
    print(f"      AIT-LDS Selective Downloader: Scenario '{scenario}'")
    print("================================================================================")
    print(f"Zenodo URL:     {url}")
    print(f"Target dir:     {out_dir}\n")

    print("[downloader] Connecting to Zenodo...")
    t0 = time.time()
    stream = BufferedRemoteStream(url)
    zf = zipfile.ZipFile(stream)
    print(f"[downloader] Connected! Total remote archive size: {stream.size / (1024*1024*1024):.2f} GB")

    # Select only the strictly essential log files and ground-truth labels
    selected_names = [
        n
        for n in zf.namelist()
        if (
            (
                (n.startswith("gather/") and ("eve.json" in n or "auth.log" in n))
                or n.startswith("labels/")
                or n == "dataset.yaml"
            )
            and not n.endswith(".pcap")
            and ".pcap." not in n
            and not n.endswith("/")
        )
    ]

    selected_infos = [zf.getinfo(n) for n in selected_names]
    tot_uncomp = sum(i.file_size for i in selected_infos)
    tot_comp = sum(i.compress_size for i in selected_infos)

    print(f"\n[downloader] Selected {len(selected_names)} essential files:")
    print(f"  - Download payload:     {tot_comp / (1024*1024):.1f} MB (vs ~15 GB full archive)")
    print(f"  - Uncompressed size:   {tot_uncomp / (1024*1024):.1f} MB\n")

    for idx, info in enumerate(selected_infos, 1):
        rel_path = info.filename
        dest_path = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if os.path.exists(dest_path) and os.path.getsize(dest_path) == info.file_size:
            print(f"[{idx:>2}/{len(selected_infos)}] Already downloaded: {rel_path}")
            continue

        print(
            f"[{idx:>2}/{len(selected_infos)}] Downloading {rel_path} "
            f"({info.file_size / (1024*1024):.2f} MB)..."
        )
        data = zf.read(info)
        with open(dest_path, "wb") as f_out:
            f_out.write(data)

    elapsed = time.time() - t0
    print(
        f"\n[downloader] SUCCESS! Extracted {len(selected_names)} files into {out_dir} "
        f"in {elapsed:.1f}s."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AIT-LDS scenario logs from Zenodo.")
    parser.add_argument("--scenario", default="fox", help="AIT scenario name (default: fox)")
    parser.add_argument(
        "--target-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ait_lds"),
        help="Destination directory",
    )
    args = parser.parse_args()
    download_ait_scenario(args.scenario, args.target_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
