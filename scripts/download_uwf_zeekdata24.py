"""Download the UWF-ZeekData24 dataset from the official repository.

Usage:
    python scripts/download_uwf_zeekdata24.py [--dest ./data/uwf_zeekdata24]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.request

BASE_URL = "https://datasets.uwf.edu/data/UWF-ZeekData24/csv"
DEFAULT_DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "uwf_zeekdata24")

CATEGORIES = [
    "Benign",
    "Credential_Access",
    "Defense_Evasion",
    "Exfiltration",
    "Initial_Access",
    "Persistence",
    "Privilege_Escalation",
    "Reconnaissance",
]


def download_file(url: str, dest_path: str) -> None:
    """Download a file with a simple progress report."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 128  # 128 KB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = downloaded * 100 / total_size
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r    [{percent:5.1f}%] {mb_down:.2f} / {mb_tot:.2f} MB")
                sys.stdout.flush()
        sys.stdout.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UWF-ZeekData24 CSV datasets.")
    parser.add_argument(
        "--dest",
        type=str,
        default=DEFAULT_DEST,
        help=f"Target directory for downloaded data (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args()

    dest_dir = os.path.abspath(args.dest)
    os.makedirs(dest_dir, exist_ok=True)
    print("=== UWF-ZeekData24 Downloader ===")
    print(f"Destination: {dest_dir}\n")

    grand_total_downloaded = 0

    for cat in CATEGORIES:
        cat_url = f"{BASE_URL}/{cat}/"
        cat_dest = os.path.join(dest_dir, cat)
        os.makedirs(cat_dest, exist_ok=True)

        print(f"-> Checking category: {cat}")
        try:
            req = urllib.request.Request(cat_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                html = resp.read().decode("utf-8")

            # Extract part-*.csv files from Apache directory listing
            files = re.findall(r'href="([^"/]+?\.csv)"', html)
            csv_files = sorted(set(files))

            if not csv_files:
                print(f"   No CSV files found in {cat_url}")
                continue

            for fname in csv_files:
                file_url = cat_url + fname
                target_file = os.path.join(cat_dest, fname)

                if os.path.exists(target_file):
                    size = os.path.getsize(target_file)
                    print(f"   [ALREADY PRESENT] {fname} ({size / (1024*1024):.2f} MB)")
                    continue

                print(f"   Downloading: {fname}...")
                download_file(file_url, target_file)
                grand_total_downloaded += os.path.getsize(target_file)

        except Exception as e:
            print(f"   [ERROR] {cat}: {e}", file=sys.stderr)

    print("\n=== Download completed successfully ===")
    if grand_total_downloaded > 0:
        print(f"Total downloaded in this session: {grand_total_downloaded / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
