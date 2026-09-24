"""Download and extract the PicoDomain dataset from GitHub.

Downloads:
  - Red Log.xlsx -> data/Red Log.xlsx
  - Zeek_Logs.7z -> data/Zeek_Logs.7z
And extracts Zeek_Logs.7z into data/logs/

Usage:
    python scripts/download_picodomain.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.request

RED_LOG_URL = "https://github.com/iHeartGraph/PicoDomain/raw/master/Red%20Log.xlsx"
ZEEK_LOGS_URL = "https://github.com/iHeartGraph/PicoDomain/raw/master/Zeek_Logs.7z"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(DATA_DIR, "logs")


def download_with_progress(url: str, dest_path: str) -> None:
    """Download a file with real-time percentage progress."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_file:
        total_size = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 128
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


def extract_7z(archive_path: str, dest_dir: str) -> bool:
    """Extract .7z archive using py7zr, Windows tar, or 7z executable."""
    os.makedirs(dest_dir, exist_ok=True)

    # Method 1: py7zr
    try:
        import py7zr
        print("  -> Extracting with py7zr...")
        with py7zr.SevenZipFile(archive_path, mode="r") as z:
            z.extractall(path=dest_dir)
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  [!] py7zr failed: {e}")

    # Method 2: Native Windows tar (bsdtar supports .7z on Windows 10/11)
    try:
        print("  -> Extracting with tar...")
        subprocess.run(
            ["tar", "-xf", archive_path, "-C", dest_dir],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except Exception:
        pass

    # Method 3: 7z executable
    candidates = [
        "7z",
        "7za",
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ]
    for cmd in candidates:
        if shutil.which(cmd) or os.path.exists(cmd):
            try:
                print(f"  -> Extracting with {cmd}...")
                subprocess.run(
                    [cmd, "x", archive_path, f"-o{dest_dir}", "-y"],
                    capture_output=True,
                    check=True,
                )
                return True
            except Exception:
                continue

    return False


def main() -> int:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print("=== PicoDomain Downloader & Extractor ===")
    print(f"Data destination: {DATA_DIR}\n")

    red_dest = os.path.join(DATA_DIR, "Red Log.xlsx")
    zeek_dest = os.path.join(DATA_DIR, "Zeek_Logs.7z")

    if os.path.exists(red_dest):
        print(f"[OK] Red Log.xlsx already present ({os.path.getsize(red_dest)} bytes)")
    else:
        print("-> Downloading Red Log.xlsx...")
        download_with_progress(RED_LOG_URL, red_dest)
        print("   Done.")

    if os.path.exists(zeek_dest):
        print(f"[OK] Zeek_Logs.7z already present ({os.path.getsize(zeek_dest) / (1024*1024):.2f} MB)")
    else:
        print("\n-> Downloading Zeek_Logs.7z (~16 MB)...")
        download_with_progress(ZEEK_LOGS_URL, zeek_dest)
        print("   Done.")

    print("\n-> Extracting Zeek_Logs.7z to data/logs/...")
    success = extract_7z(zeek_dest, LOGS_DIR)

    if success:
        extracted_files = []
        for root, _, files in os.walk(LOGS_DIR):
            for f in files:
                if f.endswith(".log"):
                    extracted_files.append(f)
        print(f"\n[OK] Extraction completed. Found {len(extracted_files)} .log file(s) in data/logs/")
    else:
        print(
            "\n[!] Unable to automatically extract .7z archive.\n"
            "    Please manually extract 'data/Zeek_Logs.7z' into 'data/logs/' "
            "using 7-Zip, tar, or PeaZip."
        )

    print("\n=== Done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
