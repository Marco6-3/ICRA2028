#!/usr/bin/env python3
"""
Stream-extract ONLY non-image arrays (6D Wrench, Gripper Force/Width, 6-DOF TCP Pose/Vel,
GelSight Mini marker offsets & embeddings, action, timestamp, meta/episode_ends) from
official RDP full .zarr.zip archives via standard HTTP Range requests.
Saves ~50 GB of RGB video bandwidth while preserving 100% of full-rate physical data!
"""

import argparse
import io
import os
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP Range with a tail buffer for zip central directory."""

    def __init__(self, url: str, tail_cache_bytes: int = 8 * 1024 * 1024):
        self.orig_url = url
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            self.url = r.url  # follow redirects once
            self.size = int(r.headers["Content-Length"])
        self.pos = 0
        # Prefetch tail so zipfile central directory parsing takes 1 HTTP call
        self.tail_start = max(0, self.size - tail_cache_bytes)
        self.tail_buf = self._fetch_range(self.tail_start, self.size - 1)

    def _fetch_range(self, start: int, end: int) -> bytes:
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.read()
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))

    def readable(self):
        return True

    def seekable(self):
        return True

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

    def read(self, n: int = -1) -> bytes:
        if n == 0 or self.pos >= self.size:
            return b""
        end = self.size - 1 if n < 0 else min(self.size - 1, self.pos + n - 1)
        if self.pos >= self.tail_start:
            s = self.pos - self.tail_start
            e = end - self.tail_start + 1
            data = self.tail_buf[s:e]
        else:
            data = self._fetch_range(self.pos, end)
        self.pos += len(data)
        return data


def download_entry(url: str, info: zipfile.ZipInfo, out_dir: Path, strip_prefix: str):
    """Fetch a single zip member's local header + compressed payload via Range and decompress."""
    rel = info.filename
    if strip_prefix and rel.startswith(strip_prefix):
        rel = rel[len(strip_prefix) :]
    if not rel or rel.endswith("/"):
        return 0
    dst = out_dir / rel
    if dst.exists() and dst.stat().st_size == info.file_size:
        return info.file_size

    # Local header is 30 + fname_len + extra_len; fetch generous margin (512B + compress_size)
    start = info.header_offset
    end = start + 512 + info.compress_size
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
            # Parse local file header to find exact payload offset
            assert raw[:4] == b"PK\x03\x04", f"Bad local header for {info.filename}"
            fn_len = int.from_bytes(raw[26:28], "little")
            ex_len = int.from_bytes(raw[28:30], "little")
            data_off = 30 + fn_len + ex_len
            comp = raw[data_off : data_off + info.compress_size]
            if info.compress_type == zipfile.ZIP_STORED:
                payload = comp
            elif info.compress_type == zipfile.ZIP_DEFLATED:
                import zlib
                payload = zlib.decompress(comp, -zlib.MAX_WBITS)
            else:
                raise ValueError(f"Unsupported compress_type={info.compress_type}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(payload)
            return len(payload)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(0.5 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Inspecting remote zip central directory: {args.url}")
    rf = HTTPRangeFile(args.url, tail_cache_bytes=12 * 1024 * 1024)
    print(f"Remote zip size: {rf.size / (1024**3):.2f} GB")

    with zipfile.ZipFile(rf, "r") as zf:
        infos = zf.infolist()
    print(f"Total zip entries: {len(infos)}")

    # Find prefix ending in replay_buffer.zarr/ (or parent of data/.zgroup)
    strip_prefix = ""
    for inf in infos:
        if inf.filename.endswith("/data/.zgroup"):
            strip_prefix = inf.filename[: -len("data/.zgroup")]
            break

    # Keep all non-image entries (skip *_img/ directories!)
    keep = []
    for inf in infos:
        fn = inf.filename
        if fn.endswith("/"):
            continue
        if "_img/" in fn:
            continue
        keep.append(inf)

    tot_comp_mb = sum(i.compress_size for i in keep) / (1024 * 1024)
    print(f"Keeping {len(keep)} non-image entries ({tot_comp_mb:.2f} MB compressed) | strip_prefix='{strip_prefix}'")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(download_entry, rf.url, inf, out_dir, strip_prefix) for inf in keep]
        for fut in as_completed(futs):
            done_bytes += fut.result()

    print(f"Extracted {len(keep)} non-image zarr files ({done_bytes / (1024*1024):.2f} MB) to {out_dir} in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
