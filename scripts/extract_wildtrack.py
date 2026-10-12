"""Extract Wildtrack_dataset_full.zip despite wrong zip64 offsets in its central directory (docs/PROTOCOL_REAL.md).

The mirror's archive stores some local-header offsets modulo 2**32 incorrectly. For every central-directory entry this
tries the stored offset and the stored offset minus 2**32, accepts a candidate only if a local header with the same file
name is found there, then inflates the data with the central directory's compressed size and checks size and CRC-32.
Entries that cannot be recovered are listed. Nothing here touches the dataset's content.

    python scripts/extract_wildtrack.py data/raw/Wildtrack_dataset_full.zip data
"""

from __future__ import annotations

import struct
import sys
import zipfile
import zlib
from pathlib import Path

WRAP = 1 << 32


def locate(f, info: zipfile.ZipInfo) -> int | None:  # type: ignore[no-untyped-def]
    name = info.filename.encode("utf-8")
    for off in (info.header_offset, info.header_offset - WRAP, info.header_offset + WRAP):
        if off < 0:
            continue
        f.seek(off)
        hdr = f.read(30)
        if len(hdr) < 30 or hdr[:4] != b"PK\x03\x04":
            continue
        n, m = struct.unpack("<HH", hdr[26:30])
        if f.read(n) == name:
            return off + 30 + n + m
    return None


def main(zip_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    ok = bad = 0
    failed: list[str] = []
    with zipfile.ZipFile(zip_path) as z, open(zip_path, "rb") as f:
        for info in z.infolist():
            if info.filename.startswith("__MACOSX") or info.filename.endswith(".DS_Store"):
                continue
            dest = out / info.filename
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            start = locate(f, info)
            if start is None:
                bad += 1
                failed.append(info.filename)
                continue
            f.seek(start)
            raw = f.read(info.compress_size)
            data = zlib.decompress(raw, -15) if info.compress_type == 8 else raw
            if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                bad += 1
                failed.append(info.filename)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            ok += 1
    print(f"extracted {ok}, failed {bad}")
    for name in failed[:20]:
        print("  FAILED", name)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
