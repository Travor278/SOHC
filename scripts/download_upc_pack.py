"""Download the UPC 36-cell pack WLTP/CC-CV dataset from Dataverse.

The script uses the public Dataverse API, writes files into
`data/pack_wltp_upc/`, and skips files whose size and MD5 checksum already
match the remote metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests


API_ROOT = "https://dataverse.csuc.cat/api"
DEFAULT_PID = "doi:10.34810/DATA2395"
DEFAULT_OUT = Path("data/pack_wltp_upc")


def fetch_manifest(persistent_id: str = DEFAULT_PID) -> list[dict]:
    """Return Dataverse file metadata for the UPC pack dataset."""
    url = f"{API_ROOT}/datasets/:persistentId/"
    response = requests.get(url, params={"persistentId": persistent_id}, timeout=60)
    response.raise_for_status()
    files = response.json()["data"]["latestVersion"]["files"]
    rows = []
    for item in files:
        data_file = item["dataFile"]
        rows.append(
            {
                "id": int(data_file["id"]),
                "filename": str(data_file["filename"]),
                "size": int(data_file.get("filesize", 0)),
                "md5": str(data_file.get("md5") or data_file.get("checksum", {}).get("value", "")),
                "content_type": str(data_file.get("contentType", "")),
            }
        )
    return sorted(rows, key=lambda row: row["filename"])


def download_file(row: dict, out_dir: Path, *, chunk_size: int = 1024 * 1024, retries: int = 3) -> Path:
    """Download one Dataverse datafile and atomically move it into place."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / row["filename"]
    if _is_complete(target, row):
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    url = f"{API_ROOT}/access/datafile/{row['id']}"
    params = {"format": "original"} if row.get("content_type") != "application/octet-stream" else None
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            if tmp.exists():
                tmp.unlink()
            with requests.get(url, params=params, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
            if not _is_complete(tmp, row):
                raise RuntimeError(f"checksum/size mismatch after downloading {row['filename']}")
            tmp.replace(target)
            return target
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                print(f"retry {attempt}/{retries} {row['filename']}: {exc}")
            elif tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"failed to download {row['filename']} after {retries} attempts: {last_error}")


def write_manifest(rows: list[dict], out_dir: Path, *, persistent_id: str) -> Path:
    """Write a local JSON manifest for reproducibility."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "CORA.RDR Dataverse",
        "persistent_id": persistent_id,
        "api_root": API_ROOT,
        "files": rows,
    }
    path = out_dir / "manifest_upc_pack.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _is_complete(path: Path, row: dict) -> bool:
    """Check file size and MD5 checksum against Dataverse metadata."""
    if not path.exists():
        return False
    expected = str(row.get("md5") or "").lower()
    if expected:
        return _md5(path) == expected
    if int(row.get("size") or 0) and path.stat().st_size != int(row["size"]):
        return False
    return True


def _md5(path: Path) -> str:
    """Compute a file MD5 checksum in chunks."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Download UPC pack files and write a manifest."""
    parser = argparse.ArgumentParser(description="Download UPC 36-cell pack dataset")
    parser.add_argument("--persistent-id", default=DEFAULT_PID)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="Debug limit after filtering")
    parser.add_argument("--pattern", default=None, help="Optional substring filter on filenames")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    all_rows = fetch_manifest(args.persistent_id)
    manifest_path = write_manifest(all_rows, args.out_dir, persistent_id=args.persistent_id)
    rows = list(all_rows)
    if args.pattern:
        rows = [row for row in rows if args.pattern.lower() in row["filename"].lower()]
    if args.limit is not None:
        rows = rows[: args.limit]
    total_size = sum(row["size"] for row in rows)
    print(f"files={len(rows)} total_MB={total_size / 1024 / 1024:.1f} manifest={manifest_path}")
    if args.metadata_only:
        return

    for idx, row in enumerate(rows, start=1):
        target = args.out_dir / row["filename"]
        status = "skip" if _is_complete(target, row) else "download"
        print(f"[{idx}/{len(rows)}] {status} {row['filename']} ({row['size'] / 1024 / 1024:.1f} MB)")
        sys.stdout.flush()
        download_file(row, args.out_dir)


if __name__ == "__main__":
    main()
