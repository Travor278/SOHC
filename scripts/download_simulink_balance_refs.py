"""Download local reference images for active-balancing Simulink/circuit study.

These images are external reference material. They are downloaded into
`external_refs/` for local reading only and are intentionally excluded from git.
Do not present them as project-generated results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen


SOURCE_REPO = "https://github.com/yavuzhanocak/Single-switch-capacitor-battery-balance"
README_RAW = (
    "https://raw.githubusercontent.com/yavuzhanocak/"
    "Single-switch-capacitor-battery-balance/master/README.md"
)
DEFAULT_OUT = Path("external_refs/simulink_balance/images")


def image_urls_from_readme() -> list[str]:
    """Extract README image URLs from the active-balancing GitHub reference."""
    text = _read_url(README_RAW).decode("utf-8")
    urls = re.findall(r"!\[[^\]]*\]\((https://[^)]+)\)", text)
    if not urls:
        raise RuntimeError(f"No image URLs found in {README_RAW}")
    return urls


def download_images(out_dir: Path = DEFAULT_OUT) -> list[dict]:
    """Download reference images and return manifest rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, url in enumerate(image_urls_from_readme(), start=1):
        suffix = Path(url.split("?")[0]).suffix or ".png"
        name = f"single_switch_capacitor_{idx:02d}{suffix}"
        path = out_dir / name
        if not path.exists():
            path.write_bytes(_read_url(url))
        rows.append(
            {
                "index": idx,
                "filename": name,
                "source_url": url,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    manifest = {
        "source_repository": SOURCE_REPO,
        "license_note": "No LICENSE file found in source repo; use only as local reference unless permission is obtained.",
        "files": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return rows


def _sha256(path: Path) -> str:
    """Compute a SHA-256 digest for one downloaded image."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_url(url: str, timeout: int = 120) -> bytes:
    """Read one URL with a small user agent for GitHub-hosted reference images."""
    request = Request(url, headers={"User-Agent": "SOHC-reference-downloader/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def main() -> None:
    """CLI entry point for downloading local reference images."""
    parser = argparse.ArgumentParser(description="Download Simulink balancing reference images")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    rows = download_images(args.out_dir)
    for row in rows:
        print(f"{row['index']:02d} {row['filename']} {row['bytes']} bytes")
    print(f"manifest={args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
