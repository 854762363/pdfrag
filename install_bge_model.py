"""
Download BGE model files from HuggingFace mirror and install them into
the HuggingFace cache so sentence-transformers can load them offline.

Usage:
  python install_bge_model.py

If proxy is needed, set ALL_PROXY or http_proxy before running:
  export ALL_PROXY=socks5h://172.30.96.1:7897
  python install_bge_model.py

Or download files manually from browser, save to ./model_files/, then run.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.request import urlopen, Request

MODEL_ID = "BAAI/bge-small-zh-v1.5"
HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
# Fallback to official HF
HF_OFFICIAL = "https://huggingface.co"

# Files needed by sentence-transformers for BGE
REQUIRED_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
]

# Model weights (pick one)
MODEL_FILES = [
    "pytorch_model.bin",     # ~95MB for bge-small-zh-v1.5
]

CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"


def download_file(url: str, dest: Path) -> bool:
    """Download a single file. Returns True on success."""
    if dest.exists():
        print(f"  SKIP (exists): {dest.name}")
        return True
    try:
        req = Request(url, headers={"User-Agent": "BGE-Model-Installer/1.0"})
        with urlopen(req, timeout=60) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  OK ({size_mb:.1f}MB): {dest.name}")
        return True
    except Exception as e:
        print(f"  FAIL: {dest.name} - {e}")
        return False


def sha256_file(path: Path) -> str:
    """SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def install_into_cache(download_dir: Path) -> str | None:
    """
    Given a directory with raw model files, install them into the HF cache.

    Returns the snapshot hash, or None on failure.
    """
    model_dir = CACHE_DIR / "models--BAAI--bge-small-zh-v1.5"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Create blobs and snapshot
    blobs = model_dir / "blobs"
    blobs.mkdir(exist_ok=True)

    all_files = list(download_dir.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]

    if not all_files:
        print("ERROR: No files found in download directory")
        return None

    # Copy files as blobs (named by sha256), then create symlinks in snapshot
    snapshot_hash = hashlib.sha256(
        b"BAAI/bge-small-zh-v1.5"
    ).hexdigest()[:40]

    snapshot_dir = model_dir / "snapshots" / snapshot_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for src_file in all_files:
        # Compute blob name = sha256 of content
        blob_hash = sha256_file(src_file)
        blob_path = blobs / blob_hash

        if not blob_path.exists():
            shutil.copy2(src_file, blob_path)

        # Symlink in snapshot
        rel_path = src_file.relative_to(download_dir)
        link_path = snapshot_dir / rel_path
        link_path.parent.mkdir(parents=True, exist_ok=True)
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(blob_path)

    # Write refs/main
    refs_dir = model_dir / "refs"
    refs_dir.mkdir(exist_ok=True)
    (refs_dir / "main").write_text(snapshot_hash)

    print(f"\nSnapshot: {snapshot_hash}")
    print(f"Cache installed at: {model_dir}")
    return snapshot_hash


def try_download(base_url: str) -> Path | None:
    """Try to download all files from a base URL."""
    download_dir = Path("_bge_download")
    download_dir.mkdir(exist_ok=True)

    all_files = REQUIRED_FILES + MODEL_FILES
    success = True

    for fname in all_files:
        url = f"{base_url}/{MODEL_ID}/resolve/main/{fname}"
        dest = download_dir / fname
        if not download_file(url, dest):
            success = False

    # Also try model.safetensors as fallback
    pytorch_bin = download_dir / "pytorch_model.bin"
    if not pytorch_bin.exists():
        print("pytorch_model.bin not found, trying model.safetensors...")
        url = f"{base_url}/{MODEL_ID}/resolve/main/model.safetensors"
        dest = download_dir / "model.safetensors"
        if download_file(url, dest):
            # Both pytorch_model.bin and model.safetensors can work;
            # sentence-transformers prefers safetensors
            pass

    if not success:
        print("\nSome files failed to download.")
        print(f"Manual download: visit {base_url}/{MODEL_ID}/tree/main in browser,")
        print(f"download all files, save to: {download_dir.resolve()}")
        return download_dir if any(download_dir.iterdir()) else None

    return download_dir


def main():
    print(f"Installing {MODEL_ID}...")
    print(f"Mirror: {HF_MIRROR}")
    print()

    # Try mirror first, then official
    download_dir = None
    for base_url in [HF_MIRROR, HF_OFFICIAL]:
        print(f"Trying: {base_url}")
        download_dir = try_download(base_url)
        if download_dir and list(download_dir.rglob("*")):
            break
        print(f"Failed with {base_url}\n")

    if not download_dir or not list(download_dir.rglob("*")):
        print("AUTOMATIC DOWNLOAD FAILED.")
        print()
        print("Manual steps:")
        print(f"1. Open in browser: {HF_MIRROR}/{MODEL_ID}/tree/main")
        print(f"2. Download all files into: _bge_download/")
        print(f"3. Run this script again: python install_bge_model.py")
        return 1

    # Install into HF cache
    snapshot = install_into_cache(download_dir)
    if snapshot:
        print()
        print("SUCCESS! Model installed to HF cache.")
        print()
        print("Now run: HF_HUB_OFFLINE=1 python -c \"")
        print("  from sentence_transformers import SentenceTransformer")
        print("  m = SentenceTransformer('BAAI/bge-small-zh-v1.5')")
        print("  print('Dim:', m.get_sentence_embedding_dimension())")
        print("\"")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
