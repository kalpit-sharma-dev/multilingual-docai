#!/usr/bin/env python3
"""
Fetch and organize model checkpoints into a local directory for offline use.

Models:
- LayoutLMv3 (default: microsoft/layoutlmv3-base)
- BLIP-2 (default: Salesforce/blip2-opt-2.7b) [LARGE]
- Pix2Struct (default: google/pix2struct-textcaps-base)
- Table T2T (default: google/flan-t5-small)

Usage examples:
  python scripts/utilities/fetch_models.py --target ./models \
    --layoutlm microsoft/layoutlmv3-base \
    --pix2struct google/pix2struct-textcaps-base \
    --table google/flan-t5-small \
    --skip-blip2

To fetch BLIP-2 as well (large download):
  python scripts/utilities/fetch_models.py --target ./models --blip2 Salesforce/blip2-opt-2.7b
"""

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_model(model_id: str, target_dir: Path) -> None:
    ensure_dir(target_dir)
    print(f"[fetch-models] Downloading {model_id} -> {target_dir} ...", flush=True)
    cache_dir = os.environ.get("HF_HOME", str(target_dir))
    # Download snapshot to a temp dir in cache, then copy/symlink to target_dir
    src = snapshot_download(repo_id=model_id, cache_dir=cache_dir, local_files_only=False)
    src_path = Path(src)
    # If target_dir is empty, copy tree; else leave as-is
    if not any(target_dir.iterdir()):
        print(f"[fetch-models] Copying files into {target_dir} ...", flush=True)
        for child in src_path.iterdir():
            dest = target_dir / child.name
            if child.is_dir():
                shutil.copytree(child, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(child, dest)
    print(f"[fetch-models] Done: {model_id}", flush=True)


def maybe_copy_fasttext_bin(target_root: Path) -> None:
    # Prefer /app/lid.176.bin if present (from Dockerfile build step)
    candidates = [Path("/app/lid.176.bin"), Path("lid.176.bin")] 
    for cand in candidates:
        if cand.exists():
            dest = target_root / "lid.176.bin"
            if not dest.exists():
                shutil.copy2(cand, dest)
                print(f"[fetch-models] Copied fastText lid.176.bin to {dest}")
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="./models", help="Target root directory for models")
    parser.add_argument("--layoutlm", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--blip2", type=str, default="Salesforce/blip2-opt-2.7b")
    parser.add_argument("--pix2struct", type=str, default="google/pix2struct-textcaps-base")
    parser.add_argument("--table", type=str, default="google/flan-t5-small")
    parser.add_argument("--skip-blip2", action="store_true", help="Skip BLIP-2 download (very large)")
    args = parser.parse_args()

    target_root = Path(args.target).resolve()
    ensure_dir(target_root)

    # HuggingFace caches
    os.environ.setdefault("HF_HOME", str(target_root))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(target_root))

    # LayoutLMv3 -> layoutlmv3-6class (fine-tuned optional). We place base weights here.
    download_model(args.layoutlm, target_root / "layoutlmv3-6class")

    # Pix2Struct -> pix2struct-chart
    download_model(args.pix2struct, target_root / "pix2struct-chart")

    # Table T2T -> table-t2t
    download_model(args.table, target_root / "table-t2t")

    # BLIP-2 (large)
    if not args.skip_blip2:
        download_model(args.blip2, target_root / "blip2-opt-2.7b")
    else:
        print("[fetch-models] Skipping BLIP-2 (use --skip-blip2=false to download)")

    # fastText language ID file
    maybe_copy_fasttext_bin(target_root)

    print("[fetch-models] All done.")


if __name__ == "__main__":
    main()


