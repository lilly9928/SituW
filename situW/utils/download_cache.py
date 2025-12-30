#!/usr/bin/env python3
import argparse
import os
import pathlib
import sys

from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser(
        description="Prefetch HuggingFace model repo into local cache (snapshot_download)."
    )
    p.add_argument("repo_id", help='e.g. "Qwen/Qwen2.5-72B-Instruct"')
    p.add_argument(
        "--revision",
        default=None,
        help="Optional git revision/tag/commit. Default: repo default branch.",
    )
    p.add_argument(
        "--cache-root",
        default="/scratch/sclab_kje/hg_weight",
        help="Root cache dir. Default: /scratch/sclab_kje/hg_weight",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel download workers (default: 8).",
    )
    args = p.parse_args()

    cache_root = pathlib.Path(args.cache_root)
    hub_cache = cache_root / "hub"
    transformers_cache = cache_root / "transformers"
    hub_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)

    # Make sure both HF hub + transformers use this cache root
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)

    token_set = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))

    print("== Prefetch HuggingFace repo ==")
    print("repo_id      :", args.repo_id)
    print("revision     :", args.revision)
    print("cache_root   :", str(cache_root))
    print("HF_TOKEN set :", token_set)
    print("max_workers  :", args.max_workers)
    sys.stdout.flush()

    local_dir = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        cache_dir=os.environ["HF_HUB_CACHE"],
        resume_download=True,
        local_files_only=False,
        max_workers=args.max_workers,
    )

    print("Cached at:", local_dir)


if __name__ == "__main__":
    main()
