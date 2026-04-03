#!/usr/bin/env python3
"""Download all benchmark datasets to data/ directory.

Usage:
    python scripts/download_datasets.py                  # Download all
    python scripts/download_datasets.py D-01 D-05        # Download specific datasets
    python scripts/download_datasets.py --list            # List available datasets
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"

# Dataset definitions: ID -> (HuggingFace repo or URL, target dir, description)
DATASETS = {
    "D-01": {
        "name": "OmniDocBench",
        "hf_repo": "opendatalab/OmniDocBench",
        "target": DATA_DIR / "omnidocbench",
        "description": "Overall text, tables, math, layout (EN + ZH). PRIMARY benchmark.",
        "priority": "HIGH",
    },
    "D-02": {
        "name": "FinTabNet",
        "hf_repo": "bsmock/FinTabNet.c",
        "target": DATA_DIR / "fintabnet",
        "description": "Financial table structure from annual reports.",
        "priority": "HIGH",
    },
    "D-03": {
        "name": "FUNSD",
        "hf_repo": "nielsr/funsd-layoutlmv3",
        "target": DATA_DIR / "funsd",
        "description": "Form/field understanding on noisy scans.",
        "priority": "MEDIUM",
    },
    "D-04": {
        "name": "DocVQA",
        "hf_repo": "lmms-lab/DocVQA",
        "target": DATA_DIR / "docvqa",
        "description": "Document Q&A; form + table comprehension.",
        "priority": "MEDIUM",
    },
    "D-05": {
        "name": "OlmOCR-Bench",
        "hf_repo": "allenai/olmOCR-bench",
        "target": DATA_DIR / "olmocr_bench",
        "description": "English precision; 1,400+ PDFs; 7,000+ binary unit tests.",
        "priority": "HIGH",
    },
    "D-06": {
        "name": "DocLayNet",
        "hf_repo": "ds4sd/DocLayNet-v1.2",
        "target": DATA_DIR / "doclaynet",
        "description": "Real enterprise layout -- 80K+ pages, 11 element types.",
        "priority": "HIGH",
    },
}


def check_huggingface_cli() -> bool:
    """Check if huggingface-cli is available."""
    try:
        subprocess.run(["huggingface-cli", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_with_huggingface_cli(repo: str, target: Path) -> bool:
    """Download dataset using huggingface-cli."""
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        "huggingface-cli", "download",
        repo,
        "--repo-type", "dataset",
        "--local-dir", str(target),
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def download_with_python(repo: str, target: Path) -> bool:
    """Download dataset using huggingface_hub Python API."""
    try:
        from huggingface_hub import snapshot_download
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(target),
        )
        return True
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error("Download failed for %s: %s", repo, e)
        return False


def download_dataset(dataset_id: str) -> bool:
    """Download a single dataset."""
    info = DATASETS.get(dataset_id)
    if not info:
        logger.error("Unknown dataset: %s", dataset_id)
        return False

    repo = info["hf_repo"]
    target = info["target"]
    name = info["name"]

    # Check if already downloaded
    if target.exists() and any(target.iterdir()):
        existing = list(target.iterdir())
        logger.info("%s (%s) already exists with %d items, skipping", dataset_id, name, len(existing))
        return True

    logger.info("Downloading %s (%s) from %s ...", dataset_id, name, repo)

    if check_huggingface_cli():
        return download_with_huggingface_cli(repo, target)
    else:
        return download_with_python(repo, target)


def main():
    parser = argparse.ArgumentParser(description="Download ExtractMark benchmark datasets")
    parser.add_argument("datasets", nargs="*", help="Dataset IDs to download (e.g. D-01 D-05)")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--high-only", action="store_true", help="Download HIGH priority only")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable datasets:\n")
        for did, info in DATASETS.items():
            status = "EXISTS" if info["target"].exists() and any(info["target"].iterdir()) else "NOT DOWNLOADED"
            print(f"  {did}  [{info['priority']}]  {info['name']:20s}  {status}")
            print(f"        {info['description']}")
            print(f"        HF: {info['hf_repo']}")
            print()
        return

    # Determine which datasets to download
    if args.datasets:
        targets = args.datasets
    elif args.high_only:
        targets = [did for did, info in DATASETS.items() if info["priority"] == "HIGH"]
    else:
        targets = list(DATASETS.keys())

    logger.info("Downloading %d datasets: %s", len(targets), targets)

    results = {}
    for did in targets:
        results[did] = download_dataset(did)

    # Summary
    print("\n--- Download Summary ---")
    for did, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {did} ({DATASETS[did]['name']}): {status}")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
