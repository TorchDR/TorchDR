#!/usr/bin/env python3
"""
Download ImageNet-22k WebDataset from Hugging Face
"""

import sys
from huggingface_hub import snapshot_download
from pathlib import Path
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_imagenet22k(
    output_dir="/braid/vanasseh/imagenet-22k-wds",
    repo_id="timm/imagenet-22k-wds",
    max_workers=8,
):
    """
    Download ImageNet-22k WebDataset from Hugging Face

    Args:
        output_dir: Directory to save the dataset
        repo_id: Hugging Face repository ID
        max_workers: Number of parallel download threads
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting download of {repo_id}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Using {max_workers} parallel workers")

    try:
        # Download the entire repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(output_path),
            max_workers=max_workers,
            # Note: local_dir_use_symlinks and resume_download are deprecated
            # Downloads always resume when possible
        )

        logger.info(f"Download completed successfully to {local_dir}")

        # Print dataset statistics
        total_size = sum(
            f.stat().st_size for f in output_path.rglob("*") if f.is_file()
        )
        total_files = len(list(output_path.rglob("*")))
        logger.info(f"Total files: {total_files}")
        logger.info(f"Total size: {total_size / (1024**3):.2f} GB")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ImageNet-22k WebDataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/braid/vanasseh/imagenet-22k-wds",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--max-workers", type=int, default=8, help="Number of parallel download threads"
    )

    args = parser.parse_args()

    # Check if we're in the right environment
    if not Path("/braid").exists():
        logger.error("/braid not accessible. Make sure you're on the right node.")
        sys.exit(1)

    download_imagenet22k(output_dir=args.output_dir, max_workers=args.max_workers)
