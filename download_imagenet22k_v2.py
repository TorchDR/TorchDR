#!/usr/bin/env python3
"""
Download ImageNet-22k WebDataset from Hugging Face using hf_hub_download
"""

import sys
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
from pathlib import Path
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_file(args):
    """Download a single file"""
    filename, repo_id, output_dir = args
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            force_filename=filename,
        )
        return filename, True, None
    except Exception as e:
        return filename, False, str(e)


def download_imagenet22k(
    output_dir="/braid/vanasseh/imagenet-22k-wds",
    repo_id="timm/imagenet-22k-wds",
    max_workers=8,
    pattern="*.tar",
):
    """
    Download ImageNet-22k WebDataset from Hugging Face

    Args:
        output_dir: Directory to save the dataset
        repo_id: Hugging Face repository ID
        max_workers: Number of parallel download threads
        pattern: File pattern to download (e.g., "*.tar" for tar files only)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    logger.info(f"Authenticated as: {api.whoami()['name']}")
    logger.info(f"Starting download of {repo_id}")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Using {max_workers} parallel workers")

    try:
        # List all files in the repository
        logger.info("Fetching file list...")
        all_files = list_repo_files(repo_id, repo_type="dataset")

        # Filter files based on pattern
        if pattern == "*.tar":
            files_to_download = [f for f in all_files if f.endswith(".tar")]
        elif pattern == "all":
            files_to_download = all_files
        else:
            files_to_download = [f for f in all_files if pattern in f]

        logger.info(f"Found {len(files_to_download)} files to download")

        # Download files in parallel
        failed_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            download_args = [(f, repo_id, str(output_path)) for f in files_to_download]
            futures = {
                executor.submit(download_file, args): args[0] for args in download_args
            }

            with tqdm(total=len(files_to_download), desc="Downloading files") as pbar:
                for future in as_completed(futures):
                    filename, success, error = future.result()
                    if success:
                        pbar.set_description(f"Downloaded {filename}")
                    else:
                        logger.warning(f"Failed to download {filename}: {error}")
                        failed_files.append(filename)
                    pbar.update(1)

        if failed_files:
            logger.warning(f"{len(failed_files)} files failed to download:")
            for f in failed_files[:10]:  # Show first 10 failures
                logger.warning(f"  - {f}")

        # Print dataset statistics
        total_size = sum(
            f.stat().st_size for f in output_path.rglob("*") if f.is_file()
        )
        total_files = len(list(output_path.rglob("*")))
        logger.info("Download completed!")
        logger.info(f"Total files: {total_files}")
        logger.info(f"Total size: {total_size / (1024**3):.2f} GB")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        logger.error("Make sure you have accepted the dataset terms at:")
        logger.error("https://huggingface.co/datasets/timm/imagenet-22k-wds")
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
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tar",
        help="File pattern to download (*.tar for tar files, 'all' for everything)",
    )

    args = parser.parse_args()

    # Check if we're in the right environment
    if not Path("/braid").exists():
        logger.error("/braid not accessible. Make sure you're on the right node.")
        sys.exit(1)

    download_imagenet22k(
        output_dir=args.output_dir, max_workers=args.max_workers, pattern=args.pattern
    )
