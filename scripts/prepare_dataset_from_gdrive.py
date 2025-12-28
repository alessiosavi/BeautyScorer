#!/usr/bin/env python3
"""
Download images from Google Drive folders and prepare dataset for training.

This script:
1. Reads a CSV with folder_name, folder_id, score, and photo columns
2. Downloads all photos from each Google Drive folder IN PARALLEL
3. Creates a local folder structure compatible with BeautyDataset
4. Generates a training-ready CSV file

Usage:
    python scripts/prepare_dataset_from_gdrive.py \
        --input data/source.csv \
        --output-dir data/images \
        --output-csv data/dataset.csv \
        --workers 8

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

# Google Drive API imports
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
except ImportError:
    print("Please install Google API dependencies:")
    print("  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    sys.exit(1)


# ============================================================================
# Configuration - Modify these as needed
# ============================================================================

# Path to your Google Cloud service account credentials JSON file
DEFAULT_CREDENTIALS_PATH = "credentials/service_account.json"

# Default output directory for downloaded images
DEFAULT_OUTPUT_DIR = "/opt/SP/DATA/beauty_dataset"

# Default output CSV path
DEFAULT_OUTPUT_CSV = "data/dataset.csv"

# Number of parallel folder download threads
DEFAULT_FOLDER_WORKERS = 8

# Number of parallel file download threads per folder
DEFAULT_FILE_WORKERS = 4

# Image extensions to download
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Rate limiting: delay between API requests (seconds)
RATE_LIMIT_DELAY = 0.05

# Maximum retries for failed downloads
MAX_RETRIES = 3

# Progress save interval (save results every N folders)
PROGRESS_SAVE_INTERVAL = 50


# ============================================================================
# Thread-safe Progress Tracker
# ============================================================================


@dataclass
class ProgressTracker:
    """Thread-safe progress tracking."""

    total: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    total_images: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    results: list = field(default_factory=list)

    def mark_completed(self, folder_name: str, score: int, num_photos: int):
        with self._lock:
            self.completed += 1
            self.total_images += num_photos
            self.results.append(
                {
                    "folder_name": folder_name,
                    "score": score,
                    "num_photos": num_photos,
                }
            )

    def mark_skipped(self, folder_name: str, score: int, num_photos: int):
        with self._lock:
            self.skipped += 1
            self.total_images += num_photos
            self.results.append(
                {
                    "folder_name": folder_name,
                    "score": score,
                    "num_photos": num_photos,
                }
            )

    def mark_failed(self):
        with self._lock:
            self.failed += 1

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "completed": self.completed,
                "skipped": self.skipped,
                "failed": self.failed,
                "total_images": self.total_images,
            }

    def get_results(self) -> list:
        with self._lock:
            return self.results.copy()


# ============================================================================
# Google Drive API Client Pool
# ============================================================================


class GoogleDriveClientPool:
    """
    Pool of Google Drive clients for thread-safe parallel downloads.

    Each thread gets its own client instance since the Google API
    client is not thread-safe.
    """

    def __init__(self, credentials_path: str | None, pool_size: int = 8):
        self.credentials_path = credentials_path
        self.pool_size = pool_size
        self._local = threading.local()
        self._credentials = self._load_credentials()

    def _load_credentials(self):
        """Load credentials once (thread-safe, immutable)."""
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]

        if self.credentials_path and Path(self.credentials_path).exists():
            return service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=scopes
            )
        else:
            from google.auth import default

            credentials, _ = default(scopes=scopes)
            return credentials

    def get_client(self):
        """Get a thread-local client instance."""
        if not hasattr(self._local, "service"):
            self._local.service = build(
                "drive",
                "v3",
                credentials=self._credentials,
                cache_discovery=False,
            )
        return self._local.service


class GoogleDriveDownloader:
    """High-performance parallel downloader for Google Drive."""

    def __init__(
        self,
        credentials_path: str | None,
        folder_workers: int = DEFAULT_FOLDER_WORKERS,
        file_workers: int = DEFAULT_FILE_WORKERS,
    ):
        self.client_pool = GoogleDriveClientPool(credentials_path, folder_workers)
        self.folder_workers = folder_workers
        self.file_workers = file_workers

    def list_files_in_folder(self, folder_id: str) -> list[dict]:
        """List all image files in a Google Drive folder."""
        service = self.client_pool.get_client()
        files = []
        page_token = None
        query = f"'{folder_id}' in parents and trashed = false"

        try:
            while True:
                response = (
                    service.files()
                    .list(
                        q=query,
                        spaces="drive",
                        fields="nextPageToken, files(id, name, mimeType, size)",
                        pageToken=page_token,
                        pageSize=100,
                    )
                    .execute()
                )

                for file in response.get("files", []):
                    name = file.get("name", "").lower()
                    if any(name.endswith(ext) for ext in IMAGE_EXTENSIONS):
                        files.append(file)

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

                time.sleep(RATE_LIMIT_DELAY)

        except HttpError as e:
            if "404" in str(e) or "403" in str(e):
                pass  # Folder not found or no access - silent fail
            else:
                print(f"\nError listing folder {folder_id}: {e}")
            return []

        return files

    def download_file(self, file_id: str, output_path: Path) -> bool:
        """Download a single file with retries."""
        service = self.client_pool.get_client()

        for attempt in range(MAX_RETRIES):
            try:
                request = service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    _, done = downloader.next_chunk()

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    fh.seek(0)
                    f.write(fh.read())

                return True

            except HttpError as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2**attempt * 0.5)
                continue
            except Exception:
                return False

        return False

    def download_folder(
        self,
        folder_id: str,
        folder_name: str,
        output_dir: Path,
    ) -> list[str]:
        """Download all images from a folder using parallel file downloads."""
        folder_path = output_dir / folder_name
        folder_path.mkdir(parents=True, exist_ok=True)

        files = self.list_files_in_folder(folder_id)
        if not files:
            return []

        downloaded = []
        files_to_download = []

        # Check which files need downloading
        for file_info in files:
            file_name = file_info["name"]
            output_path = folder_path / file_name

            if output_path.exists():
                downloaded.append(str(output_path))
            else:
                files_to_download.append((file_info["id"], output_path))

        # Download missing files in parallel
        if files_to_download:
            with ThreadPoolExecutor(max_workers=self.file_workers) as executor:
                futures = {
                    executor.submit(self.download_file, fid, path): (fid, path)
                    for fid, path in files_to_download
                }

                for future in as_completed(futures):
                    fid, path = futures[future]
                    try:
                        if future.result():
                            downloaded.append(str(path))
                    except Exception:
                        pass

        return downloaded


# ============================================================================
# Dataset Preparation with Parallel Downloads
# ============================================================================


def load_source_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate the source CSV file."""
    df = pd.read_csv(csv_path)
    df = df[~df["score"].isna()]

    required_cols = ["folder_name", "folder_id", "score"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required_cols].drop_duplicates().reset_index(drop=True)

    if df["score"].min() < 1 or df["score"].max() > 9:
        print(
            f"Warning: Scores outside 1-9 range: "
            f"min={df['score'].min()}, max={df['score'].max()}"
        )

    return df


def process_folder(
    downloader: GoogleDriveDownloader,
    folder_name: str,
    folder_id: str,
    score: int,
    output_dir: Path,
    skip_existing: bool,
    tracker: ProgressTracker,
) -> None:
    """Process a single folder (called by thread pool)."""
    folder_path = output_dir / folder_name

    # Check if already downloaded
    if skip_existing and folder_path.exists():
        existing_images = (
            list(folder_path.glob("*.jpg"))
            + list(folder_path.glob("*.jpeg"))
            + list(folder_path.glob("*.png"))
            + list(folder_path.glob("*.webp"))
        )
        if existing_images:
            tracker.mark_skipped(folder_name, score, len(existing_images))
            return

    # Download images
    downloaded = downloader.download_folder(folder_id, folder_name, output_dir)

    if downloaded:
        tracker.mark_completed(folder_name, score, len(downloaded))
    else:
        tracker.mark_failed()


def save_progress(tracker: ProgressTracker, output_csv: Path):
    """Save current progress to CSV."""
    results = tracker.get_results()
    if results:
        df = pd.DataFrame(results)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df[["folder_name", "score"]].to_csv(output_csv, index=False)


def prepare_dataset(
    source_csv: Path,
    output_dir: Path,
    output_csv: Path,
    credentials_path: str | None = None,
    folder_workers: int = DEFAULT_FOLDER_WORKERS,
    file_workers: int = DEFAULT_FILE_WORKERS,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Download images and prepare the dataset using parallel downloads.

    Args:
        source_csv: Path to source CSV with folder_id and scores.
        output_dir: Directory to save downloaded images.
        output_csv: Path for the output CSV file.
        credentials_path: Path to Google credentials JSON.
        folder_workers: Number of parallel folder download threads.
        file_workers: Number of parallel file download threads per folder.
        skip_existing: Skip folders that already have images.

    Returns:
        DataFrame with the prepared dataset.
    """
    print(f"Loading source CSV: {source_csv}")
    df = load_source_csv(source_csv)
    print(f"Found {len(df)} unique folders to process")
    print(f"Using {folder_workers} folder workers, {file_workers} file workers per folder")

    # Initialize downloader and progress tracker
    print("Initializing Google Drive client pool...")
    downloader = GoogleDriveDownloader(
        credentials_path,
        folder_workers=folder_workers,
        file_workers=file_workers,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = ProgressTracker(total=len(df))

    # Prepare work items
    work_items = [
        (str(row["folder_name"]), str(row["folder_id"]), int(row["score"]))
        for _, row in df.iterrows()
    ]

    # Process folders in parallel with progress bar
    print(f"\nDownloading {len(work_items)} folders...")

    with ThreadPoolExecutor(max_workers=folder_workers) as executor:
        futures = {
            executor.submit(
                process_folder,
                downloader,
                folder_name,
                folder_id,
                score,
                output_dir,
                skip_existing,
                tracker,
            ): folder_name
            for folder_name, folder_id, score in work_items
        }

        with tqdm(total=len(futures), desc="Downloading", unit="folder") as pbar:
            completed_count = 0
            for future in as_completed(futures):
                folder_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"\nError processing {folder_name}: {e}")
                    tracker.mark_failed()

                completed_count += 1
                stats = tracker.get_stats()
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "done": stats["completed"],
                        "skip": stats["skipped"],
                        "fail": stats["failed"],
                        "imgs": stats["total_images"],
                    }
                )

                # Periodic progress save
                if completed_count % PROGRESS_SAVE_INTERVAL == 0:
                    save_progress(tracker, output_csv)

    # Final save
    results = tracker.get_results()
    result_df = pd.DataFrame(results)

    if len(result_df) == 0:
        print("Warning: No images were downloaded!")
        return result_df

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df[["folder_name", "score"]].to_csv(output_csv, index=False)
    print(f"\nSaved dataset CSV to: {output_csv}")

    # Print summary
    stats = tracker.get_stats()
    print("\n" + "=" * 60)
    print("Dataset Preparation Summary")
    print("=" * 60)
    print(f"Total folders processed: {len(df)}")
    print(f"Successfully downloaded: {stats['completed']}")
    print(f"Skipped (existing):      {stats['skipped']}")
    print(f"Failed:                  {stats['failed']}")
    print(f"Total images:            {stats['total_images']}")
    print(f"\nScore distribution:")
    if len(result_df) > 0:
        print(result_df.groupby("score")["num_photos"].agg(["count", "sum"]).to_string())
    print("=" * 60)

    return result_df


# ============================================================================
# Alternative: Parallel download using gdown (public folders)
# ============================================================================


def download_public_folder_gdown(
    folder_id: str,
    folder_name: str,
    output_dir: Path,
) -> list[str]:
    """Download a public Google Drive folder using gdown."""
    try:
        import gdown
    except ImportError:
        return []

    folder_path = output_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{folder_id}"

    try:
        gdown.download_folder(
            url=url,
            output=str(folder_path),
            quiet=True,
            use_cookies=False,
        )
    except Exception:
        return []

    downloaded = []
    for ext in IMAGE_EXTENSIONS:
        downloaded.extend([str(p) for p in folder_path.glob(f"*{ext}")])

    return downloaded


def process_folder_public(
    folder_name: str,
    folder_id: str,
    score: int,
    output_dir: Path,
    skip_existing: bool,
    tracker: ProgressTracker,
) -> None:
    """Process a single folder using gdown (called by thread pool)."""
    folder_path = output_dir / folder_name

    if skip_existing and folder_path.exists():
        existing = (
            list(folder_path.glob("*.jpg"))
            + list(folder_path.glob("*.jpeg"))
            + list(folder_path.glob("*.png"))
        )
        if existing:
            tracker.mark_skipped(folder_name, score, len(existing))
            return

    downloaded = download_public_folder_gdown(folder_id, folder_name, output_dir)

    if downloaded:
        tracker.mark_completed(folder_name, score, len(downloaded))
    else:
        tracker.mark_failed()


def prepare_dataset_public(
    source_csv: Path,
    output_dir: Path,
    output_csv: Path,
    num_workers: int = DEFAULT_FOLDER_WORKERS,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Prepare dataset from public Google Drive folders using gdown with parallelism.

    Args:
        source_csv: Path to source CSV.
        output_dir: Directory to save images.
        output_csv: Path for output CSV.
        num_workers: Number of parallel download workers.
        skip_existing: Skip folders that already have images.

    Returns:
        DataFrame with prepared dataset.
    """
    try:
        import gdown
    except ImportError:
        print("Please install gdown: pip install gdown")
        sys.exit(1)

    print(f"Loading source CSV: {source_csv}")
    df = load_source_csv(source_csv)
    print(f"Found {len(df)} folders to process")
    print(f"Using {num_workers} parallel workers")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker = ProgressTracker(total=len(df))

    work_items = [
        (str(row["folder_name"]), str(row["folder_id"]), int(row["score"]))
        for _, row in df.iterrows()
    ]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_folder_public,
                folder_name,
                folder_id,
                score,
                output_dir,
                skip_existing,
                tracker,
            ): folder_name
            for folder_name, folder_id, score in work_items
        }

        with tqdm(total=len(futures), desc="Downloading", unit="folder") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    tracker.mark_failed()

                stats = tracker.get_stats()
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "done": stats["completed"],
                        "skip": stats["skipped"],
                        "fail": stats["failed"],
                    }
                )

    results = tracker.get_results()
    result_df = pd.DataFrame(results)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if len(result_df) > 0:
        result_df[["folder_name", "score"]].to_csv(output_csv, index=False)

    stats = tracker.get_stats()
    print(f"\nSaved dataset CSV to: {output_csv}")
    print(f"Total samples: {len(result_df)}")
    print(f"Total images: {stats['total_images']}")

    return result_df


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download images from Google Drive and prepare training dataset (PARALLEL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast parallel download with Google API
  python scripts/prepare_dataset_from_gdrive.py \\
      --input data/source.csv \\
      --output-dir data/images \\
      --folder-workers 16 \\
      --file-workers 8

  # Using public links (no credentials needed)
  python scripts/prepare_dataset_from_gdrive.py \\
      --input data/source.csv \\
      --public \\
      --folder-workers 8

CSV Format:
  folder_name,folder_id,score,photo_0.jpg,photo_1.jpg,...
  person_001,1ABC...xyz,7,img1.jpg,img2.jpg,...
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-csv",
        "-c",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path for output CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS_PATH,
        help=f"Path to Google credentials JSON (default: {DEFAULT_CREDENTIALS_PATH})",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Use gdown for public folders (no credentials needed)",
    )
    parser.add_argument(
        "--folder-workers",
        type=int,
        default=DEFAULT_FOLDER_WORKERS,
        help=f"Parallel folder download threads (default: {DEFAULT_FOLDER_WORKERS})",
    )
    parser.add_argument(
        "--file-workers",
        type=int,
        default=DEFAULT_FILE_WORKERS,
        help=f"Parallel file download threads per folder (default: {DEFAULT_FILE_WORKERS})",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download existing folders",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input CSV not found: {args.input}")
        sys.exit(1)

    if args.public:
        prepare_dataset_public(
            source_csv=args.input,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            num_workers=args.folder_workers,
            skip_existing=not args.no_skip_existing,
        )
    else:
        if not Path(args.credentials).exists():
            print(f"Warning: Credentials not found: {args.credentials}")
            print("Trying Application Default Credentials...")

        prepare_dataset(
            source_csv=args.input,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            credentials_path=str(args.credentials),
            folder_workers=args.folder_workers,
            file_workers=args.file_workers,
            skip_existing=not args.no_skip_existing,
        )


if __name__ == "__main__":
    main()
