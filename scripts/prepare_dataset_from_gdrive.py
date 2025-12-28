#!/usr/bin/env python3
"""
Download images from Google Drive folders and prepare dataset for training.

This script:
1. Reads a CSV with folder_name, folder_id, score, and photo columns
2. Downloads all photos from each Google Drive folder
3. Creates a local folder structure compatible with BeautyDataset
4. Generates a training-ready CSV file

Usage:
    python scripts/prepare_dataset_from_gdrive.py \
        --input data/source.csv \
        --output-dir data/images \
        --output-csv data/dataset.csv

Requirements:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

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
# Download from: https://console.cloud.google.com/iam-admin/serviceaccounts
DEFAULT_CREDENTIALS_PATH = "credentials/service_account.json"

# Default output directory for downloaded images
DEFAULT_OUTPUT_DIR = "/opt/SP/DATA/beauty_dataset"

# Default output CSV path
DEFAULT_OUTPUT_CSV = "data/dataset.csv"

# Number of parallel download threads
DEFAULT_NUM_WORKERS = 4

# Image extensions to download
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Rate limiting: delay between requests (seconds)
RATE_LIMIT_DELAY = 0.1

# Maximum retries for failed downloads
MAX_RETRIES = 3


# ============================================================================
# Google Drive API Client
# ============================================================================


class GoogleDriveClient:
    """Client for downloading files from Google Drive."""

    def __init__(self, credentials_path: str | None = None):
        """
        Initialize the Google Drive client.

        Args:
            credentials_path: Path to service account JSON credentials.
                            If None, uses Application Default Credentials.
        """
        self.service = self._build_service(credentials_path)

    def _build_service(self, credentials_path: str | None):
        """Build the Drive API service."""
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]

        if credentials_path and Path(credentials_path).exists():
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=scopes
            )
        else:
            # Try Application Default Credentials
            from google.auth import default

            credentials, _ = default(scopes=scopes)

        return build("drive", "v3", credentials=credentials)

    def list_files_in_folder(self, folder_id: str) -> list[dict]:
        """
        List all image files in a Google Drive folder.

        Args:
            folder_id: The Google Drive folder ID.

        Returns:
            List of file metadata dictionaries.
        """
        files = []
        page_token = None

        # Build query for image files
        query = f"'{folder_id}' in parents and trashed = false"

        try:
            while True:
                response = (
                    self.service.files()
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
                    # Filter by image extensions
                    name = file.get("name", "").lower()
                    if any(name.endswith(ext) for ext in IMAGE_EXTENSIONS):
                        files.append(file)

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

                time.sleep(RATE_LIMIT_DELAY)

        except HttpError as e:
            print(f"Error listing files in folder {folder_id}: {e}")
            return []

        return files

    def download_file(
        self,
        file_id: str,
        output_path: Path,
        retries: int = MAX_RETRIES,
    ) -> bool:
        """
        Download a file from Google Drive.

        Args:
            file_id: The Google Drive file ID.
            output_path: Local path to save the file.
            retries: Number of retry attempts.

        Returns:
            True if download succeeded, False otherwise.
        """
        for attempt in range(retries):
            try:
                request = self.service.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    _, done = downloader.next_chunk()

                # Write to file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "wb") as f:
                    fh.seek(0)
                    f.write(fh.read())

                return True

            except HttpError as e:
                if attempt < retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    print(f"Failed to download file {file_id}: {e}")
                    return False
            except Exception as e:
                print(f"Unexpected error downloading {file_id}: {e}")
                return False

        return False


# ============================================================================
# Dataset Preparation
# ============================================================================


def load_source_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load the source CSV file.

    Expected columns:
    - folder_name: Name/identifier for the person
    - folder_id: Google Drive folder ID
    - score: Beauty score (1-9)
    - photo_0.jpg, photo_1.jpg, ...: (ignored, we download all from folder)

    Args:
        csv_path: Path to the source CSV.

    Returns:
        DataFrame with folder_name, folder_id, and score columns.
    """
    df = pd.read_csv(csv_path)
    df = df[~df["score"].isna()]

    # Validate required columns
    required_cols = ["folder_name", "folder_id", "score"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only required columns and drop duplicates
    df = df[required_cols].drop_duplicates().reset_index(drop=True)

    # Validate scores
    if df["score"].min() < 1 or df["score"].max() > 9:
        print(
            f"Warning: Scores outside 1-9 range detected: "
            f"min={df['score'].min()}, max={df['score'].max()}"
        )

    return df


def download_folder_images(
    client: GoogleDriveClient,
    folder_id: str,
    folder_name: str,
    output_dir: Path,
) -> list[str]:
    """
    Download all images from a Google Drive folder.

    Args:
        client: Google Drive client.
        folder_id: Google Drive folder ID.
        folder_name: Local folder name to use.
        output_dir: Base output directory.

    Returns:
        List of downloaded file paths.
    """
    folder_path = output_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # List files in the folder
    files = client.list_files_in_folder(folder_id)
    if not files:
        return []

    downloaded = []
    for file_info in files:
        file_name = file_info["name"]
        file_id = file_info["id"]
        output_path = folder_path / file_name

        # Skip if already exists
        if output_path.exists():
            downloaded.append(str(output_path))
            continue

        if client.download_file(file_id, output_path):
            downloaded.append(str(output_path))

        time.sleep(RATE_LIMIT_DELAY)

    return downloaded


def prepare_dataset(
    source_csv: Path,
    output_dir: Path,
    output_csv: Path,
    credentials_path: str | None = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    skip_existing: bool = True,
) -> pd.DataFrame:
    """
    Download images and prepare the dataset.

    Args:
        source_csv: Path to source CSV with folder_id and scores.
        output_dir: Directory to save downloaded images.
        output_csv: Path for the output CSV file.
        credentials_path: Path to Google credentials JSON.
        num_workers: Number of parallel download workers.
        skip_existing: Skip folders that already have images.

    Returns:
        DataFrame with the prepared dataset.
    """
    print(f"Loading source CSV: {source_csv}")
    df = load_source_csv(source_csv)
    print(f"Found {len(df)} unique folders to process")

    # Initialize Google Drive client
    print("Initializing Google Drive client...")
    client = GoogleDriveClient(credentials_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    skipped = 0
    failed = 0

    # Process each folder
    with tqdm(total=len(df), desc="Downloading folders") as pbar:
        for _, row in df.iterrows():
            folder_name = str(row["folder_name"])
            folder_id = str(row["folder_id"])
            score = int(row["score"])

            folder_path = output_dir / folder_name

            # Check if already downloaded
            if skip_existing and folder_path.exists():
                existing_images = (
                    list(folder_path.glob("*.jpg"))
                    + list(folder_path.glob("*.jpeg"))
                    + list(folder_path.glob("*.png"))
                )
                if existing_images:
                    results.append(
                        {
                            "folder_name": folder_name,
                            "score": score,
                            "num_photos": len(existing_images),
                        }
                    )
                    skipped += 1
                    pbar.update(1)
                    pbar.set_postfix({"skipped": skipped, "failed": failed})
                    continue

            # Download images
            downloaded = download_folder_images(client, folder_id, folder_name, output_dir)

            if downloaded:
                results.append(
                    {
                        "folder_name": folder_name,
                        "score": score,
                        "num_photos": len(downloaded),
                    }
                )
            else:
                failed += 1
                print(f"\nNo images downloaded for {folder_name} ({folder_id})")

            pbar.update(1)
            pbar.set_postfix({"skipped": skipped, "failed": failed})

    # Create output dataframe
    result_df = pd.DataFrame(results)

    if len(result_df) == 0:
        print("Warning: No images were downloaded!")
        return result_df

    # Save the dataset CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df[["folder_name", "score"]].to_csv(output_csv, index=False)
    print(f"\nSaved dataset CSV to: {output_csv}")

    # Print summary
    print("\n" + "=" * 50)
    print("Dataset Preparation Summary")
    print("=" * 50)
    print(f"Total folders processed: {len(df)}")
    print(f"Successfully downloaded: {len(result_df)}")
    print(f"Skipped (existing):      {skipped}")
    print(f"Failed:                  {failed}")
    print(f"Total images:            {result_df['num_photos'].sum()}")
    print(f"\nScore distribution:")
    print(result_df.groupby("score")["num_photos"].agg(["count", "sum"]))
    print("=" * 50)

    return result_df


# ============================================================================
# Alternative: Download using public links (no API credentials needed)
# ============================================================================


def download_public_folder_gdown(
    folder_id: str,
    folder_name: str,
    output_dir: Path,
) -> list[str]:
    """
    Download a public Google Drive folder using gdown.

    This method works for publicly shared folders without API credentials.

    Args:
        folder_id: Google Drive folder ID.
        folder_name: Local folder name.
        output_dir: Base output directory.

    Returns:
        List of downloaded file paths.

    Note:
        Requires: pip install gdown
    """
    try:
        import gdown
    except ImportError:
        print("Please install gdown: pip install gdown")
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
    except Exception as e:
        print(f"Error downloading folder {folder_id}: {e}")
        return []

    # List downloaded files
    downloaded = []
    for ext in IMAGE_EXTENSIONS:
        downloaded.extend([str(p) for p in folder_path.glob(f"*{ext}")])

    return downloaded


def prepare_dataset_public(
    source_csv: Path,
    output_dir: Path,
    output_csv: Path,
) -> pd.DataFrame:
    """
    Prepare dataset from publicly shared Google Drive folders using gdown.

    This is a simpler alternative that doesn't require API credentials,
    but only works for publicly shared folders.

    Args:
        source_csv: Path to source CSV.
        output_dir: Directory to save images.
        output_csv: Path for output CSV.

    Returns:
        DataFrame with prepared dataset.
    """
    print(f"Loading source CSV: {source_csv}")
    df = load_source_csv(source_csv)
    print(f"Found {len(df)} folders to process")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        folder_name = str(row["folder_name"])
        folder_id = str(row["folder_id"])
        score = int(row["score"])

        folder_path = output_dir / folder_name

        # Skip if already exists with images
        if folder_path.exists():
            existing = (
                list(folder_path.glob("*.jpg"))
                + list(folder_path.glob("*.jpeg"))
                + list(folder_path.glob("*.png"))
            )
            if existing:
                results.append(
                    {
                        "folder_name": folder_name,
                        "score": score,
                        "num_photos": len(existing),
                    }
                )
                continue

        downloaded = download_public_folder_gdown(folder_id, folder_name, output_dir)

        if downloaded:
            results.append(
                {
                    "folder_name": folder_name,
                    "score": score,
                    "num_photos": len(downloaded),
                }
            )

    result_df = pd.DataFrame(results)

    # Save CSV
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df[["folder_name", "score"]].to_csv(output_csv, index=False)

    print(f"\nSaved dataset CSV to: {output_csv}")
    print(f"Total samples: {len(result_df)}")
    print(f"Total images: {result_df['num_photos'].sum() if len(result_df) > 0 else 0}")

    return result_df


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Download images from Google Drive and prepare training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using Google API (requires credentials)
  python scripts/prepare_dataset_from_gdrive.py \\
      --input data/source.csv \\
      --output-dir data/images \\
      --output-csv data/dataset.csv \\
      --credentials credentials/service_account.json

  # Using public links (no credentials needed, but folders must be public)
  python scripts/prepare_dataset_from_gdrive.py \\
      --input data/source.csv \\
      --output-dir data/images \\
      --output-csv data/dataset.csv \\
      --public

CSV Format:
  folder_name,folder_id,score,photo_0.jpg,photo_1.jpg,...
  person_001,1ABC...xyz,7,img1.jpg,img2.jpg,...
  person_002,2DEF...abc,5,img1.jpg,img2.jpg,...
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Path to input CSV file with folder_id and scores",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save downloaded images (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--output-csv",
        "-c",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Path for output dataset CSV (default: {DEFAULT_OUTPUT_CSV})",
    )
    parser.add_argument(
        "--credentials",
        type=Path,
        default=DEFAULT_CREDENTIALS_PATH,
        help=f"Path to Google service account credentials JSON (default: {DEFAULT_CREDENTIALS_PATH})",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Use gdown for public folders (no credentials needed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel download workers (default: {DEFAULT_NUM_WORKERS})",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download folders that already have images",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input CSV not found: {args.input}")
        sys.exit(1)

    if args.public:
        # Use gdown for public folders
        prepare_dataset_public(
            source_csv=args.input,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
        )
    else:
        # Use Google API
        if not Path(args.credentials).exists():
            print(f"Warning: Credentials file not found: {args.credentials}")
            print("Trying Application Default Credentials...")

        prepare_dataset(
            source_csv=args.input,
            output_dir=args.output_dir,
            output_csv=args.output_csv,
            credentials_path=str(args.credentials),
            num_workers=args.workers,
            skip_existing=not args.no_skip_existing,
        )


if __name__ == "__main__":
    main()
