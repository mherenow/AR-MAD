"""
Download and extract ONLY train2017 folder from COCO 2017 dataset
"""
import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

def setup_kaggle_credentials():
    api_token = os.getenv('KAGGLE_API_TOKEN')
    username = os.getenv('KAGGLE_USERNAME')

    if not api_token or not username:
        raise ValueError("KAGGLE_API_TOKEN or KAGGLE_USERNAME missing in .env")

    os.environ['KAGGLE_KEY'] = api_token
    os.environ['KAGGLE_USERNAME'] = username

    print("✓ Kaggle credentials configured")


def download_dataset(output_dir='data', force_redownload=False):
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    Path(output_dir).mkdir(exist_ok=True)
    zip_path = Path(output_dir) / "coco-2017-dataset.zip"

    if zip_path.exists() and not force_redownload:
        print("✓ Using existing dataset zip")
        return zip_path

    print("\nDownloading COCO 2017 dataset (this is large ~25GB)...")
    print("Note: Kaggle API doesn't show download progress, please wait...")

    api.dataset_download_files(
        "awsaf49/coco-2017-dataset",
        path=output_dir,
        unzip=False
    )

    print("✓ Download complete")
    return zip_path


def extract_train2017(zip_path, output_dir='data'):
    print("\nExtracting ONLY train2017 folder...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()

            train_files = [
                m for m in members
                if "train2017/" in m
            ]

            print(f"Found {len(train_files)} train2017 files")

            for member in tqdm(train_files, desc="Extracting", unit="file"):
                zip_ref.extract(member, output_dir)

        print("✓ train2017 extracted successfully")

    except Exception as e:
        print(f"✗ Extraction error: {e}")
        raise


if __name__ == '__main__':
    import sys

    force = '--force' in sys.argv

    setup_kaggle_credentials()

    zip_path = download_dataset(force_redownload=force)

    extract_train2017(zip_path)

    print("\n✓ Done. Only train2017 is available.")