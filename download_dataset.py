"""
Download SynthBuster dataset from Kaggle using API token from .env
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_kaggle_credentials():
    """Set up Kaggle API credentials from environment variable"""
    api_token = os.getenv('KAGGLE_API_TOKEN')
    
    if not api_token:
        raise ValueError("KAGGLE_API_TOKEN not found in .env file")
    
    # Kaggle API expects credentials in ~/.kaggle/kaggle.json
    # But we can also set environment variables
    os.environ['KAGGLE_KEY'] = api_token
    os.environ['KAGGLE_USERNAME'] = 'your_username'  # Update if needed
    
    print(f"✓ Kaggle credentials configured")

def download_synthbuster_dataset(output_dir='data'):
    """Download the SynthBuster dataset"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        import zipfile
        
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Download dataset (without auto-unzip)
        dataset_name = 'erengencturk/synthbuster'
        print(f"Downloading {dataset_name}...")
        
        api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=False  # We'll unzip manually with progress
        )
        
        print(f"✓ Dataset downloaded")
        
        # Manual unzip with progress
        zip_path = Path(output_dir) / 'synthbuster.zip'
        if zip_path.exists():
            print(f"\nUnzipping {zip_path.name}...")
            print("This may take a few minutes for large datasets...")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                total = len(members)
                
                for i, member in enumerate(members, 1):
                    zip_ref.extract(member, output_dir)
                    if i % 100 == 0 or i == total:
                        print(f"  Extracted {i}/{total} files ({i*100//total}%)")
            
            print(f"✓ Extraction complete")
            
            # Optionally remove zip file
            # zip_path.unlink()
            # print(f"✓ Removed {zip_path.name}")
        
        # List downloaded files
        data_path = Path(output_dir)
        files = [f for f in data_path.rglob('*') if f.is_file() and f.suffix != '.zip']
        print(f"\n✓ Dataset ready: {len(files)} files in {output_dir}/")
        
        # Show directory structure
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
        if subdirs:
            print("\nDirectory structure:")
            for d in sorted(subdirs):
                file_count = len(list(d.rglob('*')))
                print(f"  - {d.name}/ ({file_count} items)")
            
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        raise

if __name__ == '__main__':
    setup_kaggle_credentials()
    download_synthbuster_dataset()
