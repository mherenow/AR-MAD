"""
Manually unzip the SynthBuster dataset with progress indicator
"""
import zipfile
from pathlib import Path

def unzip_with_progress(zip_path, output_dir):
    """Unzip file with progress updates"""
    print(f"Unzipping {zip_path}...")
    print("This may take several minutes for large datasets...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        total = len(members)
        print(f"Total files to extract: {total}")
        
        for i, member in enumerate(members, 1):
            zip_ref.extract(member, output_dir)
            if i % 100 == 0 or i == total:
                print(f"  Progress: {i}/{total} files ({i*100//total}%)")
    
    print(f"✓ Extraction complete!")
    
    # Show what was extracted
    data_path = Path(output_dir)
    files = [f for f in data_path.rglob('*') if f.is_file() and f.suffix != '.zip']
    print(f"\nExtracted {len(files)} files")
    
    subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    if subdirs:
        print("\nDirectory structure:")
        for d in sorted(subdirs):
            file_count = len([f for f in d.rglob('*') if f.is_file()])
            print(f"  - {d.name}/ ({file_count} files)")

if __name__ == '__main__':
    zip_file = Path('data/synthbuster.zip')
    
    if not zip_file.exists():
        print(f"✗ Error: {zip_file} not found")
        exit(1)
    
    unzip_with_progress(zip_file, 'data')
