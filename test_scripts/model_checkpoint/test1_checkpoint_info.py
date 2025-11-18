# tests/test_checkpoint_info.py
import torch
from pathlib import Path
import sys

def main():
    checkpoint_path = Path(sys.argv[1])
    
    print(f"\n📦 Checkpoint: {checkpoint_path.name}")
    print(f"   Size: {checkpoint_path.stat().st_size / (1024**2):.2f} MB")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\n📋 Keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  - {key}: {len(checkpoint[key])} items")
        else:
            print(f"  - {key}: {checkpoint[key]}")

if __name__ == "__main__":
    main()