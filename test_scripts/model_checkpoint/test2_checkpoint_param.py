# tests/test_model_params.py
import torch
from pathlib import Path
import sys

def main():
    checkpoint_path = Path(sys.argv[1])
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())
    expected_size_mb = total_params * 4 / (1024**2)
    actual_size_mb = checkpoint_path.stat().st_size / (1024**2)
    
    print(f"\n📊 Parameters: {total_params:,}")
    print(f"   Expected size: {expected_size_mb:.2f} MB")
    print(f"   Actual size: {actual_size_mb:.2f} MB")
    
    # Check if too small
    if actual_size_mb < expected_size_mb * 0.5:
        print(f"\n⚠️  WARNING: File size too small!")
        sys.exit(1)

if __name__ == "__main__":
    main()