# test_scripts/data/test_load_npy.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from data.loader import load_hbm_data, create_dataloaders

def test_load_npy():
    print("=== Testing .npy Data Loading ===")
    
    # 데이터 로드
    data_dir = '/data3/hai_ts/on-device/data/2mm_#1_seed42'
    X_train, y_train, X_val, y_val = load_hbm_data(data_dir)
    
    # 기본 정보
    print(f"\n=== Data Info ===")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Input shape: {X_train.shape}")
    print(f"Unique classes: {np.unique(y_train)}")
    
    # DataLoader 생성
    train_loader, val_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        seed=42,
        num_workers=0  # 테스트할 땐 0
    )
    
    print(f"\n=== DataLoader Info ===")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # 첫 배치 확인
    print(f"\n=== First Train Batch ===")
    for batch_x, batch_y in train_loader:
        print(f"Batch X: {batch_x.shape}, dtype: {batch_x.dtype}")
        print(f"Batch y: {batch_y.shape}, dtype: {batch_y.dtype}")
        print(f"X range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
        print(f"y values: {batch_y[:10]}")
        break
    
    print(f"\n=== First Val Batch ===")
    for batch_x, batch_y in val_loader:
        print(f"Batch X: {batch_x.shape}, dtype: {batch_x.dtype}")
        print(f"Batch y: {batch_y.shape}, dtype: {batch_y.dtype}")
        break
    
    print("\n✅ .npy loading test passed!")

if __name__ == "__main__":
    test_load_npy()