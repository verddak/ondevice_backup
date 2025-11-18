# scripts/check_data_leakage.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from data.loader import load_hbm_data

def check_data_leakage():
    """Check if train/val data overlap"""
    DATA_DIR = "/data3/hai_ts/on-device/data/hbm_t30a100"
    X_train, y_train, X_val, y_val = load_hbm_data(DATA_DIR)
    
    print("="*60)
    print("Data Leakage Check")
    print("="*60)
    
    # 1. Shape 확인
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    
    # 2. 중복 샘플 확인
    print(f"\n1. Checking for duplicate samples...")
    
    # Train 내부 중복
    train_unique = len(np.unique(X_train.reshape(len(X_train), -1), axis=0))
    print(f"   Train unique samples: {train_unique}/{len(X_train)}")
    if train_unique < len(X_train):
        print(f"   ⚠️  {len(X_train) - train_unique} duplicate samples in train!")
    
    # Val 내부 중복
    val_unique = len(np.unique(X_val.reshape(len(X_val), -1), axis=0))
    print(f"   Val unique samples: {val_unique}/{len(X_val)}")
    if val_unique < len(X_val):
        print(f"   ⚠️  {len(X_val) - val_unique} duplicate samples in val!")
    
    # Train-Val 중복 (가장 중요!)
    print(f"\n2. Checking train-val overlap...")
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    
    overlap_count = 0
    for i, val_sample in enumerate(X_val_flat):
        if np.any(np.all(X_train_flat == val_sample, axis=1)):
            overlap_count += 1
    
    print(f"   Overlapping samples: {overlap_count}/{len(X_val)}")
    if overlap_count > 0:
        print(f"   ❌ DATA LEAKAGE DETECTED! {overlap_count} val samples are in train!")
    else:
        print(f"   ✅ No overlap between train and val")
    
    # 3. 레이블 분포 확인
    print(f"\n3. Label distribution:")
    train_dist = {label: np.sum(y_train == label) for label in np.unique(y_train)}
    val_dist = {label: np.sum(y_val == label) for label in np.unique(y_val)}
    
    print(f"   Train: {train_dist}")
    print(f"   Val: {val_dist}")
    
    # 비율 확인
    for label in np.unique(y_train):
        train_ratio = train_dist[label] / len(y_train) * 100
        val_ratio = val_dist[label] / len(y_val) * 100
        print(f"   Class {label}: Train {train_ratio:.1f}%, Val {val_ratio:.1f}%")
        
        if abs(train_ratio - val_ratio) > 20:
            print(f"   ⚠️  Large distribution difference for class {label}!")
    
    # 4. 통계량 확인
    print(f"\n4. Statistical properties:")
    print(f"   Train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"   Val mean: {X_val.mean():.4f}, std: {X_val.std():.4f}")
    print(f"   Train range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"   Val range: [{X_val.min():.4f}, {X_val.max():.4f}]")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    check_data_leakage()