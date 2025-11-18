# test_dataset.py
import torch
import numpy as np
from data.dataset import TimeSeriesDataset

# 더미 데이터 생성
n_samples = 100
n_channels = 12
n_timesteps = 50
n_classes = 2

X_dummy = np.random.randn(n_samples, n_channels, n_timesteps).astype(np.float32)
y_dummy = np.random.randint(0, n_classes, size=n_samples)

print(f"X shape: {X_dummy.shape}")
print(f"y shape: {y_dummy.shape}")

# Dataset 생성
dataset = TimeSeriesDataset(X_dummy, y_dummy)

print(f"\n=== Dataset Info ===")
print(f"Dataset length: {len(dataset)}")
print(f"First sample shape: {dataset[0][0].shape}")
print(f"First label: {dataset[0][1]}")
print(f"First sample type: {type(dataset[0][0])}")
print(f"First label type: {type(dataset[0][1])}")

# 여러 샘플 확인
print(f"\n=== Multiple Samples ===")
for i in range(3):
    x, y = dataset[i]
    print(f"Sample {i}: x.shape={x.shape}, y={y.item()}, dtype={x.dtype}")

print("\n✅ Dataset test passed!")