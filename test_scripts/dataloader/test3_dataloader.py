# test_dataloader.py
import torch
import numpy as np
from data.loader import create_dataloaders

# 더미 데이터
n_train = 100
n_val = 30
n_channels = 12
n_timesteps = 50

X_train = np.random.randn(n_train, n_channels, n_timesteps).astype(np.float32)
y_train = np.random.randint(0, 2, size=n_train)
X_val = np.random.randn(n_val, n_channels, n_timesteps).astype(np.float32)
y_val = np.random.randint(0, 2, size=n_val)

# DataLoader 생성
train_loader, val_loader = create_dataloaders(
    X_train, y_train,
    X_val, y_val,
    batch_size=16,
    seed=42,
    num_workers=0  # 테스트할 땐 0으로 (디버깅 쉬움)
)

print(f"=== DataLoader Info ===")
print(f"Train loader batches: {len(train_loader)}")
print(f"Val loader batches: {len(val_loader)}")

# 첫 배치 확인
print(f"\n=== First Batch ===")
for batch_x, batch_y in train_loader:
    print(f"Batch X shape: {batch_x.shape}")
    print(f"Batch y shape: {batch_y.shape}")
    print(f"Batch X dtype: {batch_x.dtype}")
    print(f"Batch y dtype: {batch_y.dtype}")
    print(f"Batch y values: {batch_y[:5]}")
    break

print("\n✅ DataLoader test passed!")