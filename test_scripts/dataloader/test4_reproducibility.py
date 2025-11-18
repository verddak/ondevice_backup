# test_reproducibility.py
import torch
import numpy as np
from data.loader import create_dataloaders

# 더미 데이터
X_train = np.random.randn(100, 12, 50).astype(np.float32)
y_train = np.random.randint(0, 2, size=100)
X_val = np.random.randn(30, 12, 50).astype(np.float32)
y_val = np.random.randint(0, 2, size=30)

print("=== Reproducibility Test ===")

# 같은 seed로 두 번 생성
train_loader1, _ = create_dataloaders(
    X_train, y_train, X_val, y_val,
    batch_size=16, seed=42, num_workers=0
)

train_loader2, _ = create_dataloaders(
    X_train, y_train, X_val, y_val,
    batch_size=16, seed=42, num_workers=0
)

# 첫 배치 비교
batch1_x, batch1_y = next(iter(train_loader1))
batch2_x, batch2_y = next(iter(train_loader2))

print(f"Same seed - X equal? {torch.equal(batch1_x, batch2_x)}")
print(f"Same seed - y equal? {torch.equal(batch1_y, batch2_y)}")

# 다른 seed
train_loader3, _ = create_dataloaders(
    X_train, y_train, X_val, y_val,
    batch_size=16, seed=999, num_workers=0
)

batch3_x, batch3_y = next(iter(train_loader3))
print(f"Different seed - X equal? {torch.equal(batch1_x, batch3_x)}")
print(f"Different seed - y equal? {torch.equal(batch1_y, batch3_y)}")

if torch.equal(batch1_x, batch2_x) and not torch.equal(batch1_x, batch3_x):
    print("\n✅ Reproducibility test passed!")
else:
    print("\n❌ Reproducibility test failed!")