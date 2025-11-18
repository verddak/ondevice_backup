# test_ucr_data.py
from tsai.all import get_UCR_data
from data.loader import create_dataloaders
import numpy as np

# 실제 데이터 로드
X, y, splits = get_UCR_data(
    'hbm', 
    path='/data2/hai_ts/hbm/maker/finaldata',
    parent_dir='YOUR_DATASET_FOLDER',  # 실제 폴더명으로 변경
    return_split=False
)

print(f"=== UCR Data Info ===")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Train split size: {len(splits[0])}")
print(f"Val split size: {len(splits[1])}")
print(f"Unique classes: {np.unique(y)}")

# DataLoader 생성
train_loader, val_loader = create_dataloaders(
    X[splits[0]], y[splits[0]],
    X[splits[1]], y[splits[1]],
    batch_size=64,
    seed=42,
    num_workers=0
)

print(f"\n=== DataLoader Info ===")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# 첫 배치 확인
print(f"\n=== First Train Batch ===")
for batch_x, batch_y in train_loader:
    print(f"Batch X: {batch_x.shape}, range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
    print(f"Batch y: {batch_y.shape}, values: {batch_y[:10]}")
    break

print(f"\n=== First Val Batch ===")
for batch_x, batch_y in val_loader:
    print(f"Batch X: {batch_x.shape}, range: [{batch_x.min():.3f}, {batch_x.max():.3f}]")
    print(f"Batch y: {batch_y.shape}, values: {batch_y[:10]}")
    break

# 전체 데이터 순회 테스트
print(f"\n=== Full Iteration Test ===")
total_samples = 0
for batch_x, batch_y in train_loader:
    total_samples += len(batch_y)

print(f"Total samples iterated: {total_samples}")
print(f"Expected samples: {len(splits[0])}")
assert total_samples == len(splits[0]), "Sample count mismatch!"

print("\n✅ UCR data test passed!")