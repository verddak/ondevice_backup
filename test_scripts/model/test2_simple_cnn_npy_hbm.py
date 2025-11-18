# test_scripts/model/test_cnn_with_npy.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from data.loader import load_hbm_data, create_dataloaders
from models.simple_cnn import SimpleCNN

def test_model_with_npy():
    print("=== Testing SimpleCNN with .npy Data ===")
    
    # 1. 데이터 로드
    data_dir = '/data3/hai_ts/on-device/data/hbm'
    X_train, y_train, X_val, y_val = load_hbm_data(data_dir)
    
    # 2. 파라미터 추출
    n_samples, n_vars, seq_len = X_train.shape
    n_classes = len(np.unique(y_train))
    
    print(f"\n=== Dataset Info ===")
    print(f"n_vars (channels): {n_vars}")
    print(f"seq_len (timesteps): {seq_len}")
    print(f"n_classes: {n_classes}")
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    
    # 3. DataLoader 생성
    train_loader, val_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=64,
        seed=42,
        num_workers=0
    )
    
    # 4. 모델 생성
    model = SimpleCNN(n_vars, n_classes, seq_len)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\n=== Model Info ===")
    print(f"Device: {device}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    
    # 5. Forward pass 테스트
    model.eval()
    print(f"\n=== Testing Forward Pass ===")
    
    with torch.no_grad():
        # Train batch
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_x)
            _, predicted = output.max(1)
            correct = predicted.eq(batch_y).sum().item()
            accuracy = 100. * correct / len(batch_y)
            
            print(f"Train batch: X={batch_x.shape}, y={batch_y.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Random accuracy: {accuracy:.2f}%")
            break
        
        # Val batch
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_x)
            _, predicted = output.max(1)
            correct = predicted.eq(batch_y).sum().item()
            accuracy = 100. * correct / len(batch_y)
            
            print(f"\nVal batch: X={batch_x.shape}, y={batch_y.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Random accuracy: {accuracy:.2f}%")
            break
    
    print("\n✅ All tests passed! Ready for training.")

if __name__ == "__main__":
    test_model_with_npy()