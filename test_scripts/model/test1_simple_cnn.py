# test_scripts/model/test_simple_cnn.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import numpy as np
from models.simple_cnn import SimpleCNN

def test_model():
    print("=== Testing SimpleCNN ===")
    
    # 모델 파라미터
    n_vars = 12
    n_classes = 2
    seq_len = 50
    batch_size = 16
    
    # 모델 생성
    model = SimpleCNN(n_vars, n_classes, seq_len)
    print(f"✅ Model created")
    print(f"Model:\n{model}")
    
    # 더미 입력
    x = torch.randn(batch_size, n_vars, seq_len)
    print(f"\n✅ Input shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Expected shape: ({batch_size}, {n_classes})")
    
    # 검증
    assert output.shape == (batch_size, n_classes), "Output shape mismatch!"
    
    # Softmax로 확률 변환
    probs = torch.softmax(output, dim=1)
    print(f"\n✅ Probabilities shape: {probs.shape}")
    print(f"✅ First sample probs: {probs[0].numpy()}")
    print(f"✅ Sum of probs: {probs[0].sum().item():.4f} (should be ~1.0)")
    
    # 파라미터 개수
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ Total parameters: {n_params:,}")
    
    print("\n✅ All model tests passed!")

if __name__ == "__main__":
    test_model()