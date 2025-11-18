# tests/test_model_load.py
import torch
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    checkpoint_path = Path(sys.argv[1])
    args_path = Path(sys.argv[2])
    
    if not args_path.exists():
        print("⚠️  No args file, skipping model load test")
        return
    
    # Load args
    with open(args_path) as f:
        args = json.load(f)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Determine model type
    model_type = checkpoint_path.parent.name
    
    print(f"\n🔧 Model type: {model_type}")
    
    # Create model
    if model_type == 'inception_time':
        from models.inceptiontime import InceptionTime
        model = InceptionTime(
            n_vars=1, n_classes=2, seq_len=1024,
            n_filters=args.get('n_filters', 32),
            n_blocks=args.get('n_blocks', 6)
        )
    elif model_type == 'patchtst':
        from models.patchtst import PatchTSTClassifier
        model = PatchTSTClassifier(
            n_vars=1, n_classes=2, seq_len=1024,
            d_model=args.get('d_model', 128),
            n_layers=args.get('n_layers', 3),
            n_heads=args.get('n_heads', 16)
        )
    elif model_type == 'simple_cnn':
        from models.simple_cnn import SimpleCNN
        model = SimpleCNN(n_vars=1, n_classes=2, seq_len=1024)
    else:
        print(f"❌ Unknown model type: {model_type}")
        sys.exit(1)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test forward pass
    dummy_input = torch.randn(2, 1, 1024)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✅ Model loaded and tested")
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output.shape}")

if __name__ == "__main__":
    main()