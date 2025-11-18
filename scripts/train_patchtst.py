# scripts/train_patchtst_hf.py
import sys
from pathlib import Path
import argparse

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data.loader import load_hbm_data, create_dataloaders
from models.patchtst import PatchTSTClassifier
from training.trainer import Trainer
from utils.seed import set_seed
from sklearn.metrics import confusion_matrix, classification_report


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train PatchTST (HuggingFace) on HBM data')
    # Data
    parser.add_argument('--data_dir', type=str, default='data/hbm',help='Data directory (relative to project root)')
    parser.add_argument('--data_dir_abs', type=str, default=None,help='Absolute path to data directory (overrides data_dir)')
    # Model
    parser.add_argument('--model_name', type=str, default='patchtst_hf',help='Model name for saving')
    # PatchTST Architecture
    parser.add_argument('--patch_len', type=int, default=16,help='Length of each patch')
    parser.add_argument('--stride', type=int, default=8,help='Stride for patching')
    parser.add_argument('--d_model', type=int, default=128,help='Dimension of the model')
    parser.add_argument('--n_heads', type=int, default=16,help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=3,help='Number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=256,help='Dimension of feedforward network')
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.1,help='Dropout rate')
    parser.add_argument('--head_dropout', type=float, default=0.1,help='Classification head dropout rate')
    # PatchTST specific
    parser.add_argument('--use_cls_token', action='store_true', default=True,help='Use [CLS] token for classification')
    parser.add_argument('--no_cls_token', dest='use_cls_token', action='store_false',help='Disable [CLS] token')
    # Training
    parser.add_argument('--batch_size', type=int, default=64,help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--patience', type=int, default=15,help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,help='Random seed')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor')
    # Checkpoint naming
    parser.add_argument('--save_format', type=str, default='timestamp_acc',choices=['timestamp_acc', 'acc_loss', 'simple'],help='Checkpoint filename format')
    # DataLoader
    parser.add_argument('--num_workers', type=int, default=4,help='Number of dataloader workers')
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',help='Checkpoint directory (relative to project root)')
    parser.add_argument('--log_dir', type=str, default='logs',help='Log directory (relative to project root)')
    # Device
    parser.add_argument('--device', type=str, default='cuda',choices=['cuda', 'cpu'],help='Device to use')
    parser.add_argument('--gpu_id', type=int, default=0,help='GPU ID to use')
    # Misc
    parser.add_argument('--verbose', action='store_true',help='Verbose output')
    parser.add_argument('--save_best_only', action='store_true',help='Save only the best model')
    
    return parser.parse_args()

def evaluate_and_show_confusion_matrix(model, dataloader, device, class_names=None):
    """
    Evaluate model and show confusion matrix
    
    Args:
        model: trained model
        dataloader: validation dataloader
        device: torch device
        class_names: list of class names (optional)
    
    Returns:
        accuracy, confusion_matrix
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    accuracy = 100.0 * sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    # Print confusion matrix
    print("\n" + "="*60)
    print("Confusion Matrix:")
    print("="*60)
    
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(cm))]
    
    # Print header
    print(f"{'Predicted →':>15}", end='')
    for name in class_names:
        print(f"{name:>15}", end='')
    print()
    print("Actual ↓" + " "*8 + "-"*(15*len(class_names)))
    
    # Print matrix
    for i, name in enumerate(class_names):
        print(f"{name:>15}", end='')
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>15}", end='')
        print()
    
    print("="*60)
    
    # Print detailed classification report
    print("\nClassification Report:")
    print("="*60)
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4
    ))
    
    # Print per-class accuracy
    print("Per-class Accuracy:")
    print("-"*60)
    for i, name in enumerate(class_names):
        class_total = cm[i].sum()
        class_correct = cm[i, i]
        class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0
        print(f"  {name:>15}: {class_correct:>3}/{class_total:<3} = {class_acc:>6.2f}%")
    print("="*60)
    
    return accuracy, cm

def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # ==================== Setup Paths ====================
    if args.data_dir_abs:
        DATA_DIR = Path(args.data_dir_abs)
    else:
        DATA_DIR = PROJECT_ROOT / args.data_dir
    
    CHECKPOINT_DIR = PROJECT_ROOT / args.checkpoint_dir
    LOG_DIR = PROJECT_ROOT / args.log_dir
    
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("PatchTST (HuggingFace) Configuration:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Model: {args.model_name}")
    print(f"  Patch length: {args.patch_len}")
    print(f"  Stride: {args.stride}")
    print(f"  d_model: {args.d_model}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  d_ff: {args.d_ff}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Head dropout: {args.head_dropout}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Use CLS token: {args.use_cls_token}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Patience: {args.patience}")
    print(f"  Seed: {args.seed}")
    print("="*60)
    
    # ==================== Device ====================
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"\n✅ Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print(f"\n✅ Using CPU")
    
    # ==================== Load Data ====================
    print("\n" + "="*60)
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_hbm_data(DATA_DIR)
    
    # Get dimensions
    n_samples, n_vars, seq_len = X_train.shape
    n_classes = len(np.unique(y_train))
    
    print(f"\nDataset info:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples: {len(X_val)}")
    print(f"  Input shape: ({n_vars}, {seq_len})")
    print(f"  Number of classes: {n_classes}")
    
    # Calculate number of patches
    patch_num = int((seq_len - args.patch_len) / args.stride + 1)
    print(f"  Number of patches: {patch_num}")
    
    # ==================== Create DataLoaders ====================
    train_loader, val_loader = create_dataloaders(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    print(f"\nDataLoader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # ==================== Create Model ====================
    print("\n" + "="*60)
    print("Creating PatchTST (HuggingFace) model...")
    print("\n" + "="*60)
    print("Checking first batch shape...")
    for batch_x, batch_y in train_loader:
        print(f"  DataLoader output: {batch_x.shape}")
        print(f"  Expected: [batch, n_vars={n_vars}, seq_len={seq_len}]")
        print(f"  Actual: [batch={batch_x.shape[0]}, {batch_x.shape[1]}, {batch_x.shape[2]}]")
        break
    print("="*60)
    
    try:
        model = PatchTSTClassifier(
            n_vars=n_vars,
            n_classes=n_classes,
            seq_len=seq_len,
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            head_dropout=args.head_dropout,
            use_cls_token=args.use_cls_token
        )
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        print("\n⚠️  Make sure transformers is properly installed!")
        print("    pip install transformers")
        return
    
    n_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ==================== Setup Training ====================
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=args.patience
    )
    
    # ==================== Train ====================
    print("\n" + "="*60)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        verbose=args.verbose or True
    )
    
    print("\n" + "="*60)
    print("Evaluating best model on validation set...")
    print("="*60)
    
    # Define class names (adjust based on your dataset)
    class_names = ['no_defect', 'defect_0.2mm', 'defect_0.5mm', 'defect_1.0mm']
    
    val_acc, val_cm = evaluate_and_show_confusion_matrix(
        model=trainer.model,
        dataloader=val_loader,
        device=device,
        class_names=class_names
    )
    
    # ==================== Final Results ====================
    print("\n" + "="*60)
    print("Final Results:")
    print(f"  Best Val Loss: {trainer.best_val_loss:.4f}")
    print(f"  Best Val Acc: {trainer.best_val_acc:.2f}%")
    print(f"  Final Train Acc: {history['train_acc'][-1]:.2f}%")
    print(f"  Final Val Acc: {history['val_acc'][-1]:.2f}%")
    print(f"  Total epochs: {len(history['train_loss'])}")
    
    # ==================== Save Model ====================
    save_path = trainer.save_checkpoint(
        checkpoint_dir=CHECKPOINT_DIR,
        model_name=args.model_name,
        save_format=args.save_format
    )
    
    # Save training args
    import json
    args_dict = vars(args)
    args_dict['project_root'] = str(PROJECT_ROOT)
    args_dict['data_dir_resolved'] = str(DATA_DIR)
    args_dict['final_val_acc'] = float(trainer.best_val_acc)
    args_dict['final_val_loss'] = float(trainer.best_val_loss)
    args_dict['save_path'] = str(save_path)
    args_dict['n_params'] = n_params
    args_dict['patch_num'] = patch_num
    args_dict['confusion_matrix'] = val_cm.tolist()
    
    # Save args next to the checkpoint
    args_file = save_path.parent / f"{save_path.stem}_args.json"
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=2)
    print(f"Arguments saved to: {args_file}")
    
    print("\n" + "="*60)
    print(f"✅ Training completed!")
    print(f"📁 Model saved to: {save_path}")
    print(f"🎯 Val Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"📉 Val Loss: {trainer.best_val_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()