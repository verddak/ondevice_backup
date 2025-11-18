# scripts/train_inception_time.py
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
from models.inceptiontime import InceptionTime
from training.trainer import Trainer
from utils.seed import set_seed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train InceptionTime on HBM data')
    # Data
    parser.add_argument('--data_dir', type=str, default='data/hbm',help='Data directory (relative to project root)')
    parser.add_argument('--data_dir_abs', type=str, default=None,help='Absolute path to data directory (overrides data_dir)')
    # Model
    parser.add_argument('--model_name', type=str, default='inception_time',help='Model name for saving')
    parser.add_argument('--n_filters', type=int, default=32,help='Number of filters per inception module')
    parser.add_argument('--n_blocks', type=int, default=6,help='Number of InceptionBlocks')
    parser.add_argument('--kernel_sizes', type=int, nargs=3, default=[9, 19, 39],help='Three kernel sizes for inception modules')
    parser.add_argument('--bottleneck_channels', type=int, default=32,help='Bottleneck channels')
    parser.add_argument('--use_residual', action='store_true', default=True,help='Use residual connections')
    parser.add_argument('--no_residual', dest='use_residual', action='store_false',help='Disable residual connections')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate in inception blocks')
    parser.add_argument('--head_dropout', type=float, default=0.0,
                       help='Dropout rate for classification head')
    # Training
    parser.add_argument('--batch_size', type=int, default=64,help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor')
    parser.add_argument('--patience', type=int, default=15,help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,help='Random seed')
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
    print("InceptionTime Configuration:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Data dir: {DATA_DIR}")
    print(f"  Model: {args.model_name}")
    print(f"  n_filters: {args.n_filters}")
    print(f"  n_blocks: {args.n_blocks}")
    print(f"  kernel_sizes: {args.kernel_sizes}")
    print(f"  bottleneck_channels: {args.bottleneck_channels}")
    print(f"  use_residual: {args.use_residual}")
    print(f"  dropout: {args.dropout}")
    print(f"  head_dropout: {args.head_dropout}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Patience: {args.patience}")
    print(f"  Seed: {args.seed}")
    print("="*60)
    
    # ==================== Device ====================
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"\nUsing GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print(f"\nUsing CPU")
    
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
    print("Creating InceptionTime model...")
    model = InceptionTime(
        n_vars=n_vars,
        n_classes=n_classes,
        seq_len=seq_len,
        n_filters=args.n_filters,
        n_blocks=args.n_blocks,
        kernel_sizes=args.kernel_sizes,
        bottleneck_channels=args.bottleneck_channels,
        use_residual=args.use_residual,
        dropout=args.dropout,
        head_dropout=args.head_dropout
    )
    
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
    
    # Save args next to the checkpoint
    args_file = save_path.parent / f"{save_path.stem}_args.json"
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=2)
    print(f"Arguments saved to: {args_file}")
    
    print("\n" + "="*60)
    print(f"Training completed!")
    print(f"Model saved to: {save_path}")
    print(f"Val Accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Val Loss: {trainer.best_val_loss:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()