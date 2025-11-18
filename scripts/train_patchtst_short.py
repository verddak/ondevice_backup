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
    # ==================== Device ====================
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    # ==================== Load Data ====================
    X_train, y_train, X_val, y_val = load_hbm_data(DATA_DIR)
    n_samples, n_vars, seq_len = X_train.shape
    n_classes = len(np.unique(y_train))
    patch_num = int((seq_len - args.patch_len) / args.stride + 1)
    # ==================== Create DataLoaders ====================
    train_loader, val_loader = create_dataloaders(X_train, y_train,X_val, y_val,batch_size=args.batch_size,seed=args.seed,num_workers=args.num_workers)
    # ==================== Create Model ====================
    model = PatchTSTClassifier(n_vars=n_vars,n_classes=n_classes,seq_len=seq_len, patch_len=args.patch_len,stride=args.stride,d_model=args.d_model,_heads=args.n_heads, n_layers=args.n_layers,d_ff=args.d_ff,dropout=args.dropout,head_dropout=args.head_dropout,use_cls_token=args.use_cls_token)
    n_params = sum(p.numel() for p in model.parameters())
    # ==================== Setup Training ====================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model=model,optimizer=optimizer,criterion=criterion,device=device,patience=args.patience)
    # ==================== Train ====================
    print("\n" + "="*60)
    history = trainer.train(train_loader=train_loader,val_loader=val_loader,epochs=args.epochs,verbose=args.verbose or True)
    val_acc = history['val_acc']
    # ==================== Final Results ====================
    # ==================== Save Model ====================
    save_path = trainer.save_checkpoint(checkpoint_dir=CHECKPOINT_DIR,model_name=args.model_name,save_format=args.save_format)
    
    # Save training arg
    import json
    args_dict = vars(args)
    args_dict['project_root'] = str(PROJECT_ROOT)
    args_dict['data_dir_resolved'] = str(DATA_DIR)
    args_dict['final_val_acc'] = float(trainer.best_val_acc)
    args_dict['final_val_loss'] = float(trainer.best_val_loss)
    args_dict['save_path'] = str(save_path)
    args_dict['n_params'] = n_params
    args_dict['patch_num'] = patch_num

    args_file = save_path.parent / f"{save_path.stem}_args.json"
    with open(args_file, 'w') as f:
        json.dump(args_dict, f, indent=2)
if __name__ == "__main__":
    main()