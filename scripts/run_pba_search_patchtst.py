# scripts/run_pba_search.py
import sys
from pathlib import Path
import argparse
import json

# 프로젝트 루트 설정
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from data.loader import load_hbm_data, create_dataloaders
from models.patchtst import PatchTSTClassifier
from training.trainer import Trainer
from search.searcher import PBASearcher
from augmentation.augment_methods import get_augmentation, get_default_augmentations
from utils.seed import set_seed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run PBA search for HBM data')
    # Data
    parser.add_argument('--data_dir', type=str, default='data/hbm',help='Data directory (relative to project root)')
    parser.add_argument('--data_dir_abs', type=str, default=None,help='Absolute path to data directory (overrides data_dir)')
    # Model Architecture (PatchTST)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=16)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--head_dropout', type=float, default=0.1)
    parser.add_argument('--use_cls_token', action='store_true', default=True)
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100,help='Epochs per policy evaluation')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay (L2 regularization)')  # 🔴 ADD
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                       help='Label smoothing factor')  # 🔴 ADD
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--num_workers', type=int, default=4)
    # PBA Search
    parser.add_argument('--population_size', type=int, default=16,help='Number of augmentation policies in population')
    parser.add_argument('--n_generations', type=int, default=50,help='Number of evolutionary generations')
    parser.add_argument('--n_elites', type=int, default=3,help='Number of elite policies to preserve')
    parser.add_argument('--mutation_prob_method', type=float, default=0.2,help='Probability of changing augmentation method')
    parser.add_argument('--mutation_prob_param', type=float, default=0.2,help='Probability of random parameter reset')
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=0)
    # Paths
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/pba_search')
    parser.add_argument('--log_dir', type=str, default='logs/pba_search')
    parser.add_argument('--results_file', type=str, default='pba_search_results.csv',help='CSV file to save results')
    
    return parser.parse_args()


def create_model(args, n_vars, n_classes, seq_len):
    """Create a fresh model instance"""
    return PatchTSTClassifier(
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

def save_results_csv(results, save_path):
    """Save results to CSV file"""
    import csv
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow([
            'generation', 'policy_idx', 
            'val_acc', 'val_loss',  # 🔴 ADD val_loss
            'method1', 'prob1', 'mag1',
            'method2', 'prob2', 'mag2'
        ])
        
        # Data
        for row in results:
            writer.writerow(row)
    
    print(f"\nResults saved to: {save_path}")


def evaluate_policy(policy, args, train_loader, val_loader, 
                   n_vars, n_classes, seq_len, device, generation, policy_idx,
                   save_checkpoint=False, checkpoint_dir=None):
    """
    Train a model with given augmentation policy and return validation accuracy
    
    Args:
        save_checkpoint: If True, save the trained model checkpoint
        checkpoint_dir: Directory to save checkpoint
    
    Returns:
        val_acc: Best validation accuracy achieved
        val_loss: Best validation loss
        trainer: Trainer object (for saving checkpoint later)
    """
    # Create fresh model
    model = create_model(args, n_vars, n_classes, seq_len)
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)  # 🔴 ADD label_smoothing
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay  # 🔴 ADD weight_decay
    )
    
    # Get augmentation pipeline from policy
    augmentation = policy.get_transforms()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=args.patience,
        augmentation=augmentation
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Generation {generation+1}, Policy {policy_idx+1}/{args.population_size}")
    print(f"Policy: {policy}")
    print(f"{'='*60}")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        verbose=False  # PBA search 중에는 간결하게
    )
    
    # Save checkpoint if requested
    if save_checkpoint and checkpoint_dir is not None:
        policy_dict = policy.to_dict()
        model_name = (f"gen{generation+1}_policy{policy_idx+1}_"
                     f"{policy_dict['op1']['method']}_"
                     f"{policy_dict['op2']['method']}")
        trainer.save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            save_format='acc_loss'
        )
    
    return trainer.best_val_acc, trainer.best_val_loss, trainer  # 🔴 ADD val_loss


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
    
    print(f"Using device: {device}")
    
    # ==================== Print Configuration ====================
    print("\n" + "="*60)
    print("PBA Search Configuration:")
    print(f"  Model: PatchTST")
    print(f"  d_model: {args.d_model}, n_layers: {args.n_layers}")
    print(f"  dropout: {args.dropout}, head_dropout: {args.head_dropout}")
    print(f"  weight_decay: {args.weight_decay}")  # 🔴 PRINT
    print(f"  label_smoothing: {args.label_smoothing}")  # 🔴 PRINT
    print(f"  Population size: {args.population_size}")
    print(f"  Generations: {args.n_generations}")
    print("="*60)
    
    # ==================== Load Data ====================
    print("\nLoading data...")
    X_train, y_train, X_val, y_val = load_hbm_data(DATA_DIR)
    n_samples, n_vars, seq_len = X_train.shape
    n_classes = len(np.unique(y_train))
    
    print(f"Data shape: {X_train.shape}")
    print(f"Number of classes: {n_classes}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers
    )
    
    # ==================== Initialize PBA Searcher ====================
    print("\nInitializing PBA searcher...")
    augmentation_classes = get_default_augmentations()
    
    searcher = PBASearcher(
        augmentation_classes=augmentation_classes,
        population_size=args.population_size,
        n_elites=args.n_elites,
        mutation_prob_method=args.mutation_prob_method,
        mutation_prob_param=args.mutation_prob_param,
        seed=args.seed
    )
    
    print(f"Population size: {args.population_size}")
    print(f"Number of elites: {args.n_elites}")
    print(f"Number of generations: {args.n_generations}")
    print(f"Available augmentations: {[cls.__name__ for cls in augmentation_classes]}")
    
    # ==================== PBA Search Loop ====================
    results = []
    
    for generation in range(args.n_generations):
        print(f"\n{'#'*60}")
        print(f"# GENERATION {generation+1}/{args.n_generations}")
        print(f"{'#'*60}")
        
        # Check if this is the last generation
        is_last_generation = (generation == args.n_generations - 1)
        
        # Evaluate all policies in population
        fitness_scores = []
        trainers = []
        
        for policy_idx, policy in enumerate(searcher.get_population()):
            val_acc, val_loss, trainer = evaluate_policy(  # 🔴 UNPACK val_loss
                policy=policy,
                args=args,
                train_loader=train_loader,
                val_loader=val_loader,
                n_vars=n_vars,
                n_classes=n_classes,
                seq_len=seq_len,
                device=device,
                generation=generation,
                policy_idx=policy_idx,
                save_checkpoint=False,
                checkpoint_dir=CHECKPOINT_DIR
            )
            
            fitness_scores.append(val_acc)
            
            if is_last_generation:
                trainers.append((policy_idx, trainer, policy))
            
            # Save result
            policy_dict = policy.to_dict()
            results.append([
                generation,
                policy_idx,
                val_acc,
                val_loss,  # 🔴 ADD val_loss
                policy_dict['op1']['method'],
                policy_dict['op1']['prob'],
                policy_dict['op1']['mag'],
                policy_dict['op2']['method'],
                policy_dict['op2']['prob'],
                policy_dict['op2']['mag']
            ])
        
        # Print generation summary
        print(f"\n{'='*60}")
        print(f"Generation {generation+1} Summary:")
        print(f"  Best Val Acc:  {max(fitness_scores):.2f}%")
        print(f"  Worst Val Acc: {min(fitness_scores):.2f}%")
        print(f"  Mean Val Acc:  {np.mean(fitness_scores):.2f}%")
        print(f"  Std Val Acc:   {np.std(fitness_scores):.2f}%")
        print(f"{'='*60}")
        
        # Save top 3 models from last generation
        if is_last_generation:
            print("\nSaving top 3 models from final generation...")
            # Sort by validation accuracy
            sorted_trainers = sorted(trainers, key=lambda x: x[1].best_val_acc, reverse=True)
            
            for rank, (policy_idx, trainer, policy) in enumerate(sorted_trainers[:3]):
                policy_dict = policy.to_dict()
                model_name = (f"top{rank+1}_"
                             f"acc{trainer.best_val_acc:.2f}_"
                             f"{policy_dict['op1']['method']}_"
                             f"{policy_dict['op2']['method']}")
                
                save_path = trainer.save_checkpoint(
                    checkpoint_dir=CHECKPOINT_DIR,
                    model_name=model_name,
                    save_format='simple'
                )
                print(f"  Rank {rank+1}: {model_name} (acc={trainer.best_val_acc:.2f}%)")
        
        # Evolve population (except for last generation)
        if not is_last_generation:
            searcher.evolve(fitness_scores)
        
        # ========== 매 세대마다 저장! ==========
        # Save intermediate CSV results
        save_results_csv(results, LOG_DIR / args.results_file)
        
        # Save intermediate search history (JSON)
        history_path = LOG_DIR / 'pba_search_history.json'
        searcher.save_history(history_path)
        
        print(f"Progress saved: Generation {generation+1}/{args.n_generations}")
        # =====================================
    
    # ==================== Final Results ====================
    print(f"\n{'#'*60}")
    print(f"# PBA SEARCH COMPLETED")
    print(f"{'#'*60}")
    
    best_policy = searcher.get_best_policy()
    print(f"\nBest Policy Found:")
    print(f"  {best_policy}")
    print(f"  Validation Accuracy: {best_policy.fitness:.2f}%")
    
    # Final save (중복이지만 확실하게)
    history_path = LOG_DIR / 'pba_search_history.json'
    searcher.save_history(history_path)
    print(f"\nSearch history saved to: {history_path}")
    
    save_results_csv(results, LOG_DIR / args.results_file)
    
    best_policy_path = LOG_DIR / 'best_policy.json'
    with open(best_policy_path, 'w') as f:
        json.dump(best_policy.to_dict(), f, indent=2)
    print(f"Best policy saved to: {best_policy_path}")
    
    print(f"\n{'='*60}")
    print(f"Top 3 models saved in: {CHECKPOINT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()