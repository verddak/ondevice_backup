#!/bin/bash

# PBA Search with Best PatchTST Configuration
# Best config from grid search:
#   patch=32, stride=32, d_model=512, n_layers=9, d_ff=1024
#   Val Acc: 50%, Params: 19M

echo "============================================"
echo "PBA Search with Best PatchTST Config"
echo "Model: d_model=512, n_layers=9, d_ff=1024"
echo "Regularization: dropout=0.4, weight_decay=1e-4"
echo "Data Augmentation: PBA (Population Based)"
echo "Seeds: 42-51 (10 runs)"
echo "============================================"

# Seed 42부터 51까지 반복 실행
for SEED in {42..51}
do
    echo -e "\n>>> Running PBA Search with Seed ${SEED} <<<"
    
    CUDA_VISIBLE_DEVICES=1 python ../run_pba_search_patchtst.py \
        --data_dir data/2mm_#1_binary_seed42 \
        --patch_len 32 \
        --stride 32 \
        --d_model 512 \
        --n_heads 16 \
        --n_layers 9 \
        --d_ff 1024 \
        --dropout 0.4 \
        --weight_decay 1e-4 \
        --use_cls_token \
        --batch_size 32 \
        --epochs 100 \
        --lr 1e-4 \
        --patience 15 \
        --population_size 16 \
        --n_generations 50 \
        --n_elites 3 \
        --mutation_prob_method 0.2 \
        --mutation_prob_param 0.2 \
        --seed ${SEED} \
        --device cuda \
        --gpu_id 0 \
        --checkpoint_dir checkpoints/pba_search_best_seed${SEED} \
        --log_dir checkpoints/pba_search_best_seed${SEED}/logs \
        --results_file pba_search_best_seed${SEED}_results.csv
    
    echo "Completed seed ${SEED}"
done

echo -e "\n============================================"
echo "All PBA searches completed!"
echo "Seeds: 42-51 (10 runs total)"
echo "============================================"

# Generate summary
echo -e "\nGenerating summary of all PBA runs..."
python3 << 'EOF'
import pandas as pd
from pathlib import Path
import numpy as np

results = []
for seed in range(42, 52):
    csv_file = Path(f"pba_search_best_seed{seed}_results.csv")
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        best_row = df.loc[df['val_acc'].idxmax()]
        results.append({
            'seed': seed,
            'best_val_acc': best_row['val_acc'],
            'best_val_loss': best_row['val_loss'],
            'best_generation': best_row['generation'],
        })

if results:
    summary_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("PBA SEARCH SUMMARY (Best PatchTST Config)")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    print(f"\nStatistics:")
    print(f"  Mean Val Acc: {summary_df['best_val_acc'].mean():.2f}% ± {summary_df['best_val_acc'].std():.2f}%")
    print(f"  Best Val Acc: {summary_df['best_val_acc'].max():.2f}%")
    print(f"  Worst Val Acc: {summary_df['best_val_acc'].min():.2f}%")
    print(f"  Median Val Acc: {summary_df['best_val_acc'].median():.2f}%")
    print("="*60)
    
    summary_df.to_csv('pba_search_best_summary.csv', index=False)
    print("\nSummary saved to: pba_search_best_summary.csv")
    
    # Compare with baseline (no augmentation)
    baseline_acc = 50.0  # From grid search
    mean_pba_acc = summary_df['best_val_acc'].mean()
    improvement = mean_pba_acc - baseline_acc
    
    print(f"\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)
    print(f"  Baseline (No Aug): {baseline_acc:.2f}%")
    print(f"  PBA (Mean): {mean_pba_acc:.2f}%")
    print(f"  Improvement: {improvement:+.2f}%")
    print("="*60)
else:
    print("No PBA results found.")
EOF

echo -e "\n✅ PBA search pipeline completed!"