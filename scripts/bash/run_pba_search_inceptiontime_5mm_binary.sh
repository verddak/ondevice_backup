#!/bin/bash

# PBA Search with Best InceptionTime Configuration
# Best config from your training:
#   n_blocks=9, batch_size=16, lr=1e-6
# Data Augmentation: PBA (Population Based)
# Seeds: 42-51 (10 runs)

echo "============================================"
echo "PBA Search with Best InceptionTime Config"
echo "Model: InceptionTime (n_blocks=9)"
echo "Regularization: weight_decay=1e-4, label_smoothing=0.1"
echo "Data Augmentation: PBA (Population Based)"
echo "Seeds: 42-51 (10 runs)"
echo "============================================"

# Seed 42부터 51까지 반복 실행
for SEED in {42..51}
do
    echo -e "\n>>> Running PBA Search with Seed ${SEED} <<<"
    
    CUDA_VISIBLE_DEVICES=0 python ../run_pba_search_inceptiontime.py \
        --data_dir data/5mm_#2_binary_seed42 \
        --n_filters 32 \
        --n_blocks 6 \
        --batch_size 32 \
        --epochs 100 \
        --lr 5e-5 \
        --patience 15 \
        --population_size 16 \
        --n_generations 50 \
        --n_elites 3 \
        --mutation_prob_method 0.2 \
        --mutation_prob_param 0.2 \
        --seed ${SEED} \
        --device cuda \
        --gpu_id 0 \
        --checkpoint_dir checkpoints/pba_search_inception_seed${SEED} \
        --log_dir checkpoints/pba_search_inception_seed${SEED}/logs \
        --results_file pba_search_inception_seed${SEED}_results.csv \
        --num_workers 4
    
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
    log_dir = Path(f"checkpoints/pba_search_inception_seed{seed}/logs")
    csv_file = log_dir / f"pba_search_inception_seed{seed}_results.csv"
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        best_row = df.loc[df['val_acc'].idxmax()]
        results.append({
            'seed': seed,
            'best_val_acc': best_row['val_acc'],
            'best_val_loss': best_row['val_loss'],
            'best_generation': best_row['generation'],
            'method1': best_row['method1'],
            'method2': best_row['method2']
        })

if results:
    summary_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("PBA SEARCH SUMMARY (InceptionTime)")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)
    print(f"\nStatistics:")
    print(f"  Mean Val Acc: {summary_df['best_val_acc'].mean():.2f}% ± {summary_df['best_val_acc'].std():.2f}%")
    print(f"  Best Val Acc: {summary_df['best_val_acc'].max():.2f}%")
    print(f"  Worst Val Acc: {summary_df['best_val_acc'].min():.2f}%")
    print(f"  Median Val Acc: {summary_df['best_val_acc'].median():.2f}%")
    print("="*60)
    
    # Save summary
    summary_path = Path('checkpoints/pba_search_inception_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")
    
    # Count augmentation method frequency
    print(f"\n" + "="*60)
    print("BEST AUGMENTATION METHODS")
    print("="*60)
    print("Method 1:")
    print(summary_df['method1'].value_counts())
    print("\nMethod 2:")
    print(summary_df['method2'].value_counts())
    print("="*60)
    
    # Compare with baseline if you have it
    # baseline_acc = XX.X  # Your baseline InceptionTime accuracy
    # mean_pba_acc = summary_df['best_val_acc'].mean()
    # improvement = mean_pba_acc - baseline_acc
    # 
    # print(f"\n" + "="*60)
    # print("COMPARISON WITH BASELINE")
    # print("="*60)
    # print(f"  Baseline (No Aug): {baseline_acc:.2f}%")
    # print(f"  PBA (Mean): {mean_pba_acc:.2f}%")
    # print(f"  Improvement: {improvement:+.2f}%")
    # print("="*60)
else:
    print("No PBA results found.")
EOF

echo -e "\n✅ PBA search pipeline completed!"