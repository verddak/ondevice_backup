#!/bin/bash

# Follow-up experiments based on grid search results
# Key findings:
# 1. patch_len=32, stride=32 works better
# 2. d_model=512 shows promise
# 3. Many models failed (25% acc = random)
# 4. Need to explore medium-large models more carefully

echo "============================================"
echo "PatchTST Follow-up Experiments"
echo "Focus: Fine-grained search around best configs"
echo "============================================"

exp_num=0

# ============================================
# Strategy 1: Explore around best config
# Best: d_model=512, n_layers=9, patch=32
# ============================================
echo -e "\n[STRATEGY 1: Around Best Config]"

# Vary n_layers around 9
exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: Best-7layers <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_best_7L \
    --patch_len 32 \
    --stride 32 \
    --d_model 512 \
    --n_heads 16 \
    --n_layers 7 \
    --d_ff 1024 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: Best-8layers <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_best_8L \
    --patch_len 32 \
    --stride 32 \
    --d_model 512 \
    --n_heads 16 \
    --n_layers 8 \
    --d_ff 1024 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

# Vary d_ff
exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: Best-d_ff=512 <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_best_ff512 \
    --patch_len 32 \
    --stride 32 \
    --d_model 512 \
    --n_heads 16 \
    --n_layers 9 \
    --d_ff 512 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: Best-d_ff=2048 <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_best_ff2048 \
    --patch_len 32 \
    --stride 32 \
    --d_model 512 \
    --n_heads 16 \
    --n_layers 9 \
    --d_ff 2048 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

# ============================================
# Strategy 2: Medium models with better regularization
# Problem: Many medium models failed (25% acc)
# Solution: Try with stronger regularization
# ============================================
echo -e "\n[STRATEGY 2: Medium Models + Stronger Regularization]"

exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: Medium-256-6L-Strong Reg <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_med_strongreg \
    --patch_len 32 \
    --stride 32 \
    --d_model 256 \
    --n_heads 16 \
    --n_layers 6 \
    --d_ff 1024 \
    --dropout 0.5 \
    --head_dropout 0.3 \
    --weight_decay 1e-3 \
    --label_smoothing 0.15 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: Medium-384-6L <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_med_384 \
    --patch_len 32 \
    --stride 32 \
    --d_model 384 \
    --n_heads 16 \
    --n_layers 6 \
    --d_ff 1536 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

# ============================================
# Strategy 3: Different patch configurations
# patch=32 worked better, explore more
# ============================================
echo -e "\n[STRATEGY 3: Patch Length Exploration]"

exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: patch=16, stride=16 <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_patch16 \
    --patch_len 16 \
    --stride 16 \
    --d_model 512 \
    --n_heads 16 \
    --n_layers 9 \
    --d_ff 1024 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: patch=32, stride=16 (overlap) <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_patch32_stride16 \
    --patch_len 32 \
    --stride 16 \
    --d_model 512 \
    --n_heads 16 \
    --n_layers 9 \
    --d_ff 1024 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: patch=64, stride=32 (overlap) <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_patch64_stride32 \
    --patch_len 64 \
    --stride 32 \
    --d_model 512 \
    --n_heads 16 \
    --n_layers 9 \
    --d_ff 1024 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

# ============================================
# Strategy 4: Moderate model sizes that failed before
# Try with different learning rates
# ============================================
echo -e "\n[STRATEGY 4: Failed Configs with Different LR]"

exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: 256-9L with higher LR <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_256_9L_highLR \
    --patch_len 32 \
    --stride 32 \
    --d_model 256 \
    --n_heads 16 \
    --n_layers 9 \
    --d_ff 1024 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 5e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

exp_num=$((exp_num+1))
echo -e "\n>>> Experiment $exp_num: 128-6L with higher LR <<<"
CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
    --data_dir data/5mm_#2_seed42 \
    --model_name patchtst_followup${exp_num}_128_6L_highLR \
    --patch_len 32 \
    --stride 32 \
    --d_model 128 \
    --n_heads 8 \
    --n_layers 6 \
    --d_ff 512 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --label_smoothing 0.1 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 100 \
    --lr 5e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose

# ============================================
# Strategy 5: Different seed for best config
# Check if 50% result is stable
# ============================================
echo -e "\n[STRATEGY 5: Best Config with Different Seeds]"

for seed in 123 456 789; do
    exp_num=$((exp_num+1))
    echo -e "\n>>> Experiment $exp_num: Best Config seed=$seed <<<"
    CUDA_VISIBLE_DEVICES=3 python ../train_patchtst.py \
        --data_dir data/5mm_#2_seed42 \
        --model_name patchtst_followup${exp_num}_best_seed${seed} \
        --patch_len 32 \
        --stride 32 \
        --d_model 512 \
        --n_heads 16 \
        --n_layers 9 \
        --d_ff 1024 \
        --dropout 0.4 \
        --head_dropout 0.2 \
        --weight_decay 1e-4 \
        --label_smoothing 0.1 \
        --use_cls_token \
        --batch_size 32 \
        --epochs 100 \
        --lr 1e-4 \
        --patience 15 \
        --seed $seed \
        --device cuda \
        --gpu_id 0 \
        --verbose
done

echo -e "\n============================================"
echo "Follow-up Experiments Completed!"
echo "Total experiments: $exp_num"
echo "============================================"

# Generate summary
python3 << 'EOF'
import json
from pathlib import Path
import pandas as pd

checkpoint_dir = Path("/data3/hai_ts/on-device/checkpoints/patchtst")
results = []

for args_file in sorted(checkpoint_dir.glob("*followup*_args.json")):
    with open(args_file) as f:
        data = json.load(f)
        results.append({
            'exp': data.get('model_name', 'N/A'),
            'd_model': data.get('d_model', 0),
            'n_layers': data.get('n_layers', 0),
            'd_ff': data.get('d_ff', 0),
            'patch': data.get('patch_len', 0),
            'stride': data.get('stride', 0),
            'lr': data.get('lr', 0),
            'seed': data.get('seed', 0),
            'params': data.get('n_params', 0),
            'val_acc': data.get('final_val_acc', 0),
            'val_loss': data.get('final_val_loss', 0),
        })

if results:
    df = pd.DataFrame(results)
    df = df.sort_values('val_acc', ascending=False)
    
    print("\n" + "="*100)
    print("FOLLOW-UP RESULTS (sorted by val_acc)")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)
    
    df.to_csv('followup_results.csv', index=False)
    print("\nResults saved to: followup_results.csv")
else:
    print("No follow-up results found yet.")
EOF

echo -e "\n✅ All follow-up experiments completed!"