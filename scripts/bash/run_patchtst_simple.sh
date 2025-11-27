#!/bin/bash

# Dataset list
DATASETS=(
    "2mm_#1_2classwith0.2mm_seed42"
    "5mm_#2_2classwith1.0mm_seed42"
)

# Model name
MODEL_NAME="patchtst"

# Loop through datasets
for DATASET in "${DATASETS[@]}"; do
    echo "Training on dataset: $DATASET"
    
    CHECKPOINT_DIR="checkpoints/${MODEL_NAME}_${DATASET}"
    
    CUDA_VISIBLE_DEVICES=2 python ../train_patchtst.py \
        --data_dir data/${DATASET} \
        --model_name ${MODEL_NAME} \
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
        --seed 42 \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --device cuda \
        --gpu_id 0 \
        --verbose
    
    echo "Completed training on $DATASET"
    echo "----------------------------------------"
done

echo "All training completed!"