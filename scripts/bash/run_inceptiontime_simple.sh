#!/bin/bash

# Dataset list
DATASETS=(
    "5mm_#2_2classwith1.0mm_seed42"
    "5mm_#2_2classwith1.0mm_seed43"
)

# Model name
MODEL_NAME="inceptiontime"

# Loop through datasets
for DATASET in "${DATASETS[@]}"; do
    echo "Training on dataset: $DATASET"
    
    CHECKPOINT_DIR="checkpoints/${MODEL_NAME}_${DATASET}"
    
    python ../train_inceptiontime.py \
        --data_dir data/${DATASET} \
        --model_name ${MODEL_NAME} \
        --n_filters 32 \
        --n_blocks 6 \
        --batch_size 32 \
        --epochs 100 \
        --lr 5e-5 \
        --patience 15 \
        --seed 42 \
        --num_workers 4 \
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --log_dir logs \
        --device cuda \
        --gpu_id 2 \
        --verbose
    
    echo "Completed training on $DATASET"
    echo "----------------------------------------"
done

echo "All training completed!"