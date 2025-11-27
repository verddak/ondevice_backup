#!/bin/bash

# ============================================
# CONFIGURATION
# ============================================
DATASET_NAMES=(
    "2mm_#1_2classwith1.0mm"
    "2mm_#1_2classwith0.5mm"
    "2mm_#1_2classwith0.2mm"
)
MODEL_NAME="inceptionTime"
SEED=42

# ============================================
# PBA SEARCH RUNS
# ============================================
for DATASET_NAME in "${DATASET_NAMES[@]}"
do
    # Derived paths
    DATA_DIR="data/${DATASET_NAME}_seed42"
    CHECKPOINT_DIR="checkpoints/pba_search_${MODEL_NAME}_${DATASET_NAME}_seed${SEED}"
    LOG_DIR="${CHECKPOINT_DIR}/logs"
    RESULTS_FILE="pba_search_${MODEL_NAME}_${DATASET_NAME}_seed${SEED}_results.csv"

    echo -e "\n>>> Running PBA Search: ${MODEL_NAME} on ${DATASET_NAME} (seed ${SEED}) <<<"

    CUDA_VISIBLE_DEVICES=2 python ../run_pba_search_inceptiontime.py \
        --data_dir ${DATA_DIR} \
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
        --checkpoint_dir ${CHECKPOINT_DIR} \
        --log_dir ${LOG_DIR} \
        --results_file ${RESULTS_FILE} \
        --num_workers 4

    echo "✅ Completed: ${DATASET_NAME}"
done

echo -e "\n============================================"
echo "All PBA searches completed!"
echo "Datasets: ${DATASET_NAMES[@]}"
echo "Model: ${MODEL_NAME}"
echo "Seed: ${SEED}"
echo "============================================"