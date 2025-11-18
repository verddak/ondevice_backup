# scripts/run_train.sh
#!/bin/bash

# Basic training script
python train_simple_cnn.py \
    --data_dir data/hbm \
    --model_name simple_cnn \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --num_workers 4 \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --device cuda \
    --gpu_id 3 \
    --verbose