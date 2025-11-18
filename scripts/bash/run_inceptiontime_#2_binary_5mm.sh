# scripts/run_train_inception.sh
#!/bin/bash

# InceptionTime training script
python ../train_inceptiontime.py \
    --data_dir data/5mm_#2_binary_seed42 \
    --model_name inception_time_dropout \
    --n_filters 32 \
    --n_blocks 3 \
    --batch_size 16 \
    --epochs 500 \
    --weight_decay 1e-5 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --num_workers 4 \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --device cuda \
    --gpu_id 1 \
    --verbose