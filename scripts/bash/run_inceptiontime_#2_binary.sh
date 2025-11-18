# scripts/run_train_inception.sh
#!/bin/bash

# InceptionTime training script
python ../train_inceptiontime.py \
    --data_dir data/5mm_#2_binarywith0.5mm_seed42 \
    --model_name inceptiontime \
    --n_filters 32 \
    --n_blocks 6 \
    --batch_size 32 \
    --epochs 500 \
    --lr 5e-5 \
    --patience 15 \
    --seed 42 \
    --num_workers 4 \
    --checkpoint_dir checkpoints \
    --log_dir logs \
    --device cuda \
    --gpu_id 0 \
    --verbose