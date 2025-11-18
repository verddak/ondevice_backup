# scripts/run_train_patchtst_hf.sh
#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python ../train_patchtst.py \
    --data_dir data/5mm_#2_binary_seed42 \
    --model_name patchtst \
    --patch_len 32 \
    --stride 32 \
    --d_model 512 \
    --n_heads 16 \
    --n_layers 9 \
    --d_ff 1024 \
    --dropout 0.4 \
    --head_dropout 0.2 \
    --weight_decay 1e-4 \
    --use_cls_token \
    --batch_size 32 \
    --epochs 500 \
    --lr 1e-4 \
    --patience 15 \
    --seed 42 \
    --device cuda \
    --gpu_id 0 \
    --verbose