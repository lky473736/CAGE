#!/bin/bash

DATASETS=("MobiFall" "SisFall")
DATE=$(date +%Y%m%d)

for dataset in "${DATASETS[@]}"; do
    TRIAL_NAME="${DATE}_${dataset}_enc1_default_lr0.001_b128_dim64_ep200"
    
    python3 train_unsupervised_CAGE.py \
        --dataset $dataset \
        --model CAGE \
        --batch_size 64 \
        --epochs 200 \
        --window_width 128 \
        --learning_rate 0.001 \
        --weight_decay 1e-7 \
        --momentum 0.9 \
        --normalize \
        --proj_dim 64 \
        --num_encoders 1 \
        --loss_type default \
        --trial $TRIAL_NAME
done