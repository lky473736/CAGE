#!/bin/bash

DATASETS=("SisFall")
ENCODER_TYPES=("default" "transformer" "unet")
NUM_ENCODERS=(2 4 8)
BATCH_SIZES=(64)
EMBEDDING_DIMS=(16)
PCA_COMPONENTS=(32 64) 
DATE=$(date +%Y%m%d)

for dataset in "${DATASETS[@]}"; do
    for num_encoder in "${NUM_ENCODERS[@]}"; do
        TRIAL_NAME="${DATE}_${dataset}_enc${num_encoder}_default_skip_b64_dim16"
        python3 train_unsupervised_CAGE.py \
            --dataset $dataset \
            --encoder_type default \
            --num_encoders $num_encoder \
            --use_skip \
            --batch_size 64 \
            --proj_dim 16 \
            --epochs 200 \
            --window_width 128 \
            --normalize \
            --trial $TRIAL_NAME
    done

    TRIAL_NAME="${DATE}_${dataset}_enc1_transformer_b64_dim16"
    python3 train_unsupervised_CAGE.py \
        --dataset $dataset \
        --encoder_type transformer \
        --num_heads 8 \
        --num_transformer_blocks 3 \
        --batch_size 64 \
        --proj_dim 16 \
        --epochs 200 \
        --window_width 128 \
        --normalize \
        --trial $TRIAL_NAME

    TRIAL_NAME="${DATE}_${dataset}_enc1_unet_b64_dim16"
    python3 train_unsupervised_CAGE.py \
        --dataset $dataset \
        --encoder_type unet \
        --batch_size 64 \
        --proj_dim 16 \
        --epochs 200 \
        --window_width 128 \
        --normalize \
        --trial $TRIAL_NAME

    for pca_comp in "${PCA_COMPONENTS[@]}"; do
        TRIAL_NAME="${DATE}_${dataset}_pca${pca_comp}_enc1_default_b64_dim16"
        python3 train_unsupervised_CAGE.py \
            --dataset $dataset \
            --encoder_type default \
            --use_pca \
            --pca_components $pca_comp \
            --batch_size 64 \
            --proj_dim 16 \
            --epochs 200 \
            --window_width 128 \
            --normalize \
            --trial $TRIAL_NAME
    done
done