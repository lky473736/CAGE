#!/bin/bash

DATASETS=("MobiFall")
ENCODER_TYPES=("default" "transformer" "unet")
NUM_ENCODERS=(2 4 8)
BATCH_SIZES=(64)
EMBEDDING_DIMS=(64)
PCA_COMPONENTS=(2 4 8 16) 
DATE=$(date +%Y%m%d)

# TRIAL_NAME="${DATE}_${dataset}_enc${num_encoder}_${loss_type}_lr0.001_b${batch_size}_dim${proj_dim}_ep200"

# if [ "$loss_type" == "default" ]; then
#     python3 train_unsupervised_CAGE.py \
#         --dataset $dataset \
#         --model CAGE \
#         --batch_size $batch_size \
#         --epochs 200 \
#         --window_width 128 \
#         --train_portion 0.6 \
#         --learning_rate 0.001 \
#         --weight_decay 1e-7 \
#         --momentum 0.9 \
#         --normalize \
#         --proj_dim $proj_dim \
#         --num_encoders $num_encoder \
#         --loss_type $loss_type \
#         --trial $TRIAL_NAME
# fi

for dataset in "${DATASETS[@]}"; do
    for num_encoder in "${NUM_ENCODERS[@]}"; do
        TRIAL_NAME="${DATE}_${dataset}_enc_default${num_encoder}_use_skip_loss_default_lr0.001_b64_dim16_ep200"
        python3 train_unsupervised_CAGE.py \
            --model CAGE \
            --dataset $dataset \
            --encoder_type default \
            --num_encoders $num_encoder \
            --use_skip \
            --batch_size 64 \
            --epochs 200 \
            --window_width 128 \
            --train_portion 0.6 \
            --learning_rate 0.001 \
            --weight_decay 1e-7 \
            --momentum 0.9 \
            --normalize \
            --proj_dim 16 \
            --trial $TRIAL_NAME
    done

    TRIAL_NAME="${DATE}_${dataset}_enc_transformer_loss_default_lr0.001_b64_dim16_ep200"
    python3 train_unsupervised_CAGE.py \
        --model CAGE \
        --dataset $dataset \
        --encoder_type transformer \
        --num_heads 8 \
        --num_transformer_blocks 3 \
        --batch_size 64 \
        --epochs 200 \
        --window_width 128 \
        --train_portion 0.6 \
        --learning_rate 0.001 \
        --weight_decay 1e-7 \
        --momentum 0.9 \
        --normalize \
        --proj_dim 16 \
        --trial $TRIAL_NAME

    TRIAL_NAME="${DATE}_${dataset}_enc_unet_loss_default_lr0.001_b64_dim16_ep200"
    python3 train_unsupervised_CAGE.py \
        --model CAGE \
        --dataset $dataset \
        --encoder_type unet \
        --batch_size 64 \
        --epochs 200 \
        --window_width 128 \
        --train_portion 0.6 \
        --learning_rate 0.001 \
        --weight_decay 1e-7 \
        --momentum 0.9 \
        --normalize \
        --proj_dim 16 \
        --trial $TRIAL_NAME

    # for pca_comp in "${PCA_COMPONENTS[@]}"; do
    #     TRIAL_NAME="${DATE}_${dataset}_pca${pca_comp}_enc_default1_loss_default_lr0.001_b64_dim16_ep200"
    #     python3 train_unsupervised_CAGE.py \
    #         --model CAGE \
    #         --dataset $dataset \
    #         --encoder_type default \
    #         --use_pca \
    #         --pca_components $pca_comp \
    #         --batch_size 64 \
    #         --epochs 200 \
    #         --window_width 128 \
    #         --train_portion 0.6 \
    #         --learning_rate 0.001 \
    #         --weight_decay 1e-7 \
    #         --momentum 0.9 \
    #         --normalize \
    #         --proj_dim 16 \
    #         --trial $TRIAL_NAME
    # done
done