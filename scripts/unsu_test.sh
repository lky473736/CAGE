#!/bin/bash

DATASETS=("MobiFall" "SisFall" "UMAFall")
LOSS_TYPES=("default")
NUM_ENCODERS=(1)
BATCH_SIZES=(64)
EMBEDDING_DIMS=(64)
DATE=$(date +%Y%m%d)

for dataset in "${DATASETS[@]}"; do
    for loss_type in "${LOSS_TYPES[@]}"; do
        for num_encoder in "${NUM_ENCODERS[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for proj_dim in "${EMBEDDING_DIMS[@]}"; do
                    TRIAL_NAME="${DATE}_${dataset}_enc${num_encoder}_${loss_type}_lr0.001_b${batch_size}_dim${proj_dim}_ep200"
                    
                    if [ "$loss_type" == "default" ]; then
                        python3 train_CAGE_edited.py \
                            --dataset $dataset \
                            --model CAGE \
                            --batch_size $batch_size \
                            --epochs 200 \
                            --window_width 128 \
                            --train_portion 0.6 \
                            --learning_rate 0.001 \
                            --weight_decay 1e-7 \
                            --momentum 0.9 \
                            --normalize \
                            --proj_dim $proj_dim \
                            --num_encoders $num_encoder \
                            --loss_type $loss_type \
                            --trial $TRIAL_NAME
                    else
                        python3 train_NTXent_triplet_CAGE.py \
                            --dataset $dataset \
                            --model CAGE \
                            --batch_size $batch_size \
                            --epochs 200 \
                            --window_width 128 \
                            --train_portion 0.6 \
                            --learning_rate 0.001 \
                            --weight_decay 1e-7 \
                            --momentum 0.9 \
                            --normalize \
                            --proj_dim $proj_dim \
                            --num_encoders $num_encoder \
                            --loss_type $loss_type \
                            --trial $TRIAL_NAME
                    fi
                done
            done
        done
    done
done