#!/bin/bash

DATASETS=("MobiFall")
DATE=$(date +%Y%m%d)
BATCH_SIZE=64
PROJ_DIM=64

# 1. Default encoder with BIRCH
# for threshold in 0.3 0.7; do
#     TRIAL_NAME="${DATE}_${DATASETS[0]}_default_birch_th${threshold}_b${BATCH_SIZE}_dim${PROJ_DIM}"
#     python3 train_unsupervised_CAGE.py \
#         --dataset ${DATASETS[0]} \
#         --model CAGE \
#         --encoder_type default \
#         --clustering_method birch \
#         --birch_threshold $threshold \
#         --batch_size $BATCH_SIZE \
#         --proj_dim $PROJ_DIM \
#         --epochs 200 \
#         --window_width 128 \
#         --train_portion 1.0 \
#         --learning_rate 0.001 \
#         --weight_decay 1e-7 \
#         --momentum 0.9 \
#         --normalize \
#         --trial $TRIAL_NAME
# done

# # 2. Transformer encoder with BIRCH
# for threshold in 0.3 0.7; do
#     TRIAL_NAME="${DATE}_${DATASETS[0]}_transformer_birch_th${threshold}_b${BATCH_SIZE}_dim${PROJ_DIM}"
#     python3 train_unsupervised_CAGE.py \
#         --dataset ${DATASETS[0]} \
#         --model CAGE \
#         --encoder_type transformer \
#         --clustering_method birch \
#         --birch_threshold $threshold \
#         --batch_size $BATCH_SIZE \
#         --proj_dim $PROJ_DIM \
#         --epochs 200 \
#         --window_width 128 \
#         --train_portion 1.0 \
#         --learning_rate 0.001 \
#         --weight_decay 1e-7 \
#         --momentum 0.9 \
#         --normalize \
#         --trial $TRIAL_NAME
# done

# 3. ResNet-Transformer with different clustering methods
# 3.1 KMeans
TRIAL_NAME="${DATE}_${DATASETS[0]}_resnet_transformer_kmeans_b${BATCH_SIZE}_dim${PROJ_DIM}"
python3 train_unsupervised_CAGE.py \
    --dataset ${DATASETS[0]} \
    --model CAGE \
    --encoder_type resnet_transformer \
    --clustering_method kmeans \
    --batch_size $BATCH_SIZE \
    --proj_dim $PROJ_DIM \
    --epochs 200 \
    --window_width 128 \
    --train_portion 1.0 \
    --learning_rate 0.001 \
    --weight_decay 1e-7 \
    --momentum 0.9 \
    --normalize \
    --trial $TRIAL_NAME

# 3.2 DBSCAN with different eps
for eps in 0.3 0.7; do
    TRIAL_NAME="${DATE}_${DATASETS[0]}_resnet_transformer_dbscan_eps${eps}_b${BATCH_SIZE}_dim${PROJ_DIM}"
    python3 train_unsupervised_CAGE.py \
        --dataset ${DATASETS[0]} \
        --model CAGE \
        --encoder_type resnet_transformer \
        --clustering_method dbscan \
        --dbscan_eps $eps \
        --dbscan_min_samples 5 \
        --batch_size $BATCH_SIZE \
        --proj_dim $PROJ_DIM \
        --epochs 200 \
        --window_width 128 \
        --train_portion 1.0 \
        --learning_rate 0.001 \
        --weight_decay 1e-7 \
        --momentum 0.9 \
        --normalize \
        --trial $TRIAL_NAME
done

# 3.3 BIRCH with different thresholds
for threshold in 0.3 0.7; do
    TRIAL_NAME="${DATE}_${DATASETS[0]}_resnet_transformer_birch_th${threshold}_b${BATCH_SIZE}_dim${PROJ_DIM}"
    python3 train_unsupervised_CAGE.py \
        --dataset ${DATASETS[0]} \
        --model CAGE \
        --encoder_type resnet_transformer \
        --clustering_method birch \
        --birch_threshold $threshold \
        --batch_size $BATCH_SIZE \
        --proj_dim $PROJ_DIM \
        --epochs 200 \
        --window_width 128 \
        --train_portion 1.0 \
        --learning_rate 0.001 \
        --weight_decay 1e-7 \
        --momentum 0.9 \
        --normalize \
        --trial $TRIAL_NAME
done

# 3.4 FastCluster with different linkages
for linkage in "ward" "average"; do
    TRIAL_NAME="${DATE}_${DATASETS[0]}_resnet_transformer_fastcluster_${linkage}_b${BATCH_SIZE}_dim${PROJ_DIM}"
    python3 train_unsupervised_CAGE.py \
        --dataset ${DATASETS[0]} \
        --model CAGE \
        --encoder_type resnet_transformer \
        --clustering_method fastcluster \
        --fastcluster_linkage $linkage \
        --batch_size $BATCH_SIZE \
        --proj_dim $PROJ_DIM \
        --epochs 200 \
        --window_width 128 \
        --train_portion 1.0 \
        --learning_rate 0.001 \
        --weight_decay 1e-7 \
        --momentum 0.9 \
        --normalize \
        --trial $TRIAL_NAME
done