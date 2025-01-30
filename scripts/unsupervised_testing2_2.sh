

# #!/bin/bash

# DATASETS=("MobiFall")
# ENCODERS=("default" "transformer" "resnet_transformer")
# CLUSTERING=("kmeans" "dbscan" "birch" "fastcluster")
# BATCH_SIZES=(64)
# EMBEDDING_DIMS=(64)
# DATE=$(date +%Y%m%d)

# for dataset in "${DATASETS[@]}"; do
#     for encoder in "${ENCODERS[@]}"; do
#         for clustering_method in "${CLUSTERING[@]}"; do
#             for batch_size in "${BATCH_SIZES[@]}"; do
#                 for proj_dim in "${EMBEDDING_DIMS[@]}"; do
#                     if [ "$clustering_method" = "dbscan" ]; then
#                         for eps in 0.3 0.7; do
#                             TRIAL_NAME="${DATE}_${dataset}_${encoder}_${clustering_method}_eps${eps}_b${batch_size}_dim${proj_dim}"
                            
#                             python3 train_unsupervised_CAGE.py \
#                                 --dataset $dataset \
#                                 --model CAGE \
#                                 --encoder_type $encoder \
#                                 --clustering_method $clustering_method \
#                                 --dbscan_ps $eps \
#                                 --dbscan_min_samples 5 \
#                                 --batch_size $batch_size \
#                                 --proj_dim $proj_dim \
#                                 --epochs 200 \
#                                 --window_width 128 \
#                                 --train_portion 1.0 \
#                                 --learning_rate 0.001 \
#                                 --weight_decay 1e-7 \
#                                 --momentum 0.9 \
#                                 --normalize \
#                                 --trial $TRIAL_NAME
#                         done

#                     elif [ "$clustering_method" = "birch" ]; then
#                         for threshold in 0.3 0.7; do
#                             TRIAL_NAME="${DATE}_${dataset}_${encoder}_${clustering_method}_th${threshold}_b${batch_size}_dim${proj_dim}"
                            
#                             python3 train_unsupervised_CAGE.py \
#                                 --dataset $dataset \
#                                 --model CAGE \
#                                 --encoder_type $encoder \
#                                 --clustering_method $clustering_method \
#                                 --birch_threshold $threshold \
#                                 --birch_branching_factor 50 \
#                                 --batch_size $batch_size \
#                                 --proj_dim $proj_dim \
#                                 --epochs 200 \
#                                 --window_width 128 \
#                                 --train_portion 1.0 \
#                                 --learning_rate 0.001 \
#                                 --weight_decay 1e-7 \
#                                 --momentum 0.9 \
#                                 --normalize \
#                                 --trial $TRIAL_NAME
#                         done
                    
#                     elif [ "$clustering_method" = "fastcluster" ]; then
#                         for linkage in "ward" "average"; do
#                             TRIAL_NAME="${DATE}_${dataset}_${encoder}_${clustering_method}_${linkage}_b${batch_size}_dim${proj_dim}"
                            
#                             python3 train_unsupervised_CAGE.py \
#                                 --dataset $dataset \
#                                 --model CAGE \
#                                 --encoder_type $encoder \
#                                 --clustering_method $clustering_method \
#                                 --fastcluster_linkage $linkage \
#                                 --batch_size $batch_size \
#                                 --proj_dim $proj_dim \
#                                 --epochs 200 \
#                                 --window_width 128 \
#                                 --train_portion 1.0 \
#                                 --learning_rate 0.001 \
#                                 --weight_decay 1e-7 \
#                                 --momentum 0.9 \
#                                 --normalize \
#                                 --trial $TRIAL_NAME
#                         done
                    
#                     else
#                         TRIAL_NAME="${DATE}_${dataset}_${encoder}_${clustering_method}_b${batch_size}_dim${proj_dim}"
                        
#                         python3 train_unsupervised_CAGE.py \
#                             --dataset $dataset \
#                             --model CAGE \
#                             --encoder_type $encoder \
#                             --clustering_method $clustering_method \
#                             --batch_size $batch_size \
#                             --proj_dim $proj_dim \
#                             --epochs 200 \
#                             --window_width 128 \
#                             --train_portion 1.0 \
#                             --learning_rate 0.001 \
#                             --weight_decay 1e-7 \
#                             --momentum 0.9 \
#                             --normalize \
#                             --trial $TRIAL_NAME
#                     fi
#                 done
#             done
#         done
#     done
# done

DATASETS=("MobiFall")
DATE=$(date +%Y%m%d)
BATCH_SIZE=64
PROJ_DIM=64

# 4. se with different clustering methods
# 3.1 SEencoder
# TRIAL_NAME="${DATE}_${DATASETS[0]}_se_kmeans_b${BATCH_SIZE}_dim${PROJ_DIM}"
# python3 train_unsupervised_CAGE.py \
#     --dataset ${DATASETS[0]} \
#     --model CAGE \
#     --encoder_type se \
#     --clustering_method kmeans \
#     --batch_size $BATCH_SIZE \
#     --proj_dim $PROJ_DIM \
#     --epochs 200 \
#     --window_width 128 \
#     --train_portion 1.0 \
#     --learning_rate 0.001 \
#     --weight_decay 1e-7 \
#     --momentum 0.9 \
#     --normalize \
#     --trial $TRIAL_NAME

# # 3.2 DBSCAN with different eps
# for eps in 0.3 0.7; do
#     TRIAL_NAME="${DATE}_${DATASETS[0]}_se_dbscan_eps${eps}_b${BATCH_SIZE}_dim${PROJ_DIM}"
#     python3 train_unsupervised_CAGE.py \
#         --dataset ${DATASETS[0]} \
#         --model CAGE \
#         --encoder_type se \
#         --clustering_method dbscan \
#         --dbscan_eps $eps \
#         --dbscan_min_samples 5 \
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

# # 3.3 BIRCH with different thresholds
# for threshold in 0.3 0.7; do
#     TRIAL_NAME="${DATE}_${DATASETS[0]}_se_birch_th${threshold}_b${BATCH_SIZE}_dim${PROJ_DIM}"
#     python3 train_unsupervised_CAGE.py \
#         --dataset ${DATASETS[0]} \
#         --model CAGE \
#         --encoder_type se \
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

# # 3.4 FastCluster with different linkages
# for linkage in "ward" "average"; do
#     TRIAL_NAME="${DATE}_${DATASETS[0]}_se_fastcluster_${linkage}_b${BATCH_SIZE}_dim${PROJ_DIM}"
#     python3 train_unsupervised_CAGE.py \
#         --dataset ${DATASETS[0]} \
#         --model CAGE \
#         --encoder_type se \
#         --clustering_method fastcluster \
#         --fastcluster_linkage $linkage \
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

# # 5. improved_default with DBSCAN
# # 5.1 DBSCAN with different eps
# # for eps in 0.3 0.7; do
#     TRIAL_NAME="${DATE}_${DATASETS[0]}_improved_default_dbscan_eps${eps}_b${BATCH_SIZE}_dim${PROJ_DIM}"
#     python3 train_unsupervised_CAGE.py \
#         --dataset ${DATASETS[0]} \
#         --model CAGE \
#         --encoder_type improved_default \
#         --clustering_method dbscan \
#         --dbscan_eps $eps \
#         --dbscan_min_samples 5 \
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

# 6. deep_default with DBSCAN
# 6.1 DBSCAN with different eps
for eps in 0.3 0.7; do
    TRIAL_NAME="${DATE}_${DATASETS[0]}_deep_default_dbscan_eps${eps}_b${BATCH_SIZE}_dim${PROJ_DIM}"
    python3 train_unsupervised_CAGE.py \
        --dataset ${DATASETS[0]} \
        --model CAGE \
        --encoder_type deep_default \
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