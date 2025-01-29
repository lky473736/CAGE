#!/bin/bash

DATASETS=("MobiFall")
ENCODERS=("default" "transformer" "resnet_transformer")
CLUSTERING=("kmeans" "dbscan" "birch" "fastcluster")
BATCH_SIZES=(64)
EMBEDDING_DIMS=(64)
DATE=$(date +%Y%m%d)

for dataset in "${DATASETS[@]}"; do
    for encoder in "${ENCODERS[@]}"; do
        for clustering_method in "${CLUSTERING[@]}"; do
            for batch_size in "${BATCH_SIZES[@]}"; do
                for proj_dim in "${EMBEDDING_DIMS[@]}"; do
                    if [ "$clustering_method" = "dbscan" ]; then
                        # DBSCAN - eps만 변경
                        for eps in 0.3 0.7; do
                            TRIAL_NAME="${DATE}_${dataset}_${encoder}_${clustering_method}_eps${eps}_b${batch_size}_dim${proj_dim}"
                            
                            python3 train_unsupervised_CAGE.py \
                                --dataset $dataset \
                                --model CAGE \
                                --encoder_type $encoder \
                                --clustering_method $clustering_method \
                                --dbscan_eps $eps \
                                --dbscan_min_samples 5 \
                                --batch_size $batch_size \
                                --proj_dim $proj_dim \
                                --epochs 200 \
                                --window_width 128 \
                                --train_portion 1.0 \
                                --learning_rate 0.001 \
                                --weight_decay 1e-7 \
                                --momentum 0.9 \
                                --normalize \
                                --trial $TRIAL_NAME
                        done

                    elif [ "$clustering_method" = "birch" ]; then
                        # BIRCH - threshold만 변경
                        for threshold in 0.3 0.7; do
                            TRIAL_NAME="${DATE}_${dataset}_${encoder}_${clustering_method}_th${threshold}_b${batch_size}_dim${proj_dim}"
                            
                            python3 train_unsupervised_CAGE.py \
                                --dataset $dataset \
                                --model CAGE \
                                --encoder_type $encoder \
                                --clustering_method $clustering_method \
                                --birch_threshold $threshold \
                                --birch_branching_factor 50 \
                                --batch_size $batch_size \
                                --proj_dim $proj_dim \
                                --epochs 200 \
                                --window_width 128 \
                                --train_portion 1.0 \
                                --learning_rate 0.001 \
                                --weight_decay 1e-7 \
                                --momentum 0.9 \
                                --normalize \
                                --trial $TRIAL_NAME
                        done
                    
                    elif [ "$clustering_method" = "fastcluster" ]; then
                        # fastcluster - ward와 average만
                        for linkage in "ward" "average"; do
                            TRIAL_NAME="${DATE}_${dataset}_${encoder}_${clustering_method}_${linkage}_b${batch_size}_dim${proj_dim}"
                            
                            python3 train_unsupervised_CAGE.py \
                                --dataset $dataset \
                                --model CAGE \
                                --encoder_type $encoder \
                                --clustering_method $clustering_method \
                                --fastcluster_linkage $linkage \
                                --batch_size $batch_size \
                                --proj_dim $proj_dim \
                                --epochs 200 \
                                --window_width 128 \
                                --train_portion 1.0 \
                                --learning_rate 0.001 \
                                --weight_decay 1e-7 \
                                --momentum 0.9 \
                                --normalize \
                                --trial $TRIAL_NAME
                        done
                    
                    else
                        TRIAL_NAME="${DATE}_${dataset}_${encoder}_${clustering_method}_b${batch_size}_dim${proj_dim}"
                        
                        python3 train_unsupervised_CAGE.py \
                            --dataset $dataset \
                            --model CAGE \
                            --encoder_type $encoder \
                            --clustering_method $clustering_method \
                            --batch_size $batch_size \
                            --proj_dim $proj_dim \
                            --epochs 200 \
                            --window_width 128 \
                            --train_portion 1.0 \
                            --learning_rate 0.001 \
                            --weight_decay 1e-7 \
                            --momentum 0.9 \
                            --normalize \
                            --trial $TRIAL_NAME
                    fi
                done
            done
        done
    done
done