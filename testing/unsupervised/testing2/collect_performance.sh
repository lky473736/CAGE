#!/bin/bash

# Initialize the output file
output_file="test_metrics_summary.md"
echo "| **Encoder Type** | **Clustering Method** | **Parameters** | **F1 Score** | **Accuracy** |" > "$output_file"
echo "|-----------------|----------------------|----------------|--------------|--------------|" >> "$output_file"

# Function to extract metrics from a result file
extract_metrics() {
    local dir="$1"
    local encoder_type="$2"
    local clustering_method="$3"
    local params="$4"
    
    # Find the result file
    result_file=$(find "$dir" -name "result")
    
    if [ -f "$result_file" ]; then
        # Extract Test Metrics more robustly
        accuracy=$(awk '/Test Metrics:/{f=1} f && /Accuracy:/{print $2; exit}' "$result_file")
        f1_score=$(awk '/Test Metrics:/{f=1} f && /F1 Score:/{print $3; exit}' "$result_file")
        
        # Append to output file
        echo "| $encoder_type | $clustering_method | $params | $f1_score | $accuracy |" >> "$output_file"
    else
        echo "| $encoder_type | $clustering_method | $params | 0.xxxx | xx.xx% |" >> "$output_file"
    fi
}


# Default Encoder
extract_metrics "20250129_MobiFall_default_dbscan_eps0.3_b64_dim64" "Default" "DBSCAN" "eps=0.3"
extract_metrics "20250129_MobiFall_default_dbscan_eps0.7_b64_dim64" "Default" "DBSCAN" "eps=0.7"
extract_metrics "20250129_MobiFall_default_birch_th0.3_b64_dim64" "Default" "BIRCH" "threshold=0.3"
extract_metrics "20250129_MobiFall_default_birch_th0.7_b64_dim64" "Default" "BIRCH" "threshold=0.7"
extract_metrics "20250129_MobiFall_default_fastcluster_ward_b64_dim64" "Default" "FastCluster" "linkage=ward"
extract_metrics "20250129_MobiFall_default_fastcluster_average_b64_dim64" "Default" "FastCluster" "linkage=average"
extract_metrics "20250129_MobiFall_default_kmeans_b64_dim64" "Default" "K-means" "-"

# Transformer Encoder
extract_metrics "20250129_MobiFall_transformer_dbscan_eps0.3_b64_dim64" "Transformer" "DBSCAN" "eps=0.3"
extract_metrics "20250129_MobiFall_transformer_dbscan_eps0.7_b64_dim64" "Transformer" "DBSCAN" "eps=0.7"
extract_metrics "20250129_MobiFall_transformer_birch_th0.3_b64_dim64" "Transformer" "BIRCH" "threshold=0.3"
extract_metrics "20250129_MobiFall_transformer_birch_th0.7_b64_dim64" "Transformer" "BIRCH" "threshold=0.7"
extract_metrics "20250129_MobiFall_transformer_fastcluster_ward_b64_dim64" "Transformer" "FastCluster" "linkage=ward"
extract_metrics "20250129_MobiFall_transformer_fastcluster_average_b64_dim64" "Transformer" "FastCluster" "linkage=average"
extract_metrics "20250129_MobiFall_transformer_kmeans_b64_dim64" "Transformer" "K-means" "-"

# ResNet-Transformer Encoder
extract_metrics "20250130_MobiFall_resnet_transformer_dbscan_eps0.3_b64_dim64" "ResNet-Transformer" "DBSCAN" "eps=0.3"
extract_metrics "20250130_MobiFall_resnet_transformer_dbscan_eps0.7_b64_dim64" "ResNet-Transformer" "DBSCAN" "eps=0.7"
extract_metrics "20250130_MobiFall_resnet_transformer_birch_th0.3_b64_dim64" "ResNet-Transformer" "BIRCH" "threshold=0.3"
extract_metrics "20250130_MobiFall_resnet_transformer_birch_th0.7_b64_dim64" "ResNet-Transformer" "BIRCH" "threshold=0.7"
extract_metrics "20250130_MobiFall_resnet_transformer_fastcluster_ward_b64_dim64" "ResNet-Transformer" "FastCluster" "linkage=ward"
extract_metrics "20250130_MobiFall_resnet_transformer_fastcluster_average_b64_dim64" "ResNet-Transformer" "FastCluster" "linkage=average"
extract_metrics "20250130_MobiFall_resnet_transformer_kmeans_b64_dim64" "ResNet-Transformer" "K-means" "-"

# SE Encoder
extract_metrics "20250131_MobiFall_se_dbscan_eps0.3_b64_dim64" "SE" "DBSCAN" "eps=0.3"
extract_metrics "20250131_MobiFall_se_dbscan_eps0.7_b64_dim64" "SE" "DBSCAN" "eps=0.7"
extract_metrics "20250131_MobiFall_se_birch_th0.3_b64_dim64" "SE" "BIRCH" "threshold=0.3"
extract_metrics "20250131_MobiFall_se_birch_th0.7_b64_dim64" "SE" "BIRCH" "threshold=0.7"
extract_metrics "20250131_MobiFall_se_fastcluster_ward_b64_dim64" "SE" "FastCluster" "linkage=ward"
extract_metrics "20250131_MobiFall_se_fastcluster_average_b64_dim64" "SE" "FastCluster" "linkage=average"
extract_metrics "20250131_MobiFall_se_kmeans_b64_dim64" "SE" "K-means" "-"

# Additional models
extract_metrics "20250131_MobiFall_improved_default_dbscan_eps0.7_b64_dim64" "Improved Default" "DBSCAN" "eps=0.7"
extract_metrics "20250131_MobiFall_deep_default_dbscan_eps0.7_b64_dim64" "Deep Default" "DBSCAN" "eps=0.7"

echo "Test metrics summary saved to $output_file"