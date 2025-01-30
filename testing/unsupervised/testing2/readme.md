### unsupervised : Testing 2 (2025.01.27. ~ 2025.01.31.)
#### MobiFall Dataset & Unsupervised_CAGE Model Performance Enhancement

- **Struggling log** : <a href="./struggling.md">HERE</a>
- **Conclusion**
    - **Point 1.** Quality embeddings are crucial for effective clustering (self-evident)
    - **Point 2.** If the similarity between positive pairs converges to zero asymptotically, performance will inevitably suffer (regardless of clustering method quality)-> Ultimately, generating high-quality embeddings is key.
    - **Point 3.** k-means appears to underperform. DBSCAN and fastcluster(ward) demonstrate superior accuracy and f1 scores.
    - **Point 4.** Among default encoder variations, results indicate that less complex and shallower encoder architectures form closer cosine similarities in the embedding space.

-----
### 1. Information of testing
#### (1) Common Parameters and Environment for All Experiments
- **Common Parameters**
    | Parameter | Value |
    |-----------|-------|
    | Dataset | MobiFall |
    | Batch Size | 64 |
    | Projection Dim | 64 |
    | Epochs | 200 |
    | Window Width | 128 |
    | Learning Rate | 0.001 |
    | Weight Decay | 1e-7 |
    | Momentum | 0.9 |
    | Train Portion | 1.0 |
    | Normalize | True |
- **Environment of learning model**
    - 6-core i7 Intel / 32GB DDR4 / internal graphics (CPU)

#### (2) Detailed Experiment Matrix

| Encoder Type | Clustering Method | Parameters | Trial Name Format |
|-------------|------------------|-------------|-------------------|
| default | kmeans | - | {DATE}_MobiFall_default_kmeans_b64_dim64 |
| default | DBSCAN | eps: [0.3, 0.7], min_samples: 5 | {DATE}_MobiFall_default_dbscan_eps{eps}_b64_dim64 |
| default | BIRCH | threshold: [0.3, 0.7], branching_factor: 50 | {DATE}_MobiFall_default_birch_th{threshold}_b64_dim64 |
| default | FastCluster | linkage: [ward, average] | {DATE}_MobiFall_default_fastcluster_{linkage}_b64_dim64 |
| transformer | kmeans | - | {DATE}_MobiFall_transformer_kmeans_b64_dim64 |
| transformer | DBSCAN | eps: [0.3, 0.7], min_samples: 5 | {DATE}_MobiFall_transformer_dbscan_eps{eps}_b64_dim64 |
| transformer | BIRCH | threshold: [0.3, 0.7], branching_factor: 50 | {DATE}_MobiFall_transformer_birch_th{threshold}_b64_dim64 |
| transformer | FastCluster | linkage: [ward, average] | {DATE}_MobiFall_transformer_fastcluster_{linkage}_b64_dim64 |
| resnet_transformer | kmeans | - | {DATE}_MobiFall_resnet_transformer_kmeans_b64_dim64 |
| resnet_transformer | DBSCAN | eps: [0.3, 0.7], min_samples: 5 | {DATE}_MobiFall_resnet_transformer_dbscan_eps{eps}_b64_dim64 |
| resnet_transformer | BIRCH | threshold: [0.3, 0.7], branching_factor: 50 | {DATE}_MobiFall_resnet_transformer_birch_th{threshold}_b64_dim64 |
| resnet_transformer | FastCluster | linkage: [ward, average] | {DATE}_MobiFall_resnet_transformer_fastcluster_{linkage}_b64_dim64 |
| se | kmeans | - | {DATE}_MobiFall_se_kmeans_b64_dim64 |
| se | DBSCAN | eps: [0.3, 0.7], min_samples: 5 | {DATE}_MobiFall_se_dbscan_eps{eps}_b64_dim64 |
| se | BIRCH | threshold: [0.3, 0.7], branching_factor: 50 | {DATE}_MobiFall_se_birch_th{threshold}_b64_dim64 |
| se | FastCluster | linkage: [ward, average] | {DATE}_MobiFall_se_fastcluster_{linkage}_b64_dim64 |
| improved_default | DBSCAN | eps: [0.7], min_samples: 5 | {DATE}_MobiFall_improved_default_dbscan_eps{eps}_b64_dim64 |
| deep_default | DBSCAN | eps: [0.7], min_samples: 5 | {DATE}_MobiFall_deep_default_dbscan_eps{eps}_b64_dim64 |


#### (3) Architecture of each models

* **Default**: Simple architecture + basic skip connection
* **Improved Default**: Multi-scale feature extraction + channel attention
* **Deep Default**: Deep hierarchical structure + local/global skip connection
* **SE Encoder**: Multi-scale + squeeze-excitation + progressive width
* **ResNet-Transformer**: Hybrid of ResNet and Transformer
* **Transformer**: Pure attention-based architecture

##### Default Encoder
```
Input → [Conv1D → MaxPool1D → Conv1D → MaxPool1D → Conv1D] * num_encoders → Global Average → Output
       ↑___________________Skip Connection (if i > 0)___________________↓
```

##### Improved Default Encoder
```
Input → Input Proj → BatchNorm → ReLU
       ↓
Multi-scale Conv (k=3,5) → Concat → Conv1D → BatchNorm → ReLU → MaxPool1D 
       ↓
Conv1D → BatchNorm → ReLU → MaxPool1D
       ↓
Channel Attention (GAP → Dense → ReLU → Dense → Sigmoid)
       ↓
Conv1D → BatchNorm → ReLU
       ↑___________________Skip Connection (if i > 0)___________________↓
       ↓
Global Average → Dense → Output
```

##### Deep Default Encoder
```
Input → Input Proj → BatchNorm → ReLU
       ↓
Block1: [Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm] + Local Skip
       ↓
MaxPool1D
       ↓
Block2: [Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → Conv1D → BatchNorm] + Local Skip
       ↓
MaxPool1D
       ↓
Block3: [Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → Conv1D → BatchNorm] + Local Skip
       ↓
Block4: [Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm]
       ↓
Channel Attention (GAP → Dense → ReLU → Dense → Sigmoid)
       ↑___________________Global Skip Connection (if i > 0)___________________↓
```

##### SE Encoder
```
Input → Input Proj → BatchNorm → ReLU
       ↓
Multi-scale Conv (k=3,5,7) → Concat → BatchNorm → ReLU → MaxPool1D
       ↓
Conv1D (2x channels) → BatchNorm → ReLU → MaxPool1D
       ↓
Conv1D (1x channels) → BatchNorm
       ↓
SE Block: GAP → Dense(reduction) → ReLU → Dense → Sigmoid
       ↓
Element-wise Multiplication
       ↑___________________Skip Connection (if i > 0)___________________↓
       ↓
Final BatchNorm → Global Average → Output
```

##### ResNet-Transformer Encoder
```
Input → Input Proj → BatchNorm → ReLU
       ↓
ResNet Blocks: [Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm] + Skip × 3
       ↓
Positional Encoding (Dense)
       ↓
Transformer Blocks: [Multi-Head Attention → Dropout → LayerNorm → FFN → LayerNorm] × 2
       ↓
Final LayerNorm → Global Average → Dense → Output
```

##### Transformer Encoder
```
Input → Input Proj → LayerNorm
       ↓
Sinusoidal Positional Encoding Addition
       ↓
Transformer Blocks: [
    Multi-Head Attention → Dropout → LayerNorm
    ↓
    FFN (Dense → GELU → Dense) → Dropout → LayerNorm
] × num_blocks
       ↓
Final LayerNorm → Global Average → Dense → Output
```

------

### 2. Results

> [!NOTE]
> **(0.6613, 76.37%) means, this indicates that all FALL instances were predicted as ADL, which means the model completely failed to perform proper predictions.** Consequently, these metrics are meaningless and do not provide any valuable insights into the model's performance. In other words, the model's classification is essentially equivalent to always predicting a single class, rendering the accuracy and F1 score useless as evaluation metrics.

| **Encoder Type** | **Clustering Method** | **Parameters** | **F1 Score** | **Accuracy** | **Confusion Matrix** |
|-----------------|----------------------|----------------|--------------|--------------|----------------|
| Default | DBSCAN | eps=0.3 | 0.8401 | 86.26% | <img src="./20250129_MobiFall_default_dbscan_eps0.3_b64_dim64/confusion_matrix_heatmap.png"> |
| Default | DBSCAN | eps=0.7 | 0.8735 | 88.73% | <img src="./20250129_MobiFall_default_dbscan_eps0.7_b64_dim64/confusion_matrix_heatmap.png"> |
| Default | BIRCH | threshold=0.3 | 0.7267 | 77.98% | <img src="./20250129_MobiFall_default_birch_th0.3_b64_dim64/confusion_matrix_heatmap.png"> |
| Default | BIRCH | threshold=0.7 | 0.7349 | 78.38% | <img src="./20250129_MobiFall_default_birch_th0.7_b64_dim64/confusion_matrix_heatmap.png"> |
| Default | FastCluster | linkage=ward | 0.7972 | 78.26% | <img src="./20250129_MobiFall_default_fastcluster_ward_b64_dim64/confusion_matrix_heatmap.png"> |
| Default | FastCluster | linkage=average | 0.6613 | 76.37% | <img src="./20250129_MobiFall_default_fastcluster_average_b64_dim64/confusion_matrix_heatmap.png"> |
| Default | K-means | - | 0.6613 | 76.37% | <img src="./20250129_MobiFall_default_kmeans_b64_dim64/confusion_matrix_heatmap.png"> |
| Transformer | DBSCAN | eps=0.3 | 0.6613 | 76.37% | <img src="./20250129_MobiFall_transformer_dbscan_eps0.3_b64_dim64/confusion_matrix_heatmap.png"> |
| Transformer | DBSCAN | eps=0.7 | 0.6613 | 76.37% | <img src="./20250129_MobiFall_transformer_dbscan_eps0.7_b64_dim64/confusion_matrix_heatmap.png"> |
| Transformer | BIRCH | threshold=0.3 | 0.6613 | 76.37% | <img src="./20250129_MobiFall_transformer_birch_th0.3_b64_dim64/confusion_matrix_heatmap.png"> |
| Transformer | BIRCH | threshold=0.7 | 0.6613 | 76.37% | <img src="./20250129_MobiFall_transformer_birch_th0.7_b64_dim64/confusion_matrix_heatmap.png"> |
| Transformer | FastCluster | linkage=ward | 0.6613 | 76.37% | <img src="./20250129_MobiFall_transformer_fastcluster_ward_b64_dim64/confusion_matrix_heatmap.png"> |
| Transformer | FastCluster | linkage=average | 0.6613 | 76.37% | <img src="./20250129_MobiFall_transformer_fastcluster_average_b64_dim64/confusion_matrix_heatmap.png"> |
| Transformer | K-means | - | 0.6613 | 76.37% | <img src="./20250129_MobiFall_transformer_kmeans_b64_dim64/confusion_matrix_heatmap.png"> |
| ResNet-Transformer | DBSCAN | eps=0.3 | 0.6613 | 76.37% | <img src="./20250130_MobiFall_resnet_transformer_dbscan_eps0.3_b64_dim64/confusion_matrix_heatmap.png"> |
| ResNet-Transformer | DBSCAN | eps=0.7 | 0.6613 | 76.37% | <img src="./20250130_MobiFall_resnet_transformer_dbscan_eps0.7_b64_dim64/confusion_matrix_heatmap.png"> |
| ResNet-Transformer | BIRCH | threshold=0.3 | 0.6613 | 76.37% | <img src="./20250130_MobiFall_resnet_transformer_birch_th0.3_b64_dim64/confusion_matrix_heatmap.png"> |
| ResNet-Transformer | BIRCH | threshold=0.7 | 0.6613 | 76.37% | <img src="./20250130_MobiFall_resnet_transformer_birch_th0.7_b64_dim64/confusion_matrix_heatmap.png"> |
| ResNet-Transformer | FastCluster | linkage=ward | 0.6613 | 76.37% | <img src="./20250130_MobiFall_resnet_transformer_fastcluster_ward_b64_dim64/confusion_matrix_heatmap.png"> |
| ResNet-Transformer | FastCluster | linkage=average | 0.8764 | 88.50% | <img src="./20250130_MobiFall_resnet_transformer_fastcluster_average_b64_dim64/confusion_matrix_heatmap.png"> |
| ResNet-Transformer | K-means | - | 0.6613 | 76.37% | <img src="./20250130_MobiFall_resnet_transformer_kmeans_b64_dim64/confusion_matrix_heatmap.png"> |
| SE | DBSCAN | eps=0.3 | 0.6613 | 76.37% | <img src="./20250131_MobiFall_se_dbscan_eps0.3_b64_dim64/confusion_matrix_heatmap.png"> |
| SE | DBSCAN | eps=0.7 | 0.6812 | 77.23% | <img src="./20250131_MobiFall_se_dbscan_eps0.7_b64_dim64/confusion_matrix_heatmap.png"> |
| SE | BIRCH | threshold=0.3 | 0.6613 | 76.37% | <img src="./20250131_MobiFall_se_birch_th0.3_b64_dim64/confusion_matrix_heatmap.png"> |
| SE | BIRCH | threshold=0.7 | 0.6613 | 76.37% | <img src="./20250131_MobiFall_se_birch_th0.7_b64_dim64/confusion_matrix_heatmap.png"> |
| SE | FastCluster | linkage=ward | 0.6613 | 76.37% | <img src="./20250131_MobiFall_se_fastcluster_ward_b64_dim64/confusion_matrix_heatmap.png"> |
| SE | FastCluster | linkage=average | 0.6613 | 76.37% | <img src="./20250131_MobiFall_se_fastcluster_average_b64_dim64/confusion_matrix_heatmap.png"> |
| SE | K-means | - | 0.6613 | 76.37% | <img src="./20250131_MobiFall_se_kmeans_b64_dim64/confusion_matrix_heatmap.png"> |
| Improved Default | DBSCAN | eps=0.7 | 0.6681 | 76.65% | <img src="./20250131_MobiFall_improved_default_dbscan_eps0.7_b64_dim64/confusion_matrix_heatmap.png"> |
| Deep Default | DBSCAN | eps=0.7 | 0.6613 | 76.37% | <img src="./20250131_MobiFall_deep_default_dbscan_eps0.7_b64_dim64/confusion_matrix_heatmap.png"> |

| **Encoder Type** | **Class-wise Cosine Similarities** | **PCA of test data points** |
|-----------------|----------------------|----|
| Default | <img src="./20250129_MobiFall_default_kmeans_b64_dim64/embedding_analysis/class_similarities.png"> | <img src="./20250129_MobiFall_default_kmeans_b64_dim64/embeddings_pca.png">|
| Transformer |  <img src="./20250129_MobiFall_transformer_kmeans_b64_dim64/embedding_analysis/class_similarities.png">| <img src="./20250129_MobiFall_transformer_kmeans_b64_dim64/embeddings_pca.png">|
| ResNet-Transformer | <img src="./20250130_MobiFall_resnet_transformer_kmeans_b64_dim64/embedding_analysis/class_similarities.png"> | <img src="./20250130_MobiFall_resnet_transformer_kmeans_b64_dim64/embeddings_pca.png">|
| SE | <img src="./20250131_MobiFall_se_kmeans_b64_dim64/embedding_analysis/class_similarities.png"> | <img src="./20250131_MobiFall_se_kmeans_b64_dim64/embeddings_pca.png"> |
| Improved Default | <img src="./20250131_MobiFall_improved_default_dbscan_eps0.7_b64_dim64/embedding_analysis/class_similarities.png"> | <img src="./20250131_MobiFall_improved_default_dbscan_eps0.7_b64_dim64/embeddings_pca.png"> |
| Deep Default | <img src="./20250131_MobiFall_deep_default_dbscan_eps0.7_b64_dim64/embedding_analysis/class_similarities.png"> | <img src="./20250131_MobiFall_deep_default_dbscan_eps0.7_b64_dim64/embeddings_pca.png"> |
