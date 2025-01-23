### Testing 2 (2025.01.20. ~ 2025.01.24.)

- **(1) experiment : for each embeddings (6-core i7 Intel / 32GB DDR4 / internal graphics (CPU))**

    - basic condition : encoder 1, default loss, learning rate 0.001, batch size 64, epoch 200, train data usage 0.6, No butterworth filter

| **Embedding Dimension** | **MobiFall** |  | **SisFall** |  | **UMAFall** | | **KFall** | |
|--------------------|----------|----------|----------|----------|----------|----------|----------|----------|
|  test score of best model | F1 Score | Accuracy | F1 Score | Accuracy | F1 Score | Accuracy | F1 Score | Accuracy |
| 8                         | 0.8989   |   90.22% |  0.7773  |  78.42%  | 0.8912   | 89.13%   |          |          |
| 16                        | 0.8946   |  89.82%  |  0.8085  |  81.09%  | 0.8543   | 86.03%   |          |          |
| 32                        |  0.8942  | 89.88%   |  0.7794  | 78.34%   | 0.8562   | 86.42%   |          |          |
| 64                        | 0.8822   |  88.73%  |  0.7891  |  79.20%  | 0.8816   | 88.23%   |          |          |
| 128                       | 0.8816   |  88.84%  | 0.7810   | 78.53%   | 0.9052   | 90.43%   |          |          |

- **(2) Confusion Matrix Analysis : Visualization of Confusion Matrices for Each Experiment**

| Embedding Dimension | MobiFall | SisFall | UMAFall | KFall |
|--------------------|----------|----------|---------|--------|
| 8                  | <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim8_ep200/confusion_matrix_heatmap.png">  |    <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim8_ep200/confusion_matrix_heatmap.png">   | <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim8_ep200/confusion_matrix_heatmap.png">  |  |
| 16                 |    <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim16_ep200/confusion_matrix_heatmap.png">    |     <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim16_ep200/confusion_matrix_heatmap.png">  | <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim16_ep200/confusion_matrix_heatmap.png">  |  |
| 32                 |    <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim32_ep200/confusion_matrix_heatmap.png">      |           <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim32_ep200/confusion_matrix_heatmap.png">            | <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim32_ep200/confusion_matrix_heatmap.png">  |  |
| 64                 |   <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim64_ep200/confusion_matrix_heatmap.png">      |      <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim64_ep200/confusion_matrix_heatmap.png">      | <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim64_ep200/confusion_matrix_heatmap.png">  |  |
| 128                |     <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim128_ep200/confusion_matrix_heatmap.png">     |             <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim128_ep200/confusion_matrix_heatmap.png">     | <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim128_ep200/confusion_matrix_heatmap.png">  |  |

- **(3) t-SNE Visualization of Embeddings: : t-SNE Visualization of Embedding Spaces for Each Experiment**

| Embedding Dimension | MobiFall | SisFall | UMAFall | KFall |
|--------------------|----------|----------|--------|--------|
| 8                  | <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim8_ep200/embedding_analysis/split_embeddings.png">  |    <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim8_ep200/embedding_analysis/split_embeddings.png">   |  <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim8_ep200/embedding_analysis/split_embeddings.png">   | |
| 16                 |    <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim16_ep200/embedding_analysis/split_embeddings.png">    |     <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim16_ep200/embedding_analysis/split_embeddings.png">   |  <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim16_ep200/embedding_analysis/split_embeddings.png">   | |
| 32                 |    <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim32_ep200/embedding_analysis/split_embeddings.png">      |           <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim32_ep200/embedding_analysis/split_embeddings.png">              |  <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim32_ep200/embedding_analysis/split_embeddings.png">   | |
| 64                 |   <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim64_ep200/embedding_analysis/split_embeddings.png">      |      <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim64_ep200/embedding_analysis/split_embeddings.png">      |  <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim64_ep200/embedding_analysis/split_embeddings.png">   | |
| 128                |     <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim128_ep200/embedding_analysis/split_embeddings.png">     |             <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim128_ep200/embedding_analysis/split_embeddings.png">     |  <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim128_ep200/embedding_analysis/split_embeddings.png">   | |

- **(4) SSL Loss Progression : Self-Supervised Learning Loss Curves Across Training**

| Embedding Dimension | MobiFall | SisFall | UMAFall | KFall |
|--------------------|----------|----------|--------|--------|
| 8                  | <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim8_ep200/training_progress.png">  |    <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim8_ep200/training_progress.png">    |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim8_ep200/training_progress.png">   | |
| 16                 |    <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim16_ep200/training_progress.png">    |     <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim16_ep200/training_progress.png">  |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim16_ep200/training_progress.png">   | |
| 32                 |    <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim32_ep200/training_progress.png">      |           <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim32_ep200/training_progress.png">             |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim32_ep200/training_progress.png">   | |
| 64                 |   <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim64_ep200/training_progress.png">      |      <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim64_ep200/training_progress.png">      |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim64_ep200/training_progress.png">   | |
| 128                |     <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim128_ep200/training_progress.png">     |             <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim128_ep200/training_progress.png">     |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim128_ep200/training_progress.png">   | |

- **(5) ROC Curve Analysis : Receiver Operating Characteristic (ROC) Curves for Each Experiment**

| Embedding Dimension | MobiFall | SisFall | UMAFall | KFall |
|--------------------|----------|----------|--------|--------|
| 8                  | <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim8_ep200/roc_curve.png">  |    <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim8_ep200/roc_curve.png">   |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim8_ep200/roc_curve.png">   | |
| 16                 |    <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim16_ep200/roc_curve.png">    |     <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim16_ep200/roc_curve.png">  |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim16_ep200/roc_curve.png">   | |
| 32                 |    <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim32_ep200/roc_curve.png">      |           <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim32_ep200/roc_curve.png">             |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim32_ep200/roc_curve.png">   | |
| 64                 |   <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim64_ep200/roc_curve.png">      |      <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim64_ep200/roc_curve.png">      |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim64_ep200/roc_curve.png">   | |
| 128                |     <img src="./20250120_MobiFall_enc1_default_lr0.001_b64_dim128_ep200/roc_curve.png">     |             <img src="./20250120_SisFall_enc1_default_lr0.001_b64_dim128_ep200/roc_curve.png">    |    <img src="./20250123_UMAFall_enc1_default_lr0.001_b64_dim128_ep200/roc_curve.png">   | |