### STRUGGLING 1/29 : Experiment Summary - MobiFall Dataset & Unsupervised_CAGE Model Performance Enhancement

Earlier this morning, I conducted a large number of parameter combinations and model experiments to improve the performance of the existing **unsupervised_CAGE** using the MobiFall dataset. However, the overall performance did not meet expectations, so I did not organize the log and weight files. That said, during the experiments, I discovered some very meaningful metrics, and I now understand why the model performance was suboptimal. I will summarize the entire experiment flow and outline future directions. Currently, I am re-structuring the experiments I conducted this morning and will upload them to **testing2**.

Before the experiments aimed at improving performance, I made the following code modifications. These were changes I added dynamically during the experiment and will be cleaned up later:

- **(1)** In the existing `train_unsupervised_CAGE.py` file, which was used for training, evaluation, and saving visualizations, I added several new visualizations. To visualize the similarity between embeddings, I created a **class_similarities.png** using a heatmap (the higher the diagonal, the better the positive pair similarity, which is a good phenomenon). I also added **similarity_analysis.png** to examine the distribution of similarities within the same class and across different classes, as well as the relationship between **Euclidean distance** and **cosine similarity**. Ideally, the pattern should be that as distance increases, similarity decreases. In **embedding_similarity_relation.png**, I visualized the embeddings with t-SNE and showed the relationship between t-SNE distance, original similarity, and sample connections. As expected, the data points related to **FALL** and **ADL** were well-separated, indicating that unsupervised learning is working well.

- **(2)** The original `unsupervised_CAGE.py` file, which was based on the Default Encoder, was modified by adding several new encoders, and the file was renamed to **embedding_CAGE.py**. The additional encoders included **SE-Encoder**, **Transformer-Encoder**, **ResNet-Transformer**, and **U-Net**.

- **(3)** In addition to K-means, I applied several clustering techniques such as **DBSCAN**, **spectral**, **GMM**, **FastCluster**, and **Birch**. **Spectral** clustering was later removed. Please refer to the following for details.

---

### Experiment Details:

- The base parameters for the experiments were as follows:
  - **batch_size**: 64
  - **proj_dim**: 64
  - **epochs**: 200
  - **window_width**: 128
  - **train_portion**: 1.0
  - **learning_rate**: 0.001
  - **weight_decay**: 1e-7

---

**(1)** In the first experiment, I tested the K-means, DBSCAN, GMM, and spectral clustering methods with the default encoder. DBSCAN was tested with different combinations of epsilon and min_cluster values: (0.3, 0.5, 0.7) and (3, 5, 7).

- Honestly, I did not expect the **default encoder** to perform well in terms of pair similarity, but I was surprised to see high diagonal values in **class_similarities.png**.
- Upon reviewing the **confusion matrix**, I noticed that K-means predicted all **FALL** points as **ADL** (the worst case). However, DBSCAN provided a reasonably good accuracy (around 90%) and predicted **FALL** quite well. GMM did not provide accurate predictions, similar to K-means.
- When applying **spectral clustering**, the process stalled for 2 hours. Since the time complexity of spectral clustering is **O(n^3)**, it performs well but takes too long, so I ended the experiment and looked for alternative clustering techniques. **FastCluster** and **Birch** were found and are included in the ongoing experiment.

---

**(2)** In the second experiment, I fixed the **K-means** clustering method and changed the encoder. The encoders tested were **default**, **default (enhanced version)**, **transformer**, **resnet-transformer**, and **SE-Encoder**.

- A notable observation here was that with the **default encoder**, pair similarity was relatively good, but with the **transformer** and **resnet-transformer**, the cosine similarity was extremely poor. While the (1, 1) pair for **FALL** predicted as **FALL** showed strong similarity, the (0, 0) pair for **ADL** predicted as **ADL** showed negative similarity. Negative similarities were also found in other negative pairs.
- The **SE-Encoder** showed average similarity, as it is based on the **default encoder** with an SE block.
- Even when I increased the parameters or made the architecture deeper for the **default encoder**, the similarity values did not change significantly. Additionally, the **SSL accuracy** stopped at a low value.
- Even more interestingly, the **confusion matrix** showed that all experiments were biased towards predicting **FALL** as **ADL**.
  
---

### Insights Gained:

- No matter how good the architecture is, if it cannot capture similarity properly or make accurate predictions, it is not effective.
- Currently, the best-performing setup is **default encoder + DBSCAN**.
  - I should consider enhancing the **default encoder** (e.g., adding an **SE block** or **residual block**).
  - How can we increase **SSL accuracy** to over 90% while improving cosine similarity for positive pairs?
  - Is **K-means** really the best clustering method?

---

### Future Experiments:

- **(1)** Test **DBSCAN** with different encoder configurations (e.g., **default**, **default + SE**, **default*2 + skip-connection**, etc.).
- **(2)** Consider a method that utilizes both **cosine similarity** and **embedding** for classification.