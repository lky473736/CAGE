| **Encoder Type** | **Clustering Method** | **Parameters** | **F1 Score** | **Accuracy** |
|-----------------|----------------------|----------------|--------------|--------------|
| Default | DBSCAN | eps=0.3 | 0.8401 | 86.26% |
| Default | DBSCAN | eps=0.7 | 0.8735 | 88.73% |
| Default | BIRCH | threshold=0.3 | 0.7267 | 77.98% |
| Default | BIRCH | threshold=0.7 | 0.7349 | 78.38% |
| Default | FastCluster | linkage=ward | 0.7972 | 78.26% |
| Default | FastCluster | linkage=average | 0.6613 | 76.37% |
| Default | K-means | - | 0.6613 | 76.37% |
| Transformer | DBSCAN | eps=0.3 | 0.6613 | 76.37% |
| Transformer | DBSCAN | eps=0.7 | 0.6613 | 76.37% |
| Transformer | BIRCH | threshold=0.3 | 0.6613 | 76.37% |
| Transformer | BIRCH | threshold=0.7 | 0.6613 | 76.37% |
| Transformer | FastCluster | linkage=ward | 0.6613 | 76.37% |
| Transformer | FastCluster | linkage=average | 0.6613 | 76.37% |
| Transformer | K-means | - | 0.6613 | 76.37% |
| ResNet-Transformer | DBSCAN | eps=0.3 | 0.6613 | 76.37% |
| ResNet-Transformer | DBSCAN | eps=0.7 | 0.6613 | 76.37% |
| ResNet-Transformer | BIRCH | threshold=0.3 | 0.6613 | 76.37% |
| ResNet-Transformer | BIRCH | threshold=0.7 | 0.6613 | 76.37% |
| ResNet-Transformer | FastCluster | linkage=ward | 0.6613 | 76.37% |
| ResNet-Transformer | FastCluster | linkage=average | 0.8764 | 88.50% |
| ResNet-Transformer | K-means | - | 0.6613 | 76.37% |
| SE | DBSCAN | eps=0.3 | 0.6613 | 76.37% |
| SE | DBSCAN | eps=0.7 | 0.6812 | 77.23% |
| SE | BIRCH | threshold=0.3 | 0.6613 | 76.37% |
| SE | BIRCH | threshold=0.7 | 0.6613 | 76.37% |
| SE | FastCluster | linkage=ward | 0.6613 | 76.37% |
| SE | FastCluster | linkage=average | 0.6613 | 76.37% |
| SE | K-means | - | 0.6613 | 76.37% |
| Improved Default | DBSCAN | eps=0.7 | 0.6681 | 76.65% |
| Deep Default | DBSCAN | eps=0.7 | 0.6613 | 76.37% |
