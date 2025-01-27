### No classifier, just using embedding value -> KNN (neighbors=7)

<img src="./no-cls-KNN/embedding_analysis/split_embeddings.png">

```sh
Test Metrics:
Accuracy: 51.24%
F1 Score: 0.5090
Confusion Matrix:
[[250   0  36  21  43   0   6   5   8  25  17  26   1]
 [  5  85   0   0   0   0   0   0   0   0   1   0   0]
 [ 40   0   7  42   0   0   1   0   0   0   0   0   0]
 [ 70   0   4  15   0   0   0   0   0   0   2   4   0]
 [ 19   0   0   0  56   1   0  12   8   3   0   0   1]
 [  4   0   2   1   1  71   0   0   2   0   0   0   8]
 [ 10   0   0   0   0   0  66   0   1   0   2   1   0]
 [  9   0   0   1  64   4   0  22   0   4   0   0   0]
 [ 27   0   0   0   1   0   0   0  17   2   0   0   4]
 [ 16   0   0   0   0   0   0   1   1  21   1   2   2]
 [ 45   0   0   0   0   0   1   0   4   1  29   2   0]
 [ 42   0   2   3   0   0   1   1   0   0  14  85   0]
 [  0   0   1   0   0   0   0   0   0   0   0   0   0]]

```

### No classifier, just using embedding value -> SVM

<img src="./no-cls-SVM/embedding_analysis/split_embeddings.png">

```sh
Test Metrics:
Accuracy: 53.43%
F1 Score: 0.5064
Confusion Matrix:
[[315   0  30   2  44   0   3   1   7   0  15  21   0]
 [  6  85   0   0   0   0   0   0   0   0   0   0   0]
 [ 74   0  15   0   0   0   0   0   0   0   1   0   0]
 [ 88   0   0   6   0   0   0   0   0   0   0   1   0]
 [ 22   0   0   0  64   1   0   7   6   0   0   0   0]
 [  8   0   0   0   5  72   0   0   1   0   0   0   3]
 [ 13   0   0   0   0   0  64   0   0   0   3   0   0]
 [  6   0   0   0  86   0   0  12   0   0   0   0   0]
 [ 33   0   0   0   0   0   0   0  17   0   0   0   1]
 [ 34   0   0   0   0   0   1   2   2   0   1   0   4]
 [ 47   0   0   0   2   0   0   0   1   0  32   0   0]
 [ 68   0   1   0   0   0   0   0   0   0   6  73   0]
 [  1   0   0   0   0   0   0   0   0   0   0   0   0]]
```