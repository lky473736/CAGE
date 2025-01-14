### using two loss at unsupervised-CAGE (no cls, using KNN) 
#### nt_xent_loss
<img src="./unsupervised_nt_xent_KNN/embedding_analysis/split_embeddings.png">

```sh
Test Metrics :
Accuracy : 49.54%
F1 Score : 0.4813
Confusion Matrix :
[[255   4  27  36  35   0   2   5  14  20  11  29   0]
 [  1  89   0   0   0   0   0   0   0   0   1   0   0]
 [ 56   0   2  26   0   0   1   0   0   0   5   0   0]
 [ 65   0   4  21   0   0   0   0   0   0   2   3   0]
 [ 38   0   0   1  44   0   0  10   6   0   1   0   0]
 [  4   0   1   1   0  71   0   0   0   0   0   0  12]
 [  4   0   0   0   0   0  69   0   2   0   2   3   0]
 [ 12   0   0   1  82   1   0   5   1   0   0   0   2]
 [ 17   0   0   2   1   0   0   0  29   1   0   0   1]
 [ 15   0   0   1   0   0   0   0   0  27   0   0   1]
 [ 56   0   1   0   0   0   0   1   8   0  16   0   0]
 [ 61   0   4   3   0   0   0   0   0   0   8  72   0]
 [  1   0   0   0   0   0   0   0   0   0   0   0   0]]
```

#### triplet loss (maybe easy)
<img src="./unsupervised_triplet_KNN/embedding_analysis/split_embeddings.png">

```sh
Test Metrics :
Accuracy : 48.48%
F1 Score : 0.4694
Confusion Matrix :
[[226   0  22  52  39   2  14  17  13  20  20  12   1]
 [  2  85   0   1   1   0   0   0   2   0   0   0   0]
 [ 70   0  13   1   0   0   0   3   0   0   0   3   0]
 [ 40   0   7  29   0  14   0   1   0   3   0   1   0]
 [ 27   0   0   1  68   0   0   0   3   1   0   0   0]
 [ 24   0   0   1   0  36   0  18   9   0   0   0   1]
 [  1   0   1   0   0   0  55   0   0   0   0  23   0]
 [ 10   0   0   1   0   1   0  91   0   1   0   0   0]
 [ 18   0   1   0   7   0   0   1  21   2   0   0   1]
 [ 25   0   4   0   1   0   0   7   0   5   1   1   0]
 [ 42   0   3   3   1   3  11   2   6   1   4   6   0]
 [ 60   0  10   4   0   2   7   4   1   7   1  52   0]
 [  1   0   0   0   0   0   0   0   0   0   0   0   0]]
 ```

#### 개선 포인트 아이디어
- nt_xent_loss랑 triplet loss를 결합하는 방식?
- https://velog.io/@iissaacc/Triplet-Loss
    - https://medium.com/analytics-vidhya/triplet-loss-b9da35be21b8
    - <b>여기에서 나오는 hard negative나 semi hard negative를 도전해볼 수 있겠음</b>

#### triplet loss (hard negative)

#### triplet loss (semi hard negative)
