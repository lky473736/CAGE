### using two loss at unsupervised-CAGE (no cls, using KNN) 
#### nt_xent_loss (vanilla)
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

#### triplet loss (vanilla)
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
- nt_xent_loss에서 temperature을 학습하면서 변하게끔 하기 (최적의 값으로 수렴되게끔)
    - 학습 초기에는 높은 temperature로 시작해서 전반적인 특징 학습하게 하기 -> 시간 지나면서 temperature 낮추기 -> 더 discriminative한 특징 학습하게 하기
- https://velog.io/@iissaacc/Triplet-Loss
- https://medium.com/analytics-vidhya/triplet-loss-b9da35be21b8
- https://dilithjay.com/blog/nt-xent-loss-explained
    - 여기에서 나오는 hard negative를 도전해볼 수 있겠음

#### nt_xent_loss (learnable parameter)
<img src="./unsupervised_nt_xent_improved_KNN/embedding_analysis/split_embeddings.png">

```sh
Test Metrics :
Accuracy : 49.40%
F1 Score : 0.4848
Confusion Matrix :
[[245   3  49  35  28   0   2   4   7  24  16  24   1]
 [  1  87   1   0   0   0   1   0   0   0   1   0   0]
 [ 40   0  10  38   0   0   0   0   0   0   2   0   0]
 [ 77   0   3  12   0   0   0   0   0   0   2   1   0]
 [ 41   0   0   0  45   0   0  11   2   0   1   0   0]
 [  3   0   1   2   0  78   0   0   1   0   0   0   4]
 [  3   0   0   1   0   0  69   0   1   1   2   3   0]
 [ 30   0   0   1  63   0   0   6   0   1   0   0   3]
 [ 18   0   0   2   0   0   0   1  25   5   0   0   0]
 [ 13   0   0   0   0   0   0   1   0  28   1   0   1]
 [ 48   0   0   1   0   0   3   0   4   1  22   2   1]
 [ 67   0   2   2   1   0   2   0   1   0   2  71   0]
 [  1   0   0   0   0   0   0   0   0   0   0   0   0]]
 ```

 #### triplet loss (hard negative)
<img src="./unsupervised_triplet_improved_KNN/embedding_analysis/split_embeddings.png">

```sh
Test Metrics :
Accuracy : 44.09%
F1 Score : 0.4262
Confusion Matrix :
[[240   5  21  29  55   1  12  17   5  18  17  17   1]
 [  2  87   0   0   0   0   0   1   1   0   0   0   0]
 [ 55   0  10   4   0   0   9   8   0   1   1   2   0]
 [ 28   0  12  32   2   0   0   2   3  11   1   3   1]
 [ 24   0   0   1  57   0   0  14   3   1   0   0   0]
 [ 31   2   1   1   1  30   0   6   0   1   1  12   3]
 [  5   0   0   0   0   0  67   0   0   1   3   4   0]
 [ 41   0   1   0  40   1   0  18   3   0   0   0   0]
 [ 28   0   1   0   8   0   0   0  12   1   0   1   0]
 [ 27   0   0   0   2   3   0   3   0   6   0   3   0]
 [ 42   2   0   0   9   0   4  13   4   1   5   2   0]
 [ 48   1   9  10   5   1   3   8   0   2   2  59   0]
 [  1   0   0   0   0   0   0   0   0   0   0   0   0]]
 ```

<b>전이랑 달라진게 없다.</b>