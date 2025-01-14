### testing 6
- unsupervised에서는 임베딩값으로만 분류하게끔 하여 cosine distance로 KNN
    - 근데 encoder를 많이 두면 성능이 더 좋아지지 않을까 싶음 (더 정보 압축을 하니깐 마지막엔 유용한 정보로 임베딩 구성)
    - encoder를 직렬적으로 많이 두면 당연히 정보 손실이 발생되니깐, skip connection block을 두기
        - 그래서 argument에 --no_skip, --use_skip 둘거임. skip conection을 쓰지 않으면 정보손실로 결과가 안좋게 나오지 않을까 싶음 (아마도)
    - argument에 --num_encoders 두어서 encoder 갯수 사용자에게 받기
- 실험
    - (1) encoder 수 : 1개(원본), 2개, 3개, 10개
    - (2) skip : no_skip, use_skip

- <b>결과</b>
    - | model | acc(test) | f1(test) |
        |----------|---------|--------|
        | encoder 1 | Accuracy: 51.24% | F1 Score: 0.5090 |
        | encoder 2 (no skip) | Accuracy: 51.45% | F1 Score: 0.4817 |
        | encoder 2 (use skip) | Accuracy: 52.23% | F1 Score: 0.4659 |
        | encoder 3 (no skip) | Accuracy: 40.13% | F1 Score: 0.3510 |
        | encoder 3 (use skip) | Accuracy: 50.18% | F1 Score: 0.4532 |
        | encoder 10 (use skip) | Accuracy: 50.04% | F1 Score: 0.4551 |
    - 실패 
        - encoder로 두 센서간의 관계는 잘 학습해도, 뭔가 분류에 필요한 discriminative feature를 파악을 못할 수도 있다고 생각함
            - 아무리 encoder가 늘어나도 acc가 거의 50%에 수렴 (use skip 모델만 보면)
        - skip connection의 중요성
        - 그러면 원래 논문에서도 임베딩은 준비물(압축정보)의 개념이고, classifier에서 성능을 끌어올린거 같은 느낌
            - embedding과 cls는 따로 봐야 하는 건가? 
        - 아니면, KNN이 너무 단순해서 그런가?
            - 임베딩은 너무 잘 되었는데, KNN이 제대로 된 분류를 못해서 정확도가 안나왔을 수도 있음