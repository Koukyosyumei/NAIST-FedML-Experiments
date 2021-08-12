## B. Pejo and G. Bicz ´ ok, “Quality inference in federated learning with secure aggregation,” ´ arXiv preprint arXiv:2007.06236, 2020

現在のラウンドのスコアの上昇幅が前回よりも良かった場合、今回サンプリングしたクライアントの信頼度を+1、前回サ　ンプリングしたクライアントの信頼度を-1する

## 実験設定

### データセット

    1. MNIST
    2. CIFAR-10

### Data Splits

    100ラウンド
    クライアント数は、5, 25, 100
    各ラウンドで、2, 5, 10人のクライアントが選ばれる


### フリーライダーの戦略

    すべての値がゼロの配列を、こう配として提出する