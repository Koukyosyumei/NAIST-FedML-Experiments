## Xinyi Xu and Lingjuan Lyu. A Reputation Mechanism Is All You Need: Collaborative Fairness and Adversarial Robustness in Federated Learning. FL-ICML'21.

個々のクライアントが送ってきたgradientと、集計したgradientのコサイン類似度をクライアントの信頼度とし、信頼度が閾値以下のクライアントを排除していく。

## 実験設定

### DataSet

    1. MNIST
    2. CIFAR-10

### Data splits

    1. standard I.I.D data sampling
    2. POW (powerlaw split ... 指数的にデータを配分)
    3. CLA (classimbalance split .. ラベルを不均衡にする)
        例. participants-{1, 2, 3, 4, 5}が持つラベルの種類数が、{1, 3, 5, 7, 10}
    4. 正常なクライアント数は5, 10, 20のいずれか

### フリーライダーの戦略

    1. [-1, 1]の一様分布からランダムサンプリングしたものをこう配として提出する
    2. フリーライダーの数は、正常なクライアントの数の20または110%

