# 関連論文

クライアントの質を推定する手法

## 手法の種類

### A: Test /Self-Reported Based Contribution Evaluation

各クライアントにデータ数やローカルでの性能を報告させたり、クライアントが送ってきたモデルをサーバー側の検証データで検証する。

### B: Marginal Loss Based Contribution Evaluation

あるクライアントの質を、そのクライアントがいる場合とそうでない場合のグローバルモデルの性能の差として定義する

### C: Similarity Based Contribution Evaluation

クライアント間の類似度や、サーバーが持つ検証データとクライアントが持つデータの類似度を測定し、外れ値を検出する


### D: Parameter checking Based Contribution Evaluation

クライアントが送ってきたパラメータやgradientを、AutoEncoderなどの外れ値を検出するアルゴリズムにかけて、異常なクライアントを検出する

## 実験の種類

    1. ラベルが間違ったデータの混入
    2. データの一部にノイズをのせる
    3. Free-Rider (ランダムもしくは人工的に生成したパラメータをサーバーに送る)
    4. Data Poisoning (モデルの性能が下がるように設計したパラメータをサーバーに送る)

## 比較

| 手法  |    タイプ     | 悪意のあるクライアント | 補助データ | 中央サーバー |                                  URL                                  |
|:-----:|:----------:|:-------------:|:-----:|:------:|:---------------------------------------------------------------------:|
| RFFL  | Similarity |       〇       |  不要   |   必要   |            [here](https://arxiv.org/pdf/2011.10464v2.pdf)             |
| FOCUS | Similarity |       ×       |  必要   |   必要   | [here](https://link.springer.com/chapter/10.1007/978-3-030-63076-8_8) |
|FedProf|Similarity|×|必要|必要| [here](https://arxiv.org/abs/2102.01733) |
|FPPDL|Similarity|||||
|QI|Marginal Loss|〇|不要|必要| [here](https://arxiv.org/abs/2007.06236) |
|F-RICE|Marginal Loss|〇|不要|必要| [here](https://ieeexplore.ieee.org/document/9425266)|
|Shapely Value|Marginal Loss||||
|FAIR|Test /Self-Reported|〇|不要|必要| [here](https://ieeexplore.ieee.org/document/9488743)|
|FairFed|Test /Self-Reported|〇|不要|不要| [here](https://ieeexplore.ieee.org/document/9425266) |
|STD_DAGMM|Parameter checking|〇|不要|必要| [here](https://arxiv.org/abs/1911.12560) |
|Spectral Anomaly Detection|Parameter checking|〇|不要|必要| [here](https://arxiv.org/abs/2002.00211) |