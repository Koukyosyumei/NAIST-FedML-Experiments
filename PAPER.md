# 関連論文

クライアントの質を推定する手法

## 手法の種類

### A: Test /Self-Reported Based Contribution Evaluation

各クライアントにデータ数やローカルでの性能を報告させたり、クライアントが送ってきたモデルをサーバー側の検証データで検証する。

### B: Marginal Loss Based Contribution Evaluation

あるクライアントの質を、そのクライアントがいる場合とそうでない場合のグローバルモデルの性能の差として定義する

### C: Similarity Based Contribution Evaluation

クライアント間の類似度や、サーバーが持つ検証データとクライアントが持つデータの類似度を測定し、外れ値を検出する

### D: ML Based Contribution Evaluation

クライアントが送ってきたパラメータやgradientを、AutoEncoderなどの外れ値を検出する機械学習アルゴリズムにかけて、異常なクライアントを検出する

### E. Blackchain Based Contribution Evaluation

## 実験の種類

    1. ラベルが間違ったデータの混入
    2. データの一部にノイズをのせる
    3. データが極端に偏っている
    4. Free-Rider (ランダムもしくは人工的に生成したパラメータをサーバーに送る)
    5. Data Poisoning (モデルの性能が下がるように設計したパラメータをサーバーに送る)

## 比較

| 手法  |    タイプ     | 悪意のあるクライアント | 補助データ | 中央サーバー | Free-Riderを考慮 |                                  URL                                  | 実装 |
|----|:----------:|:-------------:|:-----:|:------:|:---:|:---------------------------------------------------------------------:|:---:|
|[1] RFFL  | Similarity |       〇       |  不要   |   必要   | 〇 |           [here](https://arxiv.org/pdf/2011.10464v2.pdf)             | [here](./src/rffl/) | 
|[2] FOCUS | Similarity |       ×       |  必要   |   必要   | × | [here](https://link.springer.com/chapter/10.1007/978-3-030-63076-8_8) | [here](./src/focus) |
|[3] FedProf|Similarity|×|必要|必要| × | [here](https://arxiv.org/abs/2102.01733) | [here](./src/fedprof) |
|[4] FPPDL|Similarity|〇|不要|不要| × | [here](https://arxiv.org/pdf/1906.01167.pdf)|
|[5] QI|Marginal Loss|〇|不要|必要| 〇 | [here](https://arxiv.org/abs/2007.06236) | [here](./src/qualityinference) |
|[6] F-RICE|Marginal Loss|〇|不要|必要| × | [here](https://arxiv.org/abs/2102.13314)|
|[7] Simple Influence|Marginal Loss|〇|不要|必要| × |[here](https://ieeexplore.ieee.org/document/9006179)|
|[8] FAIR|Test /Self-Reported|〇|不要|必要| × | [here](https://ieeexplore.ieee.org/document/9488743)|
|[9] FairFed|Test /Self-Reported|〇|不要|不要| × | [here](https://ieeexplore.ieee.org/document/9425266) |
|[10] STD_DAGMM| ML |〇|不要|必要| 〇 | [here](https://arxiv.org/abs/1911.12560) | [here](./src/autoencoder) |
|[11] Spectral Anomaly Detection| ML |〇|不要|必要| × | [here](https://arxiv.org/abs/2002.00211) |

- [1] A Reputation Mechanism Is All You Need: Collaborative Fairness and Adversarial Robustness in Federated Learning, ICML 2021

個々のクライアントが送ってきたgradientと、集計したgradientのコサイン類似度を、クライアントの信頼度とす　　
<br/>

- [2] Chen Y., Yang X., Qin X., Yu H., Chan P., Shen Z. (2020) Dealing with Label Quality Disparity in Federated Learning. In: Yang Q., Fan L., Yu H. (eds) Federated Learning. Lecture Notes in Computer Science, vol 12500. Springer, Cham. https://doi.org/10.1007/978-3-030-63076-8_8

サーバー側も検証データを持っており、各クライアントが更新したモデルの、クライアントサイドのデータに対するスコ　アとサーバーサイドのデータに対するスコアの類似度を比較する。
<br/>

- [3] Wentai Wu, Ligang He, Weiwei Lin, Rui Mao. "FedProf: Optimizing Federated Learning with Dynamic Data Profiling." ICML2020, https://arxiv.org/abs/2102.01733, 2021.

サーバー側も検証データを持っており、検証データとクライアントが持っているデータに対する、モデルの中間層の出力　の分布の違いを計算する。
<br/>

- [5] J. Kang, Z. Xiong, D. Niyato, H. Yu, Y. -C. Liang and D. I. Kim, "Incentive Design for Efficient Federated Learning in Mobile Networks: A Contract Theory Approach," 2019 IEEE VTS Asia Pacific Wireless Communications Symposium (APWCS), 2019, pp. 1-5, doi: 10.1109/VTS-APWCS.2019.8851649.

現在のラウンドのスコアの上昇幅が前回よりも良かった場合、今回サンプリングしたクライアントの信頼度を+1、前回サ　ンプリングしたクライアントの信頼度を-1する
<br/>

- [6] Jie Zhao, Xinghua Zhu, Jianzong Wang, Jing Xiao. "Efficient Client Contribution Evaluation for Horizontal Federated Learning". https://arxiv.org/abs/2102.13314. 2021

強化学習を用いて、パフォーマンスが最も上がるようなクライアントの選び方を見つける  
<br/>

- [7] G. Wang, C. X. Dang and Z. Zhou, "Measure Contribution of Participants in Federated Learning," 2019 IEEE International Conference on Big Data (Big Data), 2019, pp. 2597-2604, doi: 10.1109/BigData47090.2019.9006179.

あるクライアントの信頼度を、そのクライアントが参加した場合のスコア - そのクライアントが参加しなかった場合のスコア　と定義する  
<br/>

- [8] Y. Deng et al., "FAIR: Quality-Aware Federated Learning with Precise User Incentive and Model Aggregation," IEEE INFOCOM 2021 - IEEE Conference on Computer Communications, 2021, pp. 1-10, doi: 10.1109/INFOCOM42981.2021.9488743

データの質を、前回集計したモデルの検証データに対するloss - クライアントサイドで訓練したモデルのクライアントサイドのデータに対するloss と定義する。この定義を用いつつ、インセンティブメカニズムを工夫して、悪意のあるクライアントが偽の報告をしても自身の効用を増やすことができないようにした　　　
<br/>

- [10] Lin, Jierui & Du, Min & Liu, Jian.. "Free-riders in Federated Learning: Attacks and Defenses. " arXiv preprint arXiv:1911.12560, 2019

クライアントが送ってきたパラメータをAutoEncoderにかけて、フリーライダーをあぶりだす。
<br/>



