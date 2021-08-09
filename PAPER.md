# 関連論文

クライアントの質を推定する手法

## タイプ

### A: Test /Self-Reported Based Contribution Evaluation

各クライアントにデータ数やローカルでの性能を報告させたり、クライアントが送ってきたモデルをサーバー側の検証データで検証する。

例: J. Kang, Z. Xiong, D. Niyato, H. Yu, Y. -C. Liang and D. I. Kim, "Incentive Design for Efficient Federated Learning in Mobile Networks: A Contract Theory Approach," 2019 IEEE VTS Asia Pacific Wireless Communications Symposium (APWCS), 2019, pp. 1-5, doi: 10.1109/VTS-APWCS.2019.8851649.

### B: Marginal Loss Based Contribution Evaluation

あるクライアントの質を、そのクライアントがいる場合とそうでない場合のグローバルモデルの性能の差として定義する

例: G. Wang, C. X. Dang and Z. Zhou, "Measure Contribution of Participants in Federated Learning," 2019 IEEE International Conference on Big Data (Big Data), 2019, pp. 2597-2604, doi: 10.1109/BigData47090.2019.9006179.

### C: Similarity Based Contribution Evaluation

クライアント間の類似度や、サーバーが持つ検証データとクライアントが持つデータの類似度を測定し、外れ値を検出する

例: Wentai Wu, Ligang He, Weiwei Lin, Rui Mao. "FedProf: Optimizing Federated Learning with Dynamic Data Profiling." ICML2020, https://arxiv.org/abs/2102.01733, 2021.

### D: Parameter checking Based Contribution Evaluation

クライアントが送ってきたパラメータやgradientを、AutoEncoderなどの外れ値を検出するアルゴリズムにかけて、異常なクライアントを検出する

例: Lin, Jierui & Du, Min & Liu, Jian.. "Free-riders in Federated Learning: Attacks and Defenses. " arXiv preprint arXiv:1911.12560, 2019


## 比較

|手法|タイプ|悪意のあるクライアント|補助データ|中央サーバー|URL|
|:---:|:---:|:---:|:---:|:---:|:---:|
|RFFL|Similarity|〇|不要|必要|[here](https://arxiv.org/pdf/2011.10464v2.pdf)|
|FOCUS|Similarity|×|必要|必要|[here](https://link.springer.com/chapter/10.1007/978-3-030-63076-8_8)|
|QI|Marginal Loss|〇|不要|必要|
|FedProf|Similarity|×|必要|必要|
|FAIR|Test /Self-Reported||||
|STD_DAGMM|Parameter checking|〇|不要|必要|
|FPPDL|Similarity|||||
|Shapely Value|Marginal Loss||||
|F-RICE|Marginal Loss||||
|FairFed|Test /Self-Reported|〇|||
|Spectral Anomaly Detection|Parameter checking|〇|||