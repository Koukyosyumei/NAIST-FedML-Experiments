## Lin, Jierui & Du, Min & Liu, Jian.. "Free-riders in Federated Learning: Attacks and Defenses. " arXiv preprint arXiv:1911.12560, 2019

## 実験設定

### DataSet

    1. MNIST

### Data Splits

    1. 各クライアントが持つデータの分布は同様 (ＭNISTの場合だと、各クライアントは10種すべてのラベルを持つ)
    2. 各クライアントが持つデータの分布は同様でない (各クライアントは最大2種のラベルしか持たない)
    3. クライアント総数が100で、1人だけフリーライダーがいる
    4. クライアント総数が100で、20人だけフリーライダーがいる

### フリーライダーの戦略

    1. Random Weight
    2. Delta Attack (今回サーバーからダウンロードしたモデルと、前回ダウンロードしたモデルの差分を用いる)
    3. Advanced Delta Attack (Delta Attack にN(0, 10^-3)のノイズを載せる)

### 指標

    1. AUC



