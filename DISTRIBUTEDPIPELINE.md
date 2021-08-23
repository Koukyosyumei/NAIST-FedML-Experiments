# MPIをバックエンドにして、FedMLを動かす

## 実行例

### CIFAR10

    sh distributed_pipeline.sh "grid_short.q" 10 1 1 "./distributed_main.py" "gpu_mapping.yaml" "resnet56" "cifar10" "/work/hideaki-t/dev/FedML/data/cifar10" "hetero" 100 20 "adam" 64 0.001 0 1