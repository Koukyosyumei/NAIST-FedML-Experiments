# NAIST-FedML-Experiments

This reporitory is my research work in Aug. 2021 - Sep. 2021 at NAIST under the supervision of Prof. Kohei Ichikawa and Prof. Keichi Takahashi. 

I made the following algorithms work in the HPC environment of NAIST with [FedML](https://github.com/FedML-AI/FedML) 

|name|code|reference|
|----|----|---------|
|FedAVG|[code](src/distributed/fedavg)|[paper](https://arxiv.org/pdf/1602.05629.pdf)|
|FedProf|[code](src/standalone/fedprof)|[paper](https://arxiv.org/abs/2102.01733)|
|FOCUS|[code](src/standalone/focus)|[paper](https://link.springer.com/chapter/10.1007/978-3-030-63076-8_8)|
|FoolsGold|[code](src/distributed/foolsgold)|[paper](https://arxiv.org/abs/1808.04866)|
|FreeRider|[code](src/distributed/freerider)|[paper](https://arxiv.org/abs/1911.12560)|
|STDDAGMM|[code](src/distributed/autoencoder)|[paper](https://arxiv.org/abs/1911.12560)|
|RFFL|[code](src/distributed/rffl)|[paper](https://arxiv.org/pdf/2011.10464v2.pdf)|
|Quality Inference|[code](src/qualityinference)|[paper](https://arxiv.org/abs/2007.06236)|

Sample: CIFAR-10
```
cd scripts/distributed
sh distributed_pipeline.sh config/cifar10.sh
```
