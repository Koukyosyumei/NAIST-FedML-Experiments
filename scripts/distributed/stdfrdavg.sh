#!/bin/sh
#$ -S /bin/bash
#$ -q pascal_short.q
#$ -pe mpi 144

module load compiler/gcc/7
module load mpi/openmpi/3.0.0

set -ex

# code checking
# pyflakes .

wandb login 02deeb10aa05ffa5e80eacf94128c7de1156d809 --relogin
wandb online

cd ../../src/distributed

hostname > mpi_host_file

mpirun -npernode 2 -np 12 python3 ./distributed_main.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_config6_4" \
  --model lr \
  --dataset mnist \
  --data_dir "/work/hideaki-t/dev/FedML/data/MNIST" \
  --partition_method hetero \
  --client_num_in_total 1000 \
  --client_num_per_round 11 \
  --comm_round 200 \
  --epochs 1 \
  --client_optimizer sgd \
  --batch_size 10 \
  --lr 0.03 \
  --ci 0