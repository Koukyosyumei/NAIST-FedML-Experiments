#!/bin/sh
#$ -S /bin/bash
#$ -q pascal_short.q

set -ex

# code checking
# pyflakes .

wandb_api_key=`cat wandb_api_key.txt`
wandb login $wandb_api_key --relogin
wandb online

assert_eq() {
  local expected="$1"
  local actual="$2"
  local msg

  if [ "$expected" == "$actual" ]; then
    return 0
  else
    echo "$expected != $actual"
    return 1
  fi
}

round() {
  printf "%.${2}f" "${1}"
}

client_num=20
max_gap=50

# 1. CIFAR10 standalone FedAvg
cd ../standalone

start_time=`date +%s`

python3 ./main.py \
--gpu 0 \
--dataset cifar10 \
--data_dir /work/hideaki-t/dev/FedML/data/cifarR10 \
--model resnet56 \
--partition_method hetero  \
--client_num_in_total $client_num \
--client_num_per_round $client_num \
--comm_round 15 \
--epochs 1 \
--batch_size 10 \
--client_optimizer sgd \
--method STD \
--max_gap $max_gap \
--lr 0.03 \
--ci 0

end_time=`date +%s`
run_time=$((end_time - start_time))
echo $run_time