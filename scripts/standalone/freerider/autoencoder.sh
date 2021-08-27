#!/bin/sh
#$ -S /bin/bash
#$ -q pascal_short.q

set -ex

# code checking
# pyflakes .

wandb_api_key=`cat ../../wandb_api_key.txt`
wandb login ${wandb_api_key} --relogin
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
max_gap=1

# 0. prepare data

cd ../../../src/data_generation

TEMP_FOLDER_NAME_1=`mktemp -d --tmpdir=../../data`
mkdir $TEMP_FOLDER_NAME_1/train
mkdir $TEMP_FOLDER_NAME_1/test

echo "grouping data"
python3 ./grouping.py \
--input_dir /work/hideaki-t/dev/FedML/data/MNIST \
--output_dir $TEMP_FOLDER_NAME_1 \
--client_num $client_num \
--max_gap $max_gap

# 1. MNIST standalone FedAvg
cd ../standalone

start_time=`date +%s`

python3 ./main.py \
--gpu 0 \
--dataset mnist \
--data_dir $TEMP_FOLDER_NAME_1 \
--model nn \
--partition_method hetero  \
--client_num_in_total $client_num \
--client_num_per_round $client_num \
--comm_round 200 \
--epochs 1 \
--batch_size 10 \
--client_optimizer sgd \
--method AE \
--freerider \
--free_rider_num 4 \
--max_gap $max_gap \
--lr 0.05 \
--ci 0

end_time=`date +%s`
run_time=$((end_time - start_time))
echo $run_time

rm -rf $TEMP_FOLDER_NAME_1