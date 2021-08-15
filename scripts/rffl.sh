#!/bin/sh
#$ -S /bin/bash
#$ -q pascal_short.q

set -ex

# code checking
# pyflakes .

wandb login 02deeb10aa05ffa5e80eacf94128c7de1156d809 --relogin
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

# 0. prepare data

cd ../data-generation

TEMP_FOLDER_NAME_1=`mktemp -d --tmpdir=../data`
mkdir $TEMP_FOLDER_NAME_1/train
mkdir $TEMP_FOLDER_NAME_1/test

echo "grouping data"
python3 ./grouping.py \
--input_dir /work/hideaki-t/dev/FedML/data/MNIST \
--output_dir $TEMP_FOLDER_NAME_1 \
--group_size 49

# 1. MNIST standalone FedAvg
cd ../src

start_time=`date +%s`

python3 ./main.py \
--gpu 0 \
--dataset mnist \
--data_dir $TEMP_FOLDER_NAME_1 \
--model nn \
--partition_method hetero  \
--client_num_in_total 20 \
--client_num_per_round 20 \
--comm_round 200 \
--epochs 1 \
--batch_size 10 \
--client_optimizer adam \
--lr 0.00003 \
--ci 0 \
--method RFFL \
--freerider \
--free_rider_num 4 \
--agg_weight 1 \
--gamma 0.5 \
--use_reputation \
--use_sparsify \
--remove

end_time=`date +%s`
run_time=$((end_time - start_time))
echo $run_time

rm -rf $TEMP_FOLDER_NAME_1