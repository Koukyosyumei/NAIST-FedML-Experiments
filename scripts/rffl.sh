#!/bin/sh
#$ -S /bin/bash
# #$ -q pascal_short.q

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

if [ -e "/work/hideaki-t/dev/NAIST-Experiments/data/grouped/*/*.json" ]; then
  rm /work/hideaki-t/dev/NAIST-Experiments/data/grouped/train/*.json
  rm /work/hideaki-t/dev/NAIST-Experiments/data/grouped/test/*.json
fi

echo "grouping data"
python3 ./grouping.py \
--input_dir /work/hideaki-t/dev/FedML/data/MNIST \
--output_dir /work/hideaki-t/dev/NAIST-Experiments/data/grouped \
--group_size 20

echo "flip label"
python3 ./label-flip.py \
--input_dir /work/hideaki-t/dev/NAIST-Experiments/data/grouped \
--output_dir /work/hideaki-t/dev/NAIST-Experiments/data/label_flip \
--flip_ratio 0.3

# 1. MNIST standalone FedAvg
cd ../src/rffl

start_time=`date +%s`

python3 ./main.py \
--gpu 0 \
--dataset mnist \
--data_dir ../../data/label_flip \
--model lr \
--partition_method hetero  \
--client_num_in_total 49 \
--client_num_per_round 49 \
--comm_round 200 \
--epochs 1 \
--batch_size 10 \
--client_optimizer adam \
--lr 0.0001 \
--ci 0 \
--agg_weight 1 \
--gamma 0.5

end_time=`date +%s`
run_time=$((end_time - start_time))
echo $run_time