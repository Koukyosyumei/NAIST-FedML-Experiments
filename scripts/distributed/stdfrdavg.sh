#!/bin/sh
#$ -S /bin/bash
#$ -q grid_short.q
#$ -pe mpi 504

module load compiler/gcc/7
module load mpi/openmpi/3.0.0

set -ex

# code checking
# pyflakes .

wandb login 02deeb10aa05ffa5e80eacf94128c7de1156d809 --relogin
wandb online

client_num=20
max_gap=10
min_mag=3.0
max_mag=5.0
inflated_client_num=0
inflator="rich"

cd ../../data-generation

TEMP_FOLDER_NAME_1=`mktemp -d --tmpdir=../data`
mkdir $TEMP_FOLDER_NAME_1/train
mkdir $TEMP_FOLDER_NAME_1/test

TEMP_FOLDER_NAME_2=`mktemp -d --tmpdir=../data`
mkdir $TEMP_FOLDER_NAME_2/train
mkdir $TEMP_FOLDER_NAME_2/test

echo "grouping data"
python3 ./grouping.py \
--input_dir /work/hideaki-t/dev/FedML/data/MNIST \
--output_dir $TEMP_FOLDER_NAME_1 \
--client_num $client_num \
--max_gap $max_gap

echo "inflating data"
python3 ./overstate.py \
--input_dir $TEMP_FOLDER_NAME_1 \
--output_dir $TEMP_FOLDER_NAME_2 \
--inflated_client_num 0 \
--min_mag 1 \
--max_mag 1 \
--inflator "random"

cd ../src/distributed

hostname > mpi_host_file

mpirun -npernode 1 -np 21 python3 ./distributed_main.py \
  --gpu_mapping_file "gpu_mapping.yaml" \
  --gpu_mapping_key "mapping_config_grid_20" \
  --model lr \
  --dataset mnist \
  --data_dir ../$TEMP_FOLDER_NAME_2 \
  --partition_method hetero \
  --client_num_in_total 20 \
  --client_num_per_round 20 \
  --comm_round 10 \
  --epochs 1 \
  --client_optimizer sgd \
  --batch_size 10 \
  --lr 0.03 \
  --ci 0