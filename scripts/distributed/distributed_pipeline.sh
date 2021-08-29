# parameters

method=$1
source ./config.sh

# check the arguments

echo "node_type=${node_type}"
echo "client_num=${client_num}"
echo "client_num_per_round=${client_num_per_round}"
echo "worker_num_pernode=${worker_num_pernode}"
echo "npernode=${npernode}"
echo "py_file=${py_file}"
echo "gpu_mapping_yaml=${gpu_mapping_yaml}"
echo "model=${model}"
echo "dataset=${dataset}"
echo "data_dir=${data_dir}"
echo "partition_method=${partition_method}"
echo "comm_round=${comm_round}"
echo "epochs=${epochs}"
echo "client_optimizer=${client_optimizer}"
echo "batch_size=${batch_size}"
echo "lr=${lr}"
echo "ci=${ci}"
echo "method=${method}"
echo "submit_script=${submit_script}"

# preparation

np=`expr ${client_num_per_round} + 1`

script_name="autogenerated_method_${method}_npn_${npernode}_np_${np}_m_${model}_ds_${dataset}_cn_${client_num}_cnp_${client_num_per_round}".sh
output_dir="output_autogenerated_method_${method}_npn_${npernode}_np_${np}_m_${model}_ds_${dataset}_cn_${client_num}_cnp_${client_num_per_round}"
wandb_api_key=`cat ../wandb_api_key.txt`

# auto-generate a script

echo -ne "#!/bin/sh
#$ -S /bin/bash
#$ -q ${node_type}
#$ -pe mpi $(($np*(24/$npernode)))

module load compiler/gcc/7
module load mpi/openmpi/3.0.0

# function cleanup_exit() {
#  hostname
#  pkill FedAvg
# }

# trap cleanup_exit SIGUSR2

set -ex

# code checking
# pyflakes .

wandb login ${wandb_api_key} --relogin
wandb online

cd ../../src/distributed

python3 gpu_mapping_yaml_generator.py --client_num_per_round $client_num_per_round --worker_num_pernode $worker_num_pernode --npernode $npernode

hostname > mpi_host_file

if [ ! -e ${output_dir} ]; then
  mkdir ${output_dir}
fi

mpirun -np ${np} -npernode ${npernode} python3 ${py_file} \\
  --gpu_mapping_file ${gpu_mapping_yaml} \\
  --gpu_mapping_key mapping_config_client_num_per_round_${client_num_per_round}_worker_num_pernode_${worker_num_pernode}_npernode_${npernode} \\
  --model ${model} \\
  --dataset ${dataset} \\
  --data_dir ${data_dir} \\
  --partition_method ${partition_method} \\
  --partition_alpha ${partition_alpha} \\
  --client_num_in_total ${client_num} \\
  --client_num_per_round ${client_num_per_round} \\
  --comm_round ${comm_round} \\
  --epochs ${epochs} \\
  --client_optimizer ${client_optimizer} \\
  --batch_size ${batch_size} \\
  --lr ${lr} \\
  --clip_grad ${clip_grad} \\
  --max_norm ${max_norm} \\
  --ci ${ci} \\
  --method ${method} \\
  --output_dir ${output_dir} \\
  --autoencoder_lr ${autoencoder_lr} \\
  --autoencoder_epochs ${autoencoder_epochs} \\
  --warm_up ${warm_up} \\
  --alpha ${alpha} \\
  --sparcity ${sparcity} \\
  --remove ${remove} \\
  --adversary_num ${adversary_num} \\
  --adversary_type ${adversary_type} \\
  --ignore_adversary ${ignore_adversary} \\
  --free_rider_strategy ${free_rider_strategy} \\
  --noise_amp ${noise_amp} \\
  --water_powered_magnification ${water_powered_magnification} \\
  --inflator_data_size ${inflator_data_size} \\
  --num_of_augmentation ${num_of_augmentation}
  
mpirun -np ${np} -npernode ${npernode} ps aux | grep FedAvg" > $script_name

# submit the auto-generated script
if [ $submit_script -eq 1 ] ; then
  qsub -notify $script_name
fi