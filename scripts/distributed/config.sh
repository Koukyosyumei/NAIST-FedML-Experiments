######## General settings ########
# cluster setting
node_type="grid_short.q"
npernode=1
gpu_mapping_yaml="gpu_mapping.yaml"

# clients setting
client_num=20
client_num_per_round=10
worker_num_pernode=1

# py file to be executed
py_file="./distributed_main.py"

# model and optimier setting
model="resnet56"
comm_round=20
epochs=5
client_optimizer="adam"
lr=0.001
clip_grad=1
max_norm=5.0

# dataset setting
dataset="cifar10"
data_dir="/work/hideaki-t/dev/FedML/data/cifar10"
partition_method="homo"
partition_alpha=0.3
batch_size=20

# other settings
ci=0
submit_script=1

######## Method settings ########

## AutoEncoder (AE) settings
autoencoder_lr=0.01
autoencoder_epochs=5

## RFFL settings
warm_up=5
alpha=0.95
sparcity=1
remove=1

## Quality Inferece (QI) settings

######## Adversary settings ########
adversary_num=4
adversary_type="freerider"
ignore_adversary=0

## Free-Rider settings
free_rider_strategy="advanced-delta"
noise_amp=0.001

## Inflator settings
water_powered_magnification=1.0
inflator_data_size=30
num_of_augmentation=0