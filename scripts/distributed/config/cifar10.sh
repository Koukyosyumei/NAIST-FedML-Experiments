######## General settings ########
# cluster setting
node_type="pascal_short.q"
gpupernode=2
gpu_mapping_yaml="gpu_mapping.yaml"

# clients setting
client_num=10
client_num_per_round=10
worker_num_pergpu=1

# method
method="FoolsGold"

# py file to be executed
py_file="./distributed_main.py"

# model and optimier setting
model="resnet56"
comm_round=55
epochs=5
client_optimizer="adam"
lr=0.001
clip_grad=0
max_norm=1

# dataset setting
dataset="cifar10"
data_dir="/work/hideaki-t/dev/FedML/data/cifar10"
partition_method="homo"
partition_alpha=0.1
batch_size=20

# other settings
frequency_of_the_test=5
ci=0
submit_script=1

######## Method settings ########

## AutoEncoder (AE) settings
autoencoder_lr=0.01
autoencoder_epochs=5
autoencoder_type="STD-NUM-DAGMM"

## RFFL settings
warm_up=5
alpha=0.95
gamma=0.5
sparcity=1
remove=1

## FoolsGold settings
k=0.02
inv=0
indicative_features="all"

## Quality Inferece (QI) settings

######## Adversary settings ########
adversary_num=2
adversary_type="inflator"
inflator_strategy="multiple_accounts"
multiple_accounts_split=1.0
ignore_adversary=0
poor_adversary=0

## Free-Rider settings
free_rider_strategy="advanced-delta"
noise_amp=0.001

## Inflator settings
water_powered_magnification=1
inflator_data_size=5000
inflator_batch_size=20
num_of_augmentation=0