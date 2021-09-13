import argparse
import copy
import logging
import os
import random
import socket
import sys
import traceback

import mpi4py
import numpy as np
import psutil
import setproctitle
import torch
import torch.utils.data as data
import torchvision

mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

import wandb

# add the FedML root directory to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))
from fedml_api.distributed.fedavg.FedAVGAggregator import FedAVGAggregator
from fedml_api.distributed.fedavg.FedAvgAPI import FedML_init
from fedml_api.distributed.fedavg.FedAvgServerManager import FedAVGServerManager
from fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer
from fedml_api.distributed.utils.gpu_mapping import (
    mapping_processes_to_gpu_device_from_yaml_file,
)
from fedml_api.standalone.fedavg.my_model_trainer_classification import (
    MyModelTrainer as MyModelTrainerCLS,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from core.distributed_api import (
    Client_Initializer,
    FedML_Distributed_Custom_API,
    Server_Initializer,
)
from core.distributed_secure_server_manager import SecureFedAVGServerManager
from core.gradient_trainer import GradientModelTrainerCLS, GradientModelTrainerNWP

from autoencoder.autoencoder_aggregator import FedAVGAutoEncoderAggregator
from distributed_args import add_args
from distributed_dataloader import load_data
from distributed_model import create_model
from fedavg.fedavg_gradient_aggregator import FedAVGGradientAggregator
from fedavg.fedavg_gradient_trainer import FedAVGGradTrainer
from foolsgold.foolsgold_api import FoolsGoldAggregator
from freerider.freerider_modeltrainer import FreeriderModelTrainer
from inflator.inflator_client_manager import FedAVGInflatorClientManager
from qualityinference.qualityinference_aggregator import (
    FedAVGQualityInferenceAggregator,
)
from rffl.rffl_aggregator import RFFLAggregator
from rffl.rffl_clientmanager import RFFLClientManager
from rffl.rffl_trainer import RFFLTrainer
from similarity.similarity_aggregator import FedAVGSimilarityAggregator
from stdmonitor.std_aggregator import STDFedAVGAggregator

SEED = 42


def set_seed(seed=0):
    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(seed=0)

    # quick fix for issue in MacOS environment: https://github.com/openai/spinningup/issues/16
    if sys.platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # initialize distributed computing (MPI)
    comm, process_id, worker_number = FedML_init()

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    logging.info(args)

    # customize the process name
    str_process_name = "FedAvg (distributed):" + str(process_id)
    setproctitle.setproctitle(str_process_name)

    # customize the log format
    # logging.basicConfig(level=logging.INFO,
    logging.basicConfig(
        level=logging.DEBUG,
        format=str(process_id)
        + " - %(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
    )
    hostname = socket.gethostname()
    logging.info(
        "#############process ID = "
        + str(process_id)
        + ", host name = "
        + hostname
        + "########"
        + ", process ID = "
        + str(os.getpid())
        + ", process Name = "
        + str(psutil.Process(os.getpid()))
    )

    # initialize the wandb machine learning experimental tracking platform (https://www.wandb.com/).
    if process_id == 0:
        wandb_object = wandb.init(
            # project="federated_nas",
            project="fedml",
            name=args.method
            + "-"
            + str(args.partition_method)
            + "-r"
            + str(args.comm_round)
            + "-e"
            + str(args.epochs)
            + "-lr"
            + str(args.lr),
            config=args,
        )

    # Please check "GPU_MAPPING.md" to see how to define the topology
    logging.info("process_id = %d, size = %d" % (process_id, worker_number))
    device = mapping_processes_to_gpu_device_from_yaml_file(
        process_id, worker_number, args.gpu_mapping_file, args.gpu_mapping_key
    )

    if args.dataset == "fed_shakespeare":
        model_trainer_class = GradientModelTrainerNWP
    else:
        model_trainer_class = GradientModelTrainerCLS

    if args.method == "FedAvg":
        trainer_class = FedAVGGradTrainer
        aggregator_class = FedAVGGradientAggregator
        server_manager_class = SecureFedAVGServerManager
        client_manager_class = FedAVGInflatorClientManager
    elif args.method == "QI":
        trainer_class = FedAVGGradTrainer
        aggregator_class = FedAVGQualityInferenceAggregator
        server_manager_class = SecureFedAVGServerManager
        client_manager_class = FedAVGInflatorClientManager
    elif args.method == "RFFL":
        trainer_class = RFFLTrainer
        aggregator_class = RFFLAggregator
        server_manager_class = SecureFedAVGServerManager
        client_manager_class = RFFLClientManager
    elif args.method == "AE":
        trainer_class = FedAVGGradTrainer
        aggregator_class = FedAVGAutoEncoderAggregator
        server_manager_class = SecureFedAVGServerManager
        client_manager_class = FedAVGInflatorClientManager
    elif args.method == "FoolsGold":
        trainer_class = FedAVGGradTrainer
        aggregator_class = FoolsGoldAggregator
        server_manager_class = SecureFedAVGServerManager
        client_manager_class = FedAVGInflatorClientManager
    elif args.method == "SIM":
        trainer_class = FedAVGGradTrainer
        aggregator_class = FedAVGSimilarityAggregator
        server_manager_class = SecureFedAVGServerManager
        client_manager_class = FedAVGInflatorClientManager

    # choose adversaries
    adversary_idx = random.sample(
        list(range(args.client_num_in_total)), args.adversary_num
    )
    adversary_flag = np.zeros(args.client_num_in_total).astype(int)
    adversary_flag[adversary_idx] += 1

    # load data
    dataset = load_data(args, args.dataset, adversary_idx=adversary_idx)
    [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ] = dataset

    # re-choose adversaries based on the local dataset size
    if args.poor_adversary == 1:
        # the clients with the least amount of data are the attacker
        logging.info(
            f"####### re-assigin adversaries: poor_adversary={args.poor_adversary} #######"
        )
        adversary_idx = np.argsort(
            [train_data_local_num_dict[i] for i in range(args.client_num_in_total)]
        )[: args.adversary_num].tolist()
        adversary_flag = np.zeros(args.client_num_in_total).astype(int)
        adversary_flag[adversary_idx] += 1

        if args.partition_method == "pow":
            for aid in adversary_idx:

                idx = random.sample(
                    range(train_data_local_num_dict[aid]),
                    int(
                        train_data_local_num_dict[aid]
                        / args.water_powered_magnification
                    ),
                )
                train_data_local_num_dict[aid] = len(idx)

                if args.dataset == "cifar10":
                    train_data_local_dict[aid].dataset.data = copy.deepcopy(
                        train_data_local_dict[aid].dataset.data[idx]
                    )
                    train_data_local_dict[aid].dataset.target = copy.deepcopy(
                        train_data_local_dict[aid].dataset.target[idx]
                    )
                elif args.dataset == "fed_shakespeare":
                    train_data_local_dict[aid].dataset.tensors = (
                        copy.deepcopy(
                            train_data_local_dict[aid].dataset.tensors[0][idx]
                        ),
                        copy.deepcopy(
                            train_data_local_dict[aid].dataset.tensors[1][idx]
                        ),
                    )

    elif args.poor_adversary == -1:
        # clients with the median number of data are the attacker
        logging.info(
            f"####### re-assigin adversaries: poor_adversary={args.poor_adversary} #######"
        )
        sorted_idx = np.argsort(
            [train_data_local_num_dict[i] for i in range(args.client_num_in_total)]
        )

        adversary_mid_idx = (
            int(args.client_num_in_total / 2)
            if args.client_num_in_total % 2 == 0
            else int((args.client_num_in_total + 1) / 2)
        )
        adversary_left = (
            int(args.adversary_num / 2)
            if args.adversary_num % 2 == 0
            else int((args.adversary_num + 1) / 2)
        )
        adversary_right = args.adversary_num - adversary_left
        adversary_idx = (
            sorted_idx[adversary_mid_idx - adversary_left : adversary_mid_idx].tolist()
            + sorted_idx[
                adversary_mid_idx : adversary_mid_idx + adversary_right
            ].tolist()
        )
        adversary_flag = np.zeros(args.client_num_in_total).astype(int)
        adversary_flag[adversary_idx] += 1

    # setting for adversaries
    water_powered_magnification = 1.0
    if process_id - 1 in adversary_idx:
        if args.adversary_type == "freerider":
            logging.info(f"####### process_id = {process_id} is a freerider #######")
            model_trainer_class = FreeriderModelTrainer
        elif args.adversary_type == "inflator":
            logging.info(f"####### process_id = {process_id} is an inflator #######")
            # change the batch_size
            train_data_local_dict[process_id - 1] = data.DataLoader(
                dataset=train_data_local_dict[process_id - 1].dataset,
                batch_size=args.inflator_batch_size,
                shuffle=True,
                drop_last=train_data_local_dict[process_id - 1].drop_last,
            )

    if args.inflator_strategy == "small_lr" and process_id - 1 in adversary_idx:
        pass

    elif args.inflator_strategy == "data_augmentation":
        if args.dataset == "cifar10":
            adversary_transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.RandomCrop(32, padding=4),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    # transforms.ToPILImage(),
                    # transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
                ]
            )
            for idx in adversary_idx:
                transformed_data = []
                transformed_target = []
                temp_dataset = copy.deepcopy(train_data_local_dict[idx].dataset)
                temp_dataset.transform = adversary_transform
                temp_dataloader = data.DataLoader(
                    dataset=temp_dataset,
                    batch_size=1,
                    shuffle=True,
                    drop_last=train_data_local_dict[idx].drop_last,
                )

                for i in range(args.water_powered_magnification - 1):
                    set_seed(seed=SEED + i)
                    for xs, ys in temp_dataloader:
                        transformed_data.append(
                            np.array(
                                torchvision.transforms.functional.to_pil_image(xs[0])
                            )
                        )
                        transformed_target.append(ys.numpy())

                transformed_data = np.stack(transformed_data)
                transformed_target = np.concatenate(transformed_target)

                train_data_local_dict[idx].dataset.data = np.concatenate(
                    [train_data_local_dict[idx].dataset.data, transformed_data]
                )
                train_data_local_dict[idx].dataset.target = np.concatenate(
                    [
                        train_data_local_dict[idx].dataset.target,
                        transformed_target,
                    ]
                )

    elif args.inflator_strategy == "multiple_accounts":

        logging.info("######## create fake accounts ########")
        mid_idx = (
            int(len(adversary_idx) / 2)
            if len(adversary_idx) % 2 == 0
            else int((len(adversary_idx) + 1) / 2)
        )
        base_local_num = train_data_local_num_dict[adversary_idx[mid_idx]]
        base_dataloader = copy.deepcopy(train_data_local_dict[adversary_idx[mid_idx]])

        for i in range(len(adversary_idx)):
            random.seed(adversary_idx[i])
            idx = random.sample(
                range(base_local_num),
                int(base_local_num * args.multiple_accounts_split),
            )

            if args.dataset == "cifar10":
                train_data_local_dict[adversary_idx[i]].dataset.data = copy.deepcopy(
                    base_dataloader.dataset.data[idx]
                )
                train_data_local_dict[adversary_idx[i]].dataset.target = copy.deepcopy(
                    base_dataloader.dataset.target[idx]
                )
            elif args.dataset == "fed_shakespeare":
                train_data_local_dict[adversary_idx[i]].dataset.tensors = (
                    copy.deepcopy(base_dataloader.dataset.tensors[0][idx]),
                    copy.deepcopy(base_dataloader.dataset.tensors[1][idx]),
                )

            train_data_local_num_dict[adversary_idx[i]] = len(idx)

    # reset the seed
    set_seed(seed=0)
    assert np.sum(adversary_flag) == args.adversary_num

    # logging the distribution
    if process_id == 0:
        logging.info(f"######## adversary_idx = {adversary_idx}   ########")
        logging.info(f"######## adversary_flag = {adversary_flag} ########")
        logging.info(
            f"######## sum of train_data_local_num_dict = {sum(list(train_data_local_num_dict.values()))} ########"
        )

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model_trainer = model_trainer_class(model)

    # initializer
    server_initializer = Server_Initializer(
        aggregator_class=aggregator_class,
        server_manager_class=server_manager_class,
    )
    client_initializer = Client_Initializer(
        trainer_class=trainer_class, client_manager_class=client_manager_class
    )

    try:
        # start "federated averaging (FedAvg)"
        FedML_Distributed_Custom_API(
            process_id,
            worker_number,
            device,
            comm,
            model,
            train_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            args,
            server_initializer,
            client_initializer,
            model_trainer,
            adversary_idx=adversary_idx,
            adversary_flag=adversary_flag,
            water_powered_magnification=args.water_powered_magnification,
        )
    except Exception as e:
        logging.info(e)
        logging.info("traceback.format_exc():\n%s" % traceback.format_exc())
        MPI.COMM_WORLD.Abort()
