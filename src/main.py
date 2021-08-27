import argparse
import logging
import os
import pickle
import random
import sys

import numpy as np
import torch

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../FedML/")))
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_classification import (
    MyModelTrainer as MyModelTrainerCLS,
)
from fedml_api.standalone.fedavg.my_model_trainer_nwp import (
    MyModelTrainer as MyModelTrainerNWP,
)
from fedml_api.standalone.fedavg.my_model_trainer_tag_prediction import (
    MyModelTrainer as MyModelTrainerTAG,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./*/")))


from args import add_args
from core.gradient_trainer import GradientModelTrainerCLS
from dataloader import load_data
from model import create_model
from standalone.autoencoder.autoencoder_api import AutoEncoder_API
from standalone.fedavg.fedavg_api import FedAvgGradientAPI
from standalone.fedprof.fedprof_api import FedProfAPI
from standalone.focus.focus_api import FOCUSAPI
from standalone.qualityinference.qualityinference_api import QualityInferenceAPI
from standalone.rffl.rffl_api import RFFLAPI
from standalone.rffl.rffl_trainer import RFFL_ModelTrainer
from standalone.std.std_api import StdFedAvgAPI


def custom_model_trainer(args, model):
    if args.method == "FedAvgGrad":
        return GradientModelTrainerCLS(model)
    elif args.method == "RFFL":
        return RFFL_ModelTrainer(model)
    elif args.dataset == "stackoverflow_lr":
        return MyModelTrainerTAG(model)
    elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
        return MyModelTrainerNWP(model)
    else:  # default model trainer is for classification problem
        return MyModelTrainerCLS(model)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = add_args(argparse.ArgumentParser(description="FedAvg-standalone"))
    args = parser.parse_args()
    logger.info(args)
    device = torch.device(
        "cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu"
    )
    logger.info(device)

    wandb.init(
        project="fedml",
        name=f"{args.method}-r"
        + str(args.comm_round)
        + "-e"
        + str(args.epochs)
        + "-lr"
        + str(args.lr),
        config=args,
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    # load data
    dataset = load_data(args, args.dataset)

    # create model.
    # Note if the model is DNN (e.g., ResNet), the training will be very slow.
    # In this case, please use our FedML distributed version (./fedml_experiments/distributed_fedavg)
    model = create_model(args, model_name=args.model, output_dim=dataset[7])
    model_trainer = custom_model_trainer(args, model)
    logging.info(model)

    if args.overstate:
        with open(f"{args.data_dir}/credibility_train_label_fliped.pickle", "rb") as f:
            true_credibility = pickle.load(f)
        true_credibility = true_credibility[: args.client_num_in_total]
    else:
        true_credibility = None

    if args.method in ["QI", "FOCUS", "FedProf"]:
        with open(f"{args.data_dir}/X_server.pickle", "rb") as inf:
            X_server = torch.Tensor(pickle.load(inf))
        with open(f"{args.data_dir}/y_server.pickle", "rb") as inf:
            y_server = torch.Tensor(pickle.load(inf))

    if args.method == "FedAvg":
        fedavg_API = FedAvgAPI(dataset, device, args, model_trainer)
        fedavg_API.train()

    if args.method == "FedAvgGrad":
        fedavg_API = FedAvgGradientAPI(
            dataset, device, args, model_trainer, true_credibility
        )
        fedavg_API.train()

    elif args.method == "QI":
        qualityinferenceAPI = QualityInferenceAPI(
            dataset, device, args, model_trainer, true_credibility, X_server, y_server
        )
        qualityinferenceAPI.train()

    elif args.method == "RFFL":
        rffl_API = RFFLAPI(dataset, device, args, model_trainer, true_credibility)
        rffl_API.train()

    elif args.method == "AE":
        autoencoderAPI = AutoEncoder_API(
            dataset, device, args, model_trainer, true_credibility=true_credibility
        )
        autoencoderAPI.train()

    elif args.method == "FOCUS":
        fedprofAPI = FOCUSAPI(
            dataset, device, args, model_trainer, true_credibility, X_server, y_server
        )
        fedprofAPI.train()

    elif args.method == "FedProf":
        fedprofAPI = FedProfAPI(
            dataset, device, args, model_trainer, true_credibility, X_server, y_server
        )
        fedprofAPI.train()

    elif args.method == "STD":
        stdfedavgAPI = StdFedAvgAPI(
            dataset, device, args, model_trainer, true_credibility=true_credibility
        )
        stdfedavgAPI.train()
