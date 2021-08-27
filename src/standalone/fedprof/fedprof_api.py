import copy
import glob
import logging
import math
import os
import random
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from standalone.fedprof.footprinter import FootPrinter

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

from fedml_api.standalone.fedavg.client import Client
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI


class FedProfAPI(FedAvgAPI):
    def __init__(
        self,
        dataset,
        device,
        args,
        model_trainer,
        true_credibility,
        X_server,
        y_server,
    ):
        super().__init__(dataset, device, args, model_trainer)
        self.X_server = X_server
        self.y_server = y_server
        self.true_credibility = true_credibility
        self.pred_credibility = np.array([0.0] * self.args.client_num_in_total)
        self.criterion = nn.CrossEntropyLoss()
        self.alpha = args.alpha
        self.device = device
        """
        self.validation_model = copy.deepcopy(self.model_trainer.model)
        self.validation_optimizer = torch.optim.SGD(
            self.validation_model.parameters(), lr=self.args.lr
        )
        """

    def train(self):
        w_global = self.model_trainer.get_model_params()
        client_indexes = []

        footprinter = FootPrinter(device=self.device)

        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

            # calculate credbility of each client
            footprinter.update_encoder(self.model_trainer.model.fc1)
            server_footprint = footprinter.culc_footprint(
                self.X_server, dataloader=False
            )
            for idx in range(self.args.client_num_in_total):
                client_footprint = footprinter.culc_footprint(
                    self.train_data_local_dict[idx]
                )

                self.pred_credibility[idx] = math.e ** (
                    -self.alpha
                    * footprinter.kldiv_between_server_and_client(
                        server_footprint, client_footprint
                    )
                )
                """
                self.pred_credibility[
                    idx
                ] -= footprinter.kldiv_between_server_and_client(
                    server_footprint, client_footprint
                )
                """

            sim_footprint = spearmanr(self.pred_credibility, self.true_credibility)[0]
            wandb.log({"Credibility/Spearmanr": sim_footprint, "round": round_idx})

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(
                    client_idx,
                    self.train_data_local_dict[client_idx],
                    self.test_data_local_dict[client_idx],
                    self.train_data_local_num_dict[client_idx],
                )

                # train on new dataset
                w = client.train(copy.deepcopy(w_global))
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update global weights
            w_global = self._aggregate(w_locals)
            self.model_trainer.set_model_params(w_global)

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                range(client_num_in_total),
                num_clients,
                replace=False,
                p=self.pred_credibility / np.sum(self.pred_credibility),
            )
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes
