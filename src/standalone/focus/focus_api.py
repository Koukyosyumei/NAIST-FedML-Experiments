import copy
import logging
import math
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from torch import nn

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI


class FOCUSAPI(FedAvgAPI):
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

        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []

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

            lskt = self._local_test_on_server_data(
                w_locals, self.X_server, self.y_server
            )

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
                    train_metrics = self._local_test_on_all_clients(round_idx)

            llkt = np.array(train_metrics["losses"]) / np.array(
                train_metrics["num_samples"]
            )

            ekt = lskt + llkt
            self.pred_credibility[client_indexes] = 1 - (
                math.e ** (self.alpha * ekt)
            ) / np.sum(math.e ** (self.alpha * ekt))

            sim_credibility = spearmanr(self.pred_credibility, self.true_credibility)[0]
            wandb.log({"Credibility/Spearmanr": sim_credibility, "round": round_idx})

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(
                0,
                self.train_data_local_dict[client_idx],
                self.test_data_local_dict[client_idx],
                self.train_data_local_num_dict[client_idx],
            )
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(
                copy.deepcopy(train_local_metrics["test_total"])
            )
            train_metrics["num_correct"].append(
                copy.deepcopy(train_local_metrics["test_correct"])
            )
            train_metrics["losses"].append(
                copy.deepcopy(train_local_metrics["test_loss"])
            )

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(
                copy.deepcopy(test_local_metrics["test_total"])
            )
            test_metrics["num_correct"].append(
                copy.deepcopy(test_local_metrics["test_correct"])
            )
            test_metrics["losses"].append(
                copy.deepcopy(test_local_metrics["test_loss"])
            )

            """
            Note: CI environment is CPU-based computing.
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(
            train_metrics["num_samples"]
        )
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        return train_metrics

    def _local_test_on_server_data(self, w_locals, X_val, y_val, func=accuracy_score):
        ls = []
        for w in w_locals:
            self.model_trainer.set_model_params(w[1])

            with torch.no_grad():
                self.model_trainer.model.to(self.device)
                y_pred = self.model_trainer.model(X_val.to(self.device))
                # entropy = ((-math.e ** y_pred) * y_pred).sum(axis=1).mean()
                _, predicted = torch.max(y_pred, 1)
                loss = self.criterion(y_pred, y_val.to(self.device).to(int))

            ls.append(loss.item())

        return np.array(ls)
