import copy
import logging
import os
import random
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))

from fedml_api.standalone.fedavg.client import Client
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from freerider.freerider_client import FreeRider_Client


class QualityInferenceAPI(FedAvgAPI):
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
        self.device = device

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        self.freeriders_idx = random.sample(
            list(range(self.args.client_num_in_total)), self.args.free_rider_num
        )
        self.y_freerider = np.array([0.0] * self.args.client_num_in_total)
        self.y_freerider[self.freeriders_idx] = 1

        self.freerider = FreeRider_Client(
            0,
            train_data_local_dict[0],
            test_data_local_dict[0],
            train_data_local_num_dict[0],
            self.args,
            self.device,
            model_trainer,
        )
        for client_idx in range(self.args.client_num_per_round):
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        client_indexes = []
        prev_client_indexes = []

        score_improve_curr = 0
        score_improve_prev = 0
        score_curr = 0
        score_prev = 0

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

                if client_idx not in self.freeriders_idx:
                    # normal client
                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )

                    # train on new dataset
                    w = client.train(copy.deepcopy(w_global))
                    # self.logger.info("local weights = " + str(w))
                else:
                    self.freerider.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )

                    # train on new dataset
                    w = self.freerider.train(copy.deepcopy(w_global))

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

            # server side test
            score_prev = score_curr
            # entropy_prev = entropy_curr

            score_curr, loss_curr = self._server_test(self.X_server, self.y_server)

            score_improve_prev = score_improve_curr
            score_improve_curr = score_curr - score_prev if round_idx > 0 else 0
            if round_idx > 0 and score_improve_curr > score_improve_prev:
                self.pred_credibility[client_indexes] += 1
                self.pred_credibility[prev_client_indexes] -= 1
            if score_improve_curr < 0:
                self.pred_credibility[client_indexes] -= 1

            logging.info("pred_credibility")
            logging.info(self.pred_credibility)

            if self.args.freerider:
                auc_crediblity = roc_auc_score(self.y_freerider, self.pred_credibility)
                wandb.log(
                    {"Credibility/FreeRider-AUC": auc_crediblity, "round": round_idx}
                )
            else:
                if round_idx > 0:
                    sim_vanila = spearmanr(
                        self.pred_credibility, self.true_credibility
                    )[0]
                else:
                    sim_vanila = 0
                wandb.log({"Credibility/Spearmanr": sim_vanila, "round": round_idx})

            prev_client_indexes = copy.deepcopy(client_indexes)

    def _server_test(self, X_val, y_val, func=accuracy_score):
        with torch.no_grad():
            self.model_trainer.model.to(self.device)
            y_pred = self.model_trainer.model(X_val.to(self.device))
            # entropy = ((-math.e ** y_pred) * y_pred).sum(axis=1).mean()
            _, predicted = torch.max(y_pred, 1)
            score = accuracy_score(
                predicted.to("cpu").detach().numpy(), y_val.to("cpu").detach().numpy()
            )
            loss = self.criterion(y_pred, y_val.to(self.device).to(int))
        return score.item(), loss.item()
