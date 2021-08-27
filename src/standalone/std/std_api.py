import copy
import logging
import math
import os
import pickle
import random
import sys

import numpy as np
import torch
import wandb
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

from fedml_api.standalone.fedavg.client import Client
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from standalone.freerider.freerider_client import FreeRider_Client


class StdFedAvgAPI(FedAvgAPI):
    def __init__(
        self,
        dataset,
        device,
        args,
        model_trainer,
        true_credibility=None,
    ):
        super().__init__(dataset, device, args, model_trainer)
        self.true_credibility = true_credibility
        self.pred_credibility = np.array([0.0] * self.args.client_num_in_total)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        """
        self.validation_model = copy.deepcopy(self.model_trainer.model)
        self.validation_optimizer = torch.optim.SGD(
            self.validation_model.parameters(), lr=self.args.lr
        )
        """

        if self.args.overstate:
            self.y_adversary = np.array(true_credibility)
            self.y_adversary = np.where(self.y_adversary < 1, 0, 1)
            self.adversary_idx = np.where(np.array(true_credibility) < 1)[0]

        self.parameter_logs = []

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        model_trainer,
    ):
        logging.info("############setup_clients (START)#############")

        if self.args.freerider:
            self.adversary_idx = random.sample(
                list(range(self.args.client_num_in_total)), self.args.free_rider_num
            )
            self.y_adversary = np.array([0.0] * self.args.client_num_in_total)
            self.y_adversary[self.adversary_idx] = 1

            self.freerider = FreeRider_Client(
                0,
                train_data_local_dict[0],
                test_data_local_dict[0],
                train_data_local_num_dict[0],
                self.args,
                self.device,
                model_trainer,
                use_gradient=False,
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

                if self.args.freerider and client_idx in self.adversary_idx:
                    self.freerider.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )

                    # train on new dataset
                    w = self.freerider.train(copy.deepcopy(w_global))

                else:
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

                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))

            # update std-dagmm
            flattend_w_locals = torch.stack(
                [torch.cat([p.reshape(-1) for p in w[1].values()]) for w in w_locals]
            ).to("cpu")
            self.parameter_logs.append(flattend_w_locals)
            # self.pred_credibility[client_indexes] = cred.to("cpu").detach().numpy()

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

            # test result
            """
            if self.args.overstate or self.args.freerider:
                auc_crediblity = roc_auc_score(self.y_adversary, self.pred_credibility)
                wandb.log(
                    {"Credibility/FreeRider-AUC": auc_crediblity, "round": round_idx}
                )

            if self.true_credibility is not None:
                sim_credibility = spearmanr(
                    self.pred_credibility, self.true_credibility
                )[0]
                wandb.log(
                    {"Credibility/Spearmanr": sim_credibility, "round": round_idx}
                )
            """

        with open("parameters.pickle", mode="wb") as f:
            pickle.dump(self.parameter_logs, f)
        with open("samples_num.pickle", mode="wb") as f:
            pickle.dump([w[0] for w in w_locals], f)
