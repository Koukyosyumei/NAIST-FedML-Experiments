import copy
import logging
import math
import os
import sys

import numpy as np
import torch
import wandb
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI

from model import STD_DAGMM


class STDDAGMM_API(FedAvgAPI):
    def __init__(
        self,
        dataset,
        device,
        args,
        model_trainer,
        true_credibility,
    ):
        super().__init__(dataset, device, args, model_trainer)
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

        self.num_parameters = torch.cat(
            [p.reshape(-1) for p in self.model_trainer.model.parameters()]
        ).shape[-1]
        self.std_dagmm = STD_DAGMM(self.num_parameters, device)

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
                [torch.cat([p.reshape(-1) for p in w[1]]) for w in w_locals]
            )
            self.std_dagmm.fit(flattend_w_locals)
            energy = self.std_dagmm.predict(flattend_w_locals)
            self.pred_credibility[client_indexes] = energy.to("cpu").detach().numpy()
            sim_credibility = spearmanr(self.pred_credibility, self.true_credibility)[0]
            wandb.log({"Credibility/Spearmanr": sim_credibility, "round": round_idx})

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
