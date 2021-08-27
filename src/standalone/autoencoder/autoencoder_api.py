import copy
import logging
import math
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
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

from fedml_api.standalone.fedavg.client import Client
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from standalone.autoencoder.detector import STD_DAGMM
from standalone.fedavg.fedavg_api import FedAvgGradientAPI
from standalone.freerider.freerider_client import FreeRider_Client


class AutoEncoder_API(FedAvgGradientAPI):
    def __init__(
        self,
        dataset,
        device,
        args,
        model_trainer,
        true_credibility=None,
    ):
        super().__init__(dataset, device, args, model_trainer, true_credibility)
        self.pred_credibility = np.array([0.0] * self.args.client_num_in_total)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device
        self.num_parameters = torch.cat(
            [p.reshape(-1) for p in self.model_trainer.model.parameters()]
        ).shape[-1]
        self.autoencoder = STD_DAGMM(self.num_parameters, device)

    def train(self):
        aggregated_gradient = None
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            gradient_locals = {}

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                client_idx = client_indexes[idx]
                if self.args.freerider and client_idx in self.adversary_idx:
                    self.freerider.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )

                    if round_idx > 0:
                        # train on new dataset
                        grad = self.freerider.train(aggregated_gradient)
                    else:
                        grad = self.freerider.train(None)

                else:
                    client = self.client_list[client_idx]
                    client.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )
                    # 学習
                    grad = client.train(copy.deepcopy(w_global))
                    # 更新したパラメータを保存

                # upload
                gradient_locals[client_idx] = grad

            # STD-DAGMM
            flattend_gradient_locals = torch.stack(
                [
                    torch.cat([g.to(self.device).reshape(-1) for g in gl[1][1]])
                    for gl in gradient_locals.items()
                ]
            )
            logging.info(
                f"the shape of flattend_w_locals is {flattend_gradient_locals.shape}"
            )

            temp_client_idx = [g[0] for g in gradient_locals.items()]
            self.autoencoder.fit(flattend_gradient_locals)
            cred = self.autoencoder.predict(flattend_gradient_locals)
            self.pred_credibility[temp_client_idx] = cred.to("cpu").detach().numpy()

            # update global weights
            aggregated_gradient = self._aggregate(gradient_locals, round_idx)
            self.model_trainer.set_model_gradients(
                aggregated_gradient, self.device, weight=1
            )
            w_global = self.model_trainer.get_model_params()

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
            if self.args.overstate or self.args.freerider:
                try:
                    auc_crediblity = roc_auc_score(
                        self.y_adversary, self.pred_credibility
                    )
                    wandb.log(
                        {
                            "Credibility/FreeRider-AUC": auc_crediblity,
                            "round": round_idx,
                        }
                    )
                except:
                    pass

            if self.true_credibility is not None:
                sim_credibility = spearmanr(
                    self.pred_credibility, self.true_credibility
                )[0]
                wandb.log(
                    {"Credibility/Spearmanr": sim_credibility, "round": round_idx}
                )
