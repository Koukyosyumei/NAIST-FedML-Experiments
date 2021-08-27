import copy
import logging
import os
import random
import sys

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.gradient_client import GradientClient
from standalone.freerider.freerider_client import FreeRider_Client

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI


class FedAvgGradientAPI(FedAvgAPI):
    def __init__(self, dataset, device, args, model_trainer, true_credibility):
        super().__init__(dataset, device, args, model_trainer)

        print(
            "the number of clients ",
            args.client_num_in_total,
            args.client_num_per_round,
        )
        assert args.client_num_in_total == args.client_num_per_round

        self.true_credibility = true_credibility
        if self.args.overstate:
            self.y_adversary = np.array(true_credibility)
            self.y_adversary = np.where(self.y_adversary < 1, 0, 1)
            self.adversary_idx = np.where(np.array(true_credibility) < 1)[0]

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
                use_gradient=True,
            )

        for client_idx in range(self.args.client_num_per_round):
            c = GradientClient(
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
                auc_crediblity = roc_auc_score(
                    self.y_adversary, -1 * self.rs.to("cpu").detach().numpy()
                )
                wandb.log(
                    {"Credibility/FreeRider-AUC": auc_crediblity, "round": round_idx}
                )
                wandb.log(
                    {
                        "Clients/Surviving FreeRider": len(
                            list(set(self.R_set).intersection(self.adversary_idx))
                        ),
                        "round": round_idx,
                    }
                )

            if self.true_credibility is not None:
                sim_credibility = spearmanr(
                    self.rs.to("cpu").detach().numpy(), self.true_credibility
                )[0]
                wandb.log(
                    {"Credibility/Spearmanr": sim_credibility, "round": round_idx}
                )

    def _aggregate(self, grad_locals, round_idx):
        """Aggerate the gradients sent by the clients
        Args:
            grad_locals: {client_idx: (saple_num, gradient)}
            round_idx: index of the current round

        Returns:
            aggregated_gradient
        """
        aggregated_gradient = [
            torch.zeros(grad.shape).to(self.device)
            for grad in list(grad_locals.values())[0][1]
        ]

        # 各クライアントのデータ数の集計
        training_num = 0
        for client_idx, (sample_num, _) in grad_locals.items():
            training_num += sample_num
        relative_sizes = {
            client_idx: sample_num / training_num
            for client_idx, (sample_num, _) in grad_locals.items()
        }

        # fedavg
        for client_idx, (local_sample_number, gradient) in grad_locals.items():
            for grad_idx in range(len(aggregated_gradient)):
                aggregated_gradient[grad_idx].data += (
                    gradient[grad_idx].data.to(self.device) * relative_sizes[client_idx]
                )
            logging.info("aggregated_gradient")

        return aggregated_gradient
