import copy
import logging
import os
import sys

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

import random

import numpy as np
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from standalone.freerider.freerider_client import FreeRider_Client
from standalone.rffl.rffl_client import RFFL_Client
from standalone.rffl.utils import flatten, mask_grad_update_by_order


class RFFLAPI(FedAvgAPI):
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

        self.rs = torch.zeros(args.client_num_in_total, device=device)
        self.past_phis = []
        self.rs_dict = []
        self.r_threshold = []
        self.qs_dict = []

        self.R_set = list(range(args.client_num_in_total))

        self.use_sparsify = args.use_sparsify
        self.use_reputation = args.use_reputation
        self.threshold = 1 / (10 * args.client_num_in_total)
        self.warm_up = 10
        self.alpha = 0.95

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

        self.client_idx_to_statedict = {}
        for client_idx in range(self.args.client_num_per_round):
            c = RFFL_Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
            self.client_idx_to_statedict[
                client_idx
            ] = c.model_trainer.get_model_params()
        logging.info("############setup_clients (END)#############")

    def train(self):
        client_idx_to_reward_gradients = {}

        for round_idx in range(self.args.comm_round):

            logging.info("################Communication round : {}".format(round_idx))

            gradient_locals = {}

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset
            """
            logging.info("client_indexes = " + str(self.R_set))

            for client_idx in self.R_set:
                if self.args.freerider and client_idx in self.adversary_idx:
                    self.freerider.update_local_dataset(
                        client_idx,
                        self.train_data_local_dict[client_idx],
                        self.test_data_local_dict[client_idx],
                        self.train_data_local_num_dict[client_idx],
                    )

                    if round_idx > 0:
                        # train on new dataset
                        grad = self.freerider.train(
                            client_idx_to_reward_gradients[client_idx]
                        )
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
                    # そのクライアントのパラメータをセット
                    client.model_trainer.set_model_params(
                        self.client_idx_to_statedict[client_idx]
                    )
                    # 前回のラウンドで計算したgradientで、パラメータを更新
                    if round_idx > 0:
                        client.download(client_idx_to_reward_gradients[client_idx])
                    # 学習
                    grad = client.train()
                    # 更新したパラメータを保存
                    self.client_idx_to_statedict[
                        client_idx
                    ] = client.model_trainer.get_model_params()

                # upload
                gradient_locals[client_idx] = grad

            # update global weights
            client_idx_to_reward_gradients = self._aggregate(gradient_locals, round_idx)

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

            wandb.log({"Clients/R_set": len(self.R_set), "round": round_idx})

    def _aggregate(self, grad_locals, round_idx):
        """Aggerate the gradients sent by the clients
        Args:
            grad_locals: {client_idx: (saple_num, gradient)}
            round_idx: index of the current round

        Returns:
            reward_gradients: {client_idx: gradient}
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

        if not self.use_reputation:
            # fedavg
            for client_idx, (local_sample_number, gradient) in grad_locals.items():
                for grad_idx in range(len(aggregated_gradient)):
                    aggregated_gradient[grad_idx].data += (
                        gradient[grad_idx].data.to(self.device)
                        * relative_sizes[client_idx]
                    )
            logging.info("aggregated_gradient")
            logging.info(aggregated_gradient)

        else:
            # Aggregation
            for client_idx, (_, gradient) in grad_locals.items():
                for grad_idx in range(len(aggregated_gradient)):
                    if round_idx == 0:
                        aggregated_gradient[grad_idx].data += (
                            gradient[grad_idx].data.to(self.device)
                            * relative_sizes[client_idx]
                        )
                    else:
                        aggregated_gradient[grad_idx].data += (
                            gradient[grad_idx].data.to(self.device)
                            * self.rs[client_idx]
                        )
            flat_aggre_grad = flatten(aggregated_gradient)
            logging.info("flat_agg_grad")
            logging.info(flat_aggre_grad)

            # culculate the reputations
            curr_threshold = self.threshold * (1.0 / len(self.R_set))
            phis = torch.zeros(self.args.client_num_in_total, device=self.device)

            for client_idx, (c_idx, (_, gradient)) in zip(
                self.R_set, grad_locals.items()
            ):
                assert client_idx == c_idx
                phis[client_idx] = F.cosine_similarity(
                    flatten(gradient).to(self.device), flat_aggre_grad, 0, 1e-10
                )
                self.rs[client_idx] = (
                    self.alpha * self.rs[client_idx]
                    + (1 - self.alpha) * phis[client_idx]
                )
            self.past_phis.append(phis)
            self.rs = torch.div(self.rs, self.rs.sum())

            # remove the unuseful cilents
            if self.args.remove:
                if round_idx > self.warm_up:
                    for client_idx in self.R_set:
                        if self.rs[client_idx] < curr_threshold:
                            self.rs[client_idx] = 0
                            self.R_set.remove(client_idx)
                self.rs = torch.div(self.rs, self.rs.sum())

            self.r_threshold.append(self.threshold * (1.0 / len(self.R_set)))
            q_ratios = torch.div(self.rs, torch.max(self.rs))
            self.rs_dict.append(self.rs)
            self.qs_dict.append(q_ratios)

        # Download
        reward_gradients = {}
        for client_idx in self.R_set:
            if self.use_sparsify and self.use_reputation:
                q_ratio = q_ratios[client_idx]
                reward_gradients[client_idx] = mask_grad_update_by_order(
                    aggregated_gradient, mask_percentile=q_ratio, mode="layer"
                )

            elif self.use_sparsify and not self.use_reputation:
                reward_gradients[client_idx] = mask_grad_update_by_order(
                    aggregated_gradient,
                    mask_percentile=relative_sizes[client_idx],
                    mode="layer",
                )

            else:
                reward_gradients[client_idx] = aggregated_gradient

            for grad_idx in range(len(reward_gradients[client_idx])):
                if round_idx == 0 or not self.use_reputation:
                    reward_gradients[client_idx][grad_idx].data -= (
                        grad_locals[client_idx][1][grad_idx].data.to(self.device)
                        * relative_sizes[client_idx]
                    )
                else:
                    reward_gradients[client_idx][grad_idx].data -= (
                        grad_locals[client_idx][1][grad_idx].data.to(self.device)
                        * self.rs[client_idx]
                    )

        logging.info("rs")
        logging.info(self.rs)

        return reward_gradients

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        client = self.client_list[0]

        for client_idx in self.R_set:
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
