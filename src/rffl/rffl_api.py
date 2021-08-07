import logging
import os
import sys

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI

from rffl_client import RFFL_Client
from utils import flatten, mask_grad_update_by_order


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

        self.rs = torch.zeros(args.client_num_in_total, device=device)
        self.past_phis = []
        self.rs_dict = []
        self.r_threshold = []
        self.qs_dict = []

        self.R_set = list(range(args.client_num_in_total))

        self.use_sparsify = args.use_sparsify
        self.use_reputation = args.use_reputation
        self.threshold = 1 / (3 * args.client_num_in_total)
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

            sim_credibility = spearmanr(
                self.rs.to("cpu").detach().numpy(), self.true_credibility
            )[0]
            wandb.log({"Credibility/Spearmanr": sim_credibility, "round": round_idx})
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
                for grad_1, grad_2 in zip(aggregated_gradient, gradient):
                    grad_1.data += grad_2.data * relative_sizes[client_idx]

        else:
            # Aggregation
            for client_idx, (_, gradient) in grad_locals.items():
                for grad_1, grad_2 in zip(aggregated_gradient, gradient):
                    if round_idx == 0:
                        grad_1.data += grad_2.data * relative_sizes[client_idx]
                    else:
                        grad_1.data += grad_2.data * self.rs[client_idx]
            flat_aggre_grad = flatten(aggregated_gradient)

            # culculate the reputations
            curr_threshold = self.threshold * (1.0 / len(self.R_set))
            phis = torch.zeros(self.args.client_num_in_total, device=self.device)

            for client_idx, (c_idx, (_, gradient)) in zip(
                self.R_set, grad_locals.items()
            ):
                assert client_idx == c_idx
                phis[client_idx] = F.cosine_similarity(
                    flatten(gradient), flat_aggre_grad, 0, 1e-10
                )
                self.rs[client_idx] = (
                    self.alpha * self.rs[client_idx]
                    + (1 - self.alpha) * phis[client_idx]
                )
            self.past_phis.append(phis)

            # remove the unuseful cilents
            self.rs = torch.div(self.rs, self.rs.sum())
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

            for client_idx, reward_gradient in reward_gradients.items():
                for grad_1, grad_2 in zip(reward_gradient, grad_locals[client_idx][1]):
                    if round_idx == 0:
                        grad_1.data -= grad_2.data * relative_sizes[client_idx]
                    else:
                        grad_1.data -= grad_2.data * self.rs[client_idx]

        return reward_gradients
