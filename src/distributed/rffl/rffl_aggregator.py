import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score

from .rffl_utils import mask_grad_update_by_order

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.utils import transform_list_to_grad
from distributed.fedavg.fedavg_gradient_aggregator import \
    FedAVGGradientAggregator

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))
from fedml_api.distributed.fedavg.FedAVGAggregator import FedAVGAggregator
from fedml_api.distributed.fedavg.utils import transform_list_to_tensor

EPS = 1e-10


class RFFLAggregator(FedAVGGradientAggregator):
    def __init__(
        self,
        train_global,
        test_global,
        all_train_data_num,
        train_data_local_dict,
        test_data_local_dict,
        train_data_local_num_dict,
        worker_num,
        device,
        args,
        model_trainer,
        adversary_flag=None,
    ):
        super().__init__(
            train_global,
            test_global,
            all_train_data_num,
            train_data_local_dict,
            test_data_local_dict,
            train_data_local_num_dict,
            worker_num,
            device,
            args,
            model_trainer,
        )
        self.adversary_flag = adversary_flag
        self.adversary_idx = [i for i, f in enumerate(adversary_flag) if f == 1]
        self.pred_credibility = np.zeros_like(adversary_flag)

        self.rs = torch.zeros(args.client_num_in_total, device=device)
        self.R_set = list(range(args.client_num_in_total))
        self.relative_size = [0] * args.client_num_in_total
        self.threshold = 1 / (10 * args.client_num_in_total)
        self.warm_up = self.args.warm_up
        self.alpha = self.args.alpha
        self.sparcity = self.args.sparcity
        self.remove = self.args.remove

    def _update_reputations(self, client_index, model_list, averaged_gradient):
        # culculate the reputations
        flatten_averaged_gradient = torch.cat(
            [g.to(self.device).reshape(-1) for g in averaged_gradient]
        )
        phis = torch.zeros(self.args.client_num_in_total, device=self.device)
        for c_idx, local_gradient in zip(client_index, model_list):
            flatten_local_gradient = torch.cat(
                [g.to(self.device).reshape(-1) for g in local_gradient[1]]
            )
            phis[c_idx] = F.cosine_similarity(
                flatten_local_gradient, flatten_averaged_gradient, 0, EPS
            )
            self.rs[c_idx] = (
                self.alpha * self.rs[c_idx] + (1 - self.alpha) * phis[c_idx]
            )
        self.rs = torch.div(self.rs, self.rs.sum())

    def _remove(self):
        # remove the unuseful cilents
        curr_threshold = self.threshold * (1.0 / len(self.R_set))
        if self.remove == 1 and self.round_idx > self.warm_up:
            for client_idx in self.R_set:
                if self.rs[client_idx] < curr_threshold:
                    self.rs[client_idx] = 0
                    self.R_set.remove(client_idx)
            self.rs = torch.div(self.rs, self.rs.sum())
        else:
            pass

    def _get_reward_gradiets(self, averaged_gradient, sender_id_to_client_index):
        client_index_to_sender_id = {
            v: k - 1 for k, v in sender_id_to_client_index.items()
        }
        q_ratios = torch.div(self.rs, torch.max(self.rs))

        reward_gradients = {}
        for client_idx in self.R_set:
            q_ratio = q_ratios[client_idx]
            if self.sparcity == 1:
                reward_gradients[client_idx] = mask_grad_update_by_order(
                    averaged_gradient, mask_percentile=q_ratio, mode="layer"
                )
            else:
                reward_gradients[client_idx] = averaged_gradient

            for grad_idx in range(len(reward_gradients[client_idx])):
                if self.round_idx == 0:
                    w = self.relative_size[client_idx]
                else:
                    w = self.rs[client_idx]

                reward_gradients[client_idx][grad_idx].data -= (
                    self.model_dict[client_index_to_sender_id[client_idx]][
                        grad_idx
                    ].data.to(self.device)
                    * w
                )

        return reward_gradients

    def anomalydetection(self, sender_id_to_client_index):
        self.pred_credibility = self.rs.detach().cpu().numpy()
        auc_crediblity = roc_auc_score(self.adversary_flag, self.pred_credibility)
        wandb.log(
            {
                "Credibility/Adversary-AUC": auc_crediblity,
                "round": self.round_idx,
            }
        )
        wandb.log({"Clients/R_set": len(self.R_set), "round": self.round_idx})
        wandb.log(
            {
                "Clients/Surviving Adversaries": len(
                    list(set(self.R_set).intersection(self.adversary_idx))
                ),
                "round": self.round_idx,
            }
        )

    def aggregate(self, sender_id_to_client_index):
        start_time = time.time()
        model_list = []
        client_index = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_grad(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]
            client_index.append(sender_id_to_client_index[idx + 1])

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        averaged_gradient = [
            torch.zeros(grad.shape).to(self.device) for grad in model_list[0][1]
        ]

        for i in range(0, len(model_list)):
            local_sample_number, local_gradient = model_list[i]
            if self.round_idx == 0:
                w = local_sample_number / training_num
                self.relative_size[client_index[i]] = w
            else:
                w = self.rs[client_index[i]]
            for grad_idx in range(len(averaged_gradient)):
                averaged_gradient[grad_idx].data += (
                    local_gradient[grad_idx].data.to(self.device) * w
                )

        self._update_reputations(client_index, model_list, averaged_gradient)
        self._remove()
        reward_gradients = self._get_reward_gradiets(
            averaged_gradient, sender_id_to_client_index
        )

        # update the global model which is cached at the server side
        self.set_global_model_gradients(averaged_gradient, self.device, weight=1)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))

        self.round_idx += 1

        return reward_gradients
