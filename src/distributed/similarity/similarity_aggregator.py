import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.utils import transform_list_to_grad
from distributed.fedavg.fedavg_gradient_aggregator import FedAVGGradientAggregator

EPS = 1e-10


class FedAVGSimilarityAggregator(FedAVGGradientAggregator):
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
        self.alpha = self.args.alpha

        self.pred_credibility = np.zeros_like(adversary_flag).astype(float)
        self.rs = torch.zeros(args.client_num_in_total, device=device)

    def _update_reputations(self, client_index, model_list, averaged_gradient):
        # culculate the reputations
        flatten_averaged_gradient = torch.cat(
            [g.to(self.device).view(-1) for g in averaged_gradient]
        )
        phis = torch.zeros(self.args.client_num_in_total, device=self.device)
        for c_idx, local_gradient in zip(client_index, model_list):
            flatten_local_gradient = torch.cat(
                [g.to(self.device).view(-1) for g in local_gradient[1]]
            )
            phis[c_idx] = F.cosine_similarity(
                flatten_local_gradient, flatten_averaged_gradient, 0, EPS
            )
            self.rs[c_idx] = (
                self.alpha * self.rs[c_idx] + (1 - self.alpha) * phis[c_idx]
            )
        self.rs = torch.div(self.rs, self.rs.sum())

    def anomalydetection(self, sender_id_to_client_index):
        self.pred_credibility = self.rs.detach().cpu().numpy()
        auc_crediblity = roc_auc_score(self.adversary_flag, -self.pred_credibility)
        wandb.log(
            {
                "Credibility/Adversary-AUC": auc_crediblity,
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
            self.model_list_history[sender_id_to_client_index[idx + 1]].append(
                (self.sample_num_dict[idx], self.model_dict[idx])
            )
            client_index.append(sender_id_to_client_index[idx + 1])

        self.round_idx += 1
        if self.round_idx == self.args.comm_round:
            with open(
                f"{self.args.output_dir}/model_list_history.pickle", mode="wb"
            ) as f:
                logging.info("saving history")
                pickle.dump(self.model_list_history, f)

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        averaged_gradient = [
            torch.zeros(grad.shape).to(self.device) for grad in model_list[0][1]
        ]

        for i in range(0, len(model_list)):
            local_sample_number, local_gradient = model_list[i]
            w = local_sample_number / training_num
            for grad_idx in range(len(averaged_gradient)):
                averaged_gradient[grad_idx].data += (
                    local_gradient[grad_idx].data.to(self.device) * w
                )

        self._update_reputations(client_index, model_list, averaged_gradient)

        # update the global model which is cached at the server side
        self.set_global_model_gradients(averaged_gradient, self.device, weight=1)
        averaged_params = self.get_global_model_params()

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params
