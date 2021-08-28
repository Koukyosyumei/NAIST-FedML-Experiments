import os
import sys

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.utils import transform_list_to_grad
from distributed.fedavg.fedavg_gradient_aggregator import \
    FedAVGGradientAggregator


class FedAVGAutoEncoderAggregator(FedAVGGradientAggregator):
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
        autoencoder,
        adversary_flag,
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
        self.model_list_history = []
        self.autoencoder = autoencoder
        self.adversary_flag = adversary_flag
        self.pred_credibility = np.zeros(len(adversary_flag))
        self.round_idx = 0

    def anomalydetection(self, sender_id_to_client_index):
        model_list = []
        client_index = []

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_grad(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            client_index.append(sender_id_to_client_index[idx])

        # parameter-checking with autoencoder
        flattend_gradient_locals = torch.stack(
            [
                torch.cat([g.to(self.device).reshape(-1) for g in local_gradient])
                for _, local_gradient in model_list
            ]
        )
        self.autoencoder.fit(flattend_gradient_locals)
        cred = self.autoencoder.predict(flattend_gradient_locals)
        self.pred_credibility[client_index] = cred.to("cpu").detach().numpy()
        auc_crediblity = roc_auc_score(self.adversary_flag, self.pred_credibility)
        wandb.log(
            {
                "Credibility/Adversary-AUC": auc_crediblity,
                "round": self.round_idx,
            }
        )

        self.round_idx += 1
