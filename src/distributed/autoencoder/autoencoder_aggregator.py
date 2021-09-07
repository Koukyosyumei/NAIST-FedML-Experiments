import logging
import os
import sys

import numpy as np
import torch
import wandb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.utils import transform_list_to_grad
from distributed.fedavg.fedavg_gradient_aggregator import FedAVGGradientAggregator
from standalone.autoencoder.detector import STD_DAGMM, STD_NUM_DAGMM


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
        self.pred_credibility = np.zeros_like(adversary_flag).astype(float)

        self.num_parameters = torch.cat(
            [p.reshape(-1) for p in model_trainer.model.parameters()]
        ).shape[-1]

        if self.args.autoencoder_type == "STD-DAGMM":
            self.autoencoder = STD_DAGMM(self.num_parameters, device)
        elif self.args.autoencoder_type == "STD-NUM-DAGMM":
            self.autoencoder = STD_NUM_DAGMM(self.num_parameters, device)

    def anomalydetection(self, sender_id_to_client_index):
        model_list = []
        client_index = []

        logging.info(f"sender_id_to_client_index = {sender_id_to_client_index}")

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_grad(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            client_index.append(sender_id_to_client_index[idx + 1])

        # parameter-checking with autoencoder
        flattend_gradient_locals = torch.stack(
            [
                torch.cat([g.to(self.device).reshape(-1) for g in local_gradient])
                for _, local_gradient in model_list
            ]
        )
        local_num_tensor = (
            torch.Tensor([num for num, _ in model_list]).to(self.device).float()
        )
        normalized_local_num_tensor = local_num_tensor / local_num_tensor.sum()

        if self.args.autoencoder_type == "STD-DAGMM":
            self.autoencoder.fit(
                flattend_gradient_locals,
                epochs=self.args.autoencoder_epochs,
                lr=self.args.autoencoder_lr,
            )
            cred = self.autoencoder.predict(flattend_gradient_locals)

        elif self.args.autoencoder_type == "STD-NUM-DAGMM":
            self.autoencoder.fit(
                flattend_gradient_locals,
                normalized_local_num_tensor,
                epochs=self.args.autoencoder_epochs,
                lr=self.args.autoencoder_lr,
            )
            cred = self.autoencoder.predict(
                flattend_gradient_locals, normalized_local_num_tensor
            )

        self.pred_credibility[client_index] = cred.to("cpu").detach().numpy()

        print(self.pred_credibility)

        auc_crediblity = roc_auc_score(self.adversary_flag, self.pred_credibility)
        wandb.log(
            {
                "Credibility/Adversary-AUC": auc_crediblity,
                "round": self.round_idx,
            }
        )
