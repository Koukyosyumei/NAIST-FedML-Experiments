import copy
import os
import sys

import numpy as np
import wandb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.utils import transform_list_to_grad
from distributed.fedavg.fedavg_gradient_aggregator import FedAVGGradientAggregator


class FedAVGQualityInferenceAggregator(FedAVGGradientAggregator):
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

        self.acc_improve_curr = 0
        self.acc_improve_prev = 0
        self.acc_curr = 0
        self.acc_prev = 0
        self.prev_client_index = None

    def anomalydetection(self, sender_id_to_client_index):
        client_index = []
        for idx in range(self.worker_num):
            client_index.append(sender_id_to_client_index[idx + 1])

        test_num_samples = []
        test_tot_corrects = []
        metrics = self.trainer.test(self.test_global, self.device, self.args)
        test_tot_correct, test_num_sample = (
            metrics["test_correct"],
            metrics["test_total"],
        )
        test_tot_corrects.append(copy.deepcopy(test_tot_correct))
        test_num_samples.append(copy.deepcopy(test_num_sample))

        # test on test dataset
        test_acc = sum(test_tot_corrects) / sum(test_num_samples)

        self.acc_prev = self.acc_curr
        self.acc_curr = test_acc
        self.acc_improve_prev = self.acc_improve_curr
        self.acc_improve_curr = (
            self.acc_curr - self.acc_prev if self.round_idx > 0 else 0
        )

        if self.round_idx > 0 and self.acc_improve_curr > self.acc_improve_prev:
            self.pred_credibility[client_index] += 1
            self.pred_credibility[self.prev_client_index] -= 1
        if self.acc_improve_curr < 0:
            self.pred_credibility[client_index] -= 1

        auc_crediblity = roc_auc_score(self.adversary_flag, -self.pred_credibility)
        wandb.log(
            {
                "Credibility/Adversary-AUC": auc_crediblity,
                "round": self.round_idx,
            }
        )

        self.prev_client_index = client_index
