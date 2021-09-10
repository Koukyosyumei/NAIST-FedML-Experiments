import copy
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
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
        self.client_index = []
        self.prev_client_index = []

    def anomalydetection(self, sender_id_to_client_index):
        client_index = self.client_index
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

    def aggregate(self, sender_id_to_client_index, client_index):
        self.client_index = client_index
        start_time = time.time()
        model_list = []
        training_num = 0

        # self.model_dict: sender_id - 1 to mdoel
        # self.sample_num_dict: sender_id - 1 to sample_num

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_grad(self.model_dict[idx])
            if sender_id_to_client_index[idx + 1] in client_index:
                model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
                training_num += self.sample_num_dict[idx]
                self.model_list_history[sender_id_to_client_index[idx + 1]].append(
                    (self.sample_num_dict[idx], self.model_dict[idx])
                )

        self.round_idx += 1
        if self.round_idx == self.args.comm_round:
            with open(
                f"{self.args.output_dir}/model_list_history.pickle", mode="wb"
            ) as f:
                logging.info("saving history")
                pickle.dump(self.model_list_history, f)

        logging.info("len of model_list = " + str(len(model_list)))

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

        # update the global model which is cached at the server side
        self.set_global_model_gradients(averaged_gradient, self.device, weight=1)
        averaged_params = self.get_global_model_params()

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params
