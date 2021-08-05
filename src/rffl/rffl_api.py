import copy
import glob
import logging
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from footprinter import FootPrinter
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))

from fedml_api.standalone.fedavg.client import Client
from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI

from utils import flatten


class RFFLAPI(FedAvgAPI):
    def __init__(self, dataset, device, args, model_trainer):
        super().__init__(dataset, device, args, model_trainer)

        self.rs = torch.zeros(args.client_num_in_total, device=device)
        self.past_phis = []
        self.rs_dict = []
        self.r_threshold = []
        self.qs_dict = []

        self.use_sparsify = True
        self.use_reputation = True

    def _aggregate(self, grad_locals, round_idx):
        aggregated_gradient = [
            torch.zeros(param.shape).to(self.device)
            for param in self.model_trainer.model.parameters()
        ]

        training_num = 0
        for sample_num, _ in grad_locals:
            training_num += sample_num

        if not self.use_reputation:
            # fedavg
            for local_sample_number, gradient in grad_locals:
                weight = local_sample_number / training_num
                for grad_1, grad_2 in zip(aggregated_gradient, gradient):
                    grad_1.data += grad_2.data * weight

        else:
            if round_idx == 0:
                weights = [
                    local_sample_number / training_num
                    for local_sample_number, _ in grad_locals
                ]
            else:
                weights = self.rs

            for gradient, weight in zip(grad_locals, weights):
                for grad_1, grad_2 in zip(aggregated_gradient, gradient):
                    grad_1.data += grad_2.data * weight

            flat_aggre_grad = flatten(aggregated_gradient)
            phis = torch.zeros(self.args.client_num_in_total, device=self.device)
            for i, gradient in enumerate(grad_locals):
                phis[i] = F.cosine_similarity(
                    flatten(gradient), flat_aggre_grad, 0, 1e-10
                )
            self.past_phis.append(phis)

            self.rs = self.alpha * self.rs + (1 - self.alpha) * phis
            for i in range(self.args.client_num_in_total):
                if i not in self.R_set:
                    self.rs[i] = 0
            self.rs = torch.div(self.rs, self.rs.sum())

            if round_idx > 10:
                R_set_copy = copy.deepcopy(self.R_set)
                curr_threshold = self.threshold * (1.0 / len(R_set_copy))

                for i in range(self.args.client_num_in_total):
                    if i in R_set_copy and self.rs[i] < curr_threshold:
                        self.rs[i] = 0
                        self.R_set.remove(i)

            self.rs = torch.div(self.rs, self.rs.sum())
            self.r_threshold.append(self.threshold * (1.0 / len(self.R_set)))
            q_ratios = torch.div(self.rs, torch.max(self.rs))

            self.rs_dict.append(self.rs)
            self.qs_dict.append(q_ratios)

        for i in range(self.args.client_num_in_total):
            if self.use_sparsify and self.use_reputation:
                q_ratio = q_ratios[i]
                reward_gradient = aggregated_gradient

            elif self.use_sparsify and not self.use_reputation:
                reward_gradient = aggregated_gradient

            else:
                reward_gradient = aggregated_gradient

        return reward_gradient
