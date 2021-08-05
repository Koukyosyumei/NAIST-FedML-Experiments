import copy
import glob
import logging
import math
import os
import random
import sys

import numpy as np
import torch
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
    def _aggregate(self, grad_locals, round_idx):
        aggregated_gradient = [
            torch.zeros(param.shape).to(self.device)
            for param in self.model_trainer.model.parameters()
        ]

        if not self.use_reputation:
            # fedavg
            for gradient, weight in zip(grad_locals, self.relative_shard_sizes):
                for grad_1, grad_2 in zip(aggregated_gradient, gradient):
                    grad_1.data += grad_2.data * self.weight

        else:
            if round_idx == 0:
                weights = torch.div(self.shard_sizes, torch.sum(self.shard_sizes))
            else:
                weights = self.rs

            for gradient, weight in zip(grad_locals, weights):
                for grad_1, grad_2 in zip(aggregated_gradient, gradient):
                    grad_1.data += grad_2.data * self.weight

            flat_aggre_grad = flatten(aggregated_gradient)
            phis = torch.zeros(self.args.client_num_in_total, device=self.device)
            for i, gradient in enumerate(grad_locals):
                phis[i] = F.cosine_similarity(
                    flatten(gradient), flat_aggre_grad, 0, 1e-10
                )
            self.past_phis.append(phis)

            rs = self.alpha * rs + (1 - self.alpha) * phis
            for i in range(self.args.client_num_in_total):
                if i not in self.R_set:
                    rs[i] = 0
            rs = torch.div(rs, rs.sum())

            if round_idx > 10:
                R_set_copy = copy.deepcopy(self.R_set)
                curr_threshold = threshold * (1.0 / len(R_set_copy))

                for i in range(self.args.client_num_in_total):
                    if i in R_set_copy and rs[i] < curr_threshold:
                        rs[i] = 0
                        R_set.remove(i)
