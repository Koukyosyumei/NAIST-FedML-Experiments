import copy
import os
import sys

import torch
from torch.linalg import norm

from utils import flatten

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))

from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer


class RFFL_ModelTrainer(MyModelTrainer):
    def get_model_gradients(self):
        grads = []
        for param in self.model.parameters():
            grads.append(param.grad)

        return grads

    def set_model_gradients(self, gradient):
        pass
        # flattened = flatten(gradient)
        # norm_value = norm(flattened) + 1e-7
