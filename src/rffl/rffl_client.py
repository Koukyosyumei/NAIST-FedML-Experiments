import copy
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))
from fedml_api.standalone.fedavg.client import Client


class RFFL(Client):
    def train(self, grad_global):
        self.model_trainer.set_model_params(grad_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        weights = self.model_trainer.get_model_params()
        return weights
