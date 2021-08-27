import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))
from fedml_api.standalone.fedavg.client import Client


class GradientClient(Client):
    def train(self, w_global):
        self.model_trainer.set_model_params(w_global)
        grad_local = self.model_trainer.train(
            self.local_training_data, self.device, self.local_sample_number, self.args
        )
        return (self.local_sample_number, grad_local)

    def download(self, grad_global):
        self.model_trainer.set_model_gradients(grad_global, self.device, weight=1)
