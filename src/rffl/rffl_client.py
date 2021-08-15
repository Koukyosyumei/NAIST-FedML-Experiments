import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))
from fedml_api.standalone.fedavg.client import Client


class RFFL_Client(Client):
    def train(self):
        grad_local = self.model_trainer.train(
            self.local_training_data, self.device, self.local_sample_number, self.args
        )
        # grad_local = self.model_trainer.get_model_gradients(gamma=self.args.gamma)
        return (self.local_sample_number, grad_local)

    def download(self, grad_global):
        self.model_trainer.set_model_gradients(
            grad_global, self.device, weight=self.args.agg_weight
        )
