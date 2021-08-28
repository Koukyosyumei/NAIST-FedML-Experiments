import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))
from fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer


class RFFLTrainer(FedAVGTrainer):
    def update_model(self, weights):
        self.trainer.set_model_gradients(weights)
