import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from distributed.fedavg.fedavg_gradient_trainer import FedAVGGradTrainer


class RFFLTrainer(FedAVGGradTrainer):
    def update_model_with_gradients(self, gradients):
        self.trainer.set_model_gradients(gradients, self.device, weight=1)
