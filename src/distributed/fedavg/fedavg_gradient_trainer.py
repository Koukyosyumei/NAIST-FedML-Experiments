import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")
from core.utils import transform_grad_to_list

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

from fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer


class FedAVGGradTrainer(FedAVGTrainer):
    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        self.trainer.train(self.train_local, self.device, self.args)

        weights = self.trainer.get_model_params()

        # transform Tensor to list
        if self.args.is_mobile == 1:
            weights = transform_grad_to_list(weights)
        return weights, self.local_sample_number
