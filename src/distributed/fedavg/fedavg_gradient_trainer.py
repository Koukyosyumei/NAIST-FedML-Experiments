import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.utils import transform_grad_to_list

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))
from fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer


class FedAVGGradTrainer(FedAVGTrainer):
    def train(self, round_idx=None):
        self.args.round_idx = round_idx
        grads = self.trainer.train(self.train_local, self.device, self.args)

        # transform Tensor to list
        if self.args.is_mobile == 1:
            grads = transform_grad_to_list(grads)
        return grads, self.local_sample_number
