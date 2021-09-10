import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.utils import transform_grad_to_list

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))
from fedml_api.distributed.fedavg.FedAVGTrainer import FedAVGTrainer


class FedAVGGradTrainer(FedAVGTrainer):
    def update_dataset(self, client_index):
        self.client_index = client_index
        self.trainer.set_id(client_index)
        self.train_local = self.train_data_local_dict[client_index]
        self.local_sample_number = self.train_data_local_num_dict[client_index]
        self.test_local = self.test_data_local_dict[client_index]

    def train(self, round_idx=None):
        self.args.round_idx = round_idx

        grads = self.trainer.train(self.train_local, self.device, self.args)

        # transform Tensor to list
        if self.args.is_mobile == 1:
            grads = transform_grad_to_list(grads)
        return grads, self.local_sample_number
