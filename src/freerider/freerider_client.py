import copy
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))
from fedml_api.standalone.fedavg.client import Client


class FreeRider_Client(Client):
    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer,
        noise_amp=0.001,
        use_gradient=False,
    ):
        super().__init__(
            client_idx,
            local_training_data,
            local_test_data,
            local_sample_number,
            args,
            device,
            model_trainer,
        )
        self.prev_weights = None
        self.use_gradient = use_gradient
        self.noise_amp = noise_amp

    def _generate_random_weight(self, weights_k):
        mi = torch.min(weights_k)
        ma = torch.max(weights_k)
        mi = mi.item() if type(mi) == torch.Tensor else mi
        ma = ma.item() if type(ma) == torch.Tensor else ma
        return torch.FloatTensor(weights_k.shape).uniform_(mi, ma)

    def train(self, w_global):
        if self.use_gradient:
            if w_global is not None:
                prev_weights = copy.deepcopy(w_global)
                weights = copy.deepcopy(w_global)
                iterator = enumerate(w_global)
            else:
                weights = self.model_trainer.model.parameters()
                weights = [w for w in weights]
                prev_weights = None
                iterator = enumerate(weights)
        else:
            self.model_trainer.set_model_params(w_global)
            # self.model_trainer.train(self.local_training_data, self.device, self.args)
            weights = self.model_trainer.get_model_params()
            prev_weights = copy.deepcopy(weights)
            iterator = weights.items()

        for k, w in iterator:

            if self.args.free_rider_strategy == "random":
                weights[k] = self._generate_random_weight(copy.deepcopy(weights[k]))

            elif self.args.free_rider_strategy == "delta":
                if self.prev_weights is not None:
                    weights[k] = self.prev_weights[k] - weights[k]
                else:
                    weights[k] = self._generate_random_weight(copy.deepcopy(weights[k]))

            elif self.args.free_rider_strategy == "advanced-delta":
                if self.prev_weights is not None:
                    weights[k] = self.prev_weights[k] - weights[k]
                else:
                    weights[k] = self._generate_random_weight(copy.deepcopy(weights[k]))

                weights[k] += torch.randn_like(weights[k]) * self.noise_amp

            weights[k] = weights[k].to("cpu")

        self.prev_weights = prev_weights

        if self.use_gradient:
            return (self.local_sample_number, weights)
        else:
            return weights
