import copy
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.gradient_trainer import GradientModelTrainerCLS


class FreeriderModelTrainer(GradientModelTrainerCLS):
    def __init__(self, model, args=None):
        super().__init__(model, args=args)
        self.prev_model = None

    def set_model_params(self, model_parameters):
        self.prev_model = copy.deepcopy(self.model)
        self.model.load_state_dict(model_parameters)

    def _generate_random_weight(self, weights_k):
        mi = torch.min(weights_k)
        ma = torch.max(weights_k)
        mi = mi.item() if type(mi) == torch.Tensor else mi
        ma = ma.item() if type(ma) == torch.Tensor else ma
        return torch.FloatTensor(weights_k.shape).uniform_(mi, ma)

    def train(self, train_data, device, args):
        fake_gradients = []
        if self.prev_model is None or args.free_rider_strategy == "random":
            for current_param in self.model.parameters():
                fake_gradients.append(
                    self._generate_random_weight(copy.deepcopy(current_param))
                )
        else:
            for prev_param, current_param in zip(
                self.prev_model.parameters(), self.model.parameters()
            ):
                if args.free_rider_strategy == "delta":
                    fake_gradients.append(prev_param - current_param)
                elif args.free_rider_strategy == "advanced-delta":
                    fake_gradients.append(
                        prev_param
                        - current_param
                        + torch.randn_like(prev_param) * args.noise_amp
                    )

        return fake_gradients

    def test(self, test_data, device, args):
        metrics = {"test_correct": -1, "test_loss": 0, "test_total": 0}
        return metrics
