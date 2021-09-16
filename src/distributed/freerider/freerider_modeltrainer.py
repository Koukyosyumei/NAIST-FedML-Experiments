import copy
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
from core.gradient_trainer import GradientModelTrainerCLS, GradientModelTrainerNWP


class FreeriderModelTrainerCLS(GradientModelTrainerCLS):
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
        return torch.FloatTensor(weights_k.shape).uniform_(0, 0)

    def get_diff_gradients(self, train_data, device, args):
        fake_gradients = []
        if self.prev_model is None or args.free_rider_strategy == "random":
            for current_param in self.model.parameters():
                fake_gradients.append(
                    self._generate_random_weight(copy.deepcopy(current_param)).to(
                        device
                    )
                )
        else:
            for prev_param, current_param in zip(
                self.prev_model.parameters(), self.model.parameters()
            ):
                if args.free_rider_strategy == "delta":
                    fake_gradients.append(prev_param - current_param)
                elif args.free_rider_strategy == "advanced-delta":
                    fake_gradients.append(
                        prev_param.to(device)
                        - current_param.to(device)
                        + torch.randn_like(prev_param).to(device) * args.noise_amp
                    )

        return fake_gradients

    def train(self, train_data, device, args, weight=0.2):
        final_grads = []
        diff_grads = self.get_diff_gradients(train_data, device, args)
        grads = super().train(train_data, device, args)
        for g, dg in zip(grads, diff_grads):
            final_grads.append(weight * g.to(device) + (1 - weight) * dg.to(device))
        return final_grads


class FreeriderModelTrainerNWP(GradientModelTrainerNWP):
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
        return torch.FloatTensor(weights_k.shape).uniform_(0, 0)

    def get_diff_gradients(self, train_data, device, args):
        fake_gradients = []
        if self.prev_model is None or args.free_rider_strategy == "random":
            for current_param in self.model.parameters():
                fake_gradients.append(
                    self._generate_random_weight(copy.deepcopy(current_param)).to(
                        device
                    )
                )
        else:
            for prev_param, current_param in zip(
                self.prev_model.parameters(), self.model.parameters()
            ):
                if args.free_rider_strategy == "delta":
                    fake_gradients.append((prev_param - current_param).to(device))
                elif args.free_rider_strategy == "advanced-delta":
                    fake_gradients.append(
                        prev_param.to(device)
                        - current_param.to(device)
                        + torch.randn_like(prev_param).to(device) * args.noise_amp
                    )

        return fake_gradients

    def train(self, train_data, device, args, weight=0.2):
        final_grads = []
        diff_grads = self.get_diff_gradients(train_data, device, args)
        grads = super().train(train_data, device, args)
        for g, dg in zip(grads, diff_grads):
            final_grads.append(weight * g.to(device) + (1 - weight) * dg.to(device))
        return final_grads
