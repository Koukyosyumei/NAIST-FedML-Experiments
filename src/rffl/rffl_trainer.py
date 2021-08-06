import logging
import os
import sys

import torch
from torch import nn
from torch.linalg import norm

from utils import flatten, unflatten

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))

from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer


class RFFL_ModelTrainer(MyModelTrainer):
    def get_model_gradients(self, gamma=0.5, weight=0.15):
        grads = [weight * g for g in self.grads]
        """
        flattened = flatten(grads)
        norm_value = norm(flattened) + 1e-7  # to prevent division by zero
        grads = unflatten(
            torch.multiply(torch.tensor(gamma), torch.div(flattened, norm_value)),
            grads,
        )
        """
        return grads

    def set_model_gradients(self, gradient, device, weight=1):
        gradient = [grad.to(device) for grad in gradient]

        for param, grad in zip(self.model.parameters(), gradient):
            param.data += weight * grad.data
        logging.info("update parameters !!!!!!!!!!!!!!!")

    def train(self, train_data, device, local_sample_number, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)

        epoch_loss = []
        self.grads = [
            torch.zeros(param.shape).to(device) for param in self.model.parameters()
        ]
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                temp_grad = [param.grad for param in model.parameters()]
                for param_idx, tg in enumerate(temp_grad):
                    self.grads[param_idx] += (x.shape[0] / local_sample_number) * tg

                logging.info(self.grads[0])

                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(
                "Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                    self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                )
            )
