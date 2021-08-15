import copy
import logging
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from rffl.utils import compute_grad_update

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../FedML/")))

from fedml_api.standalone.fedavg.my_model_trainer_classification import MyModelTrainer


class RFFL_ModelTrainer(MyModelTrainer):
    """
    def get_model_gradients(self, gamma=0.5):
        grads = self.grads
        # logging.info("before normalize")
        # logging.info(grads)
        flattened = flatten(grads)
        norm_value = norm(flattened) + 1e-7  # to prevent division by zero
        grads = unflatten(
            torch.multiply(torch.tensor(gamma), torch.div(flattened, norm_value)),
            grads,
        )
        # logging.info("after normalize")
        # logging.info(grads)
        return grads
    """

    def set_model_gradients(self, gradient, device, weight=1):
        self.model.to(device)
        gradient = [grad.to(device) for grad in gradient]

        for param, grad in zip(self.model.parameters(), gradient):
            param.data += weight * grad.data

    def train(self, train_data, device, local_sample_number, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.wd,
                amsgrad=True,
            )

        epoch_loss = []
        backup = copy.deepcopy(model)

        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # model.zero_grad()
                optimizer.zero_grad()

                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gamma)

                optimizer.step()

                # logging.info("model update")

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

        grads = compute_grad_update(old_model=backup, new_model=model)
        return grads
