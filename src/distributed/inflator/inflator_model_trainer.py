import copy
import logging
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML/")))

from fedml_api.standalone.fedavg.my_model_trainer_classification import (
    MyModelTrainer as MyModelTrainerCLS,
)


class InflatorModelTrainerCLS(MyModelTrainerCLS):
    def __init__(self, model, transform=None, times=1, args=None):
        self.model = model
        self.id = 0
        self.transform = transform
        self.times = times
        self.args = args

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.lr,
                weight_decay=args.wd,
                amsgrad=True,
            )

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
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

        for t in range(self.times):
            torch.manual_seed(t)
            augmented_data = copy.deepcopy(train_data)
            augmented_data.dataset.transform = self.transform

            epoch_loss = []
            for epoch in range(args.epochs):
                batch_loss = []
                for batch_idx, (x, labels) in enumerate(train_data):
                    x, labels = x.to(device), labels.to(device)
                    model.zero_grad()
                    log_probs = model(x)
                    loss = criterion(log_probs, labels)
                    loss.backward()

                    # Uncommet this following line to avoid nan loss
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()
                    # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                    #            100. * (batch_idx + 1) / len(train_data), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info(
                    "Client Index = {}\tEpoch: {}\tLoss with augmentation: {:.6f}".format(
                        self.id, epoch, sum(epoch_loss) / len(epoch_loss)
                    )
                )
