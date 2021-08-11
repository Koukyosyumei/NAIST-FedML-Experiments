import logging
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../FedML/")))

from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow


class MyNet(torch.nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=1000, output_dim=10):
        super(MyNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x


class CNN_Net(nn.Module):
    def __init__(self, device=None):
        super(CNN_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 16, 7, 1)
        self.fc1 = nn.Linear(4 * 4 * 16, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 1, 32, 32)
        x = torch.tanh(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = torch.tanh(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 16)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def create_model(args, model_name, output_dim):
    logging.info(
        "create_model. model_name = %s, output_dim = %s" % (model_name, output_dim)
    )
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == "nn":
        model = MyNet()
    return model
