import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, class_num=10):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout()
        self.linear1 = nn.Linear(256 * 2 * 2, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.linear2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = x.view(x.size(0), 256 * 2 * 2)

        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.relu6(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        x = self.relu7(x)
        x = self.linear3(x)

        return x
