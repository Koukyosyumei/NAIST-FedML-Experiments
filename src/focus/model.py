import torch


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
