import torch.nn as nn
import torch.nn.functional as f


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 46 * 46, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 64 * 46 * 46)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x
