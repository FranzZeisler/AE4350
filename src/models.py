
import torch.nn as nn
import torch

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(MLPPolicy, self).__init__()
        self.layer1 = nn.Linear(obs_dim, 256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class MLPQ(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(MLPQ, self).__init__()
        self.layer1 = nn.Linear(obs_dim + action_dim, 256)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=-1)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)
