import numpy as np
import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.Tanh())
            last = h
        self.shared = nn.Sequential(*layers)

        # Continuous actor
        self.mean = nn.Linear(last, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic
        self.value = nn.Linear(last, 1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, x):
        x = self.shared(x)
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        value = self.value(x).squeeze(-1)
        return mean, std, value
