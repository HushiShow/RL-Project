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
        self.mean = nn.Sequential(
            nn.LayerNorm((obs_dim)),
            self._init_layers(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            self._init_layers(nn.Linear(64,64)),
            nn.Tanh(),
            self._init_layers(nn.Linear(64,64)),
            nn.Tanh(),
            self._init_layers(nn.Linear(64, act_dim), std=0.01)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Critic
        self.value = nn.Sequential(
            nn.LayerNorm((obs_dim)),
            self._init_layers(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            self._init_layers(nn.Linear(64,64)),
            nn.Tanh(),
            self._init_layers(nn.Linear(64,64)),
            nn.Tanh(),
            self._init_layers(nn.Linear(64, 1), std=1)
        )

    def _init_layers(self,layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x):
        mean = self.mean(x)
        std = self.log_std.exp().expand_as(mean)
        value = self.value(x).squeeze(-1)
        return mean, std, value
