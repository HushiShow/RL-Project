import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    #actor-critic module is a nice way to compute both actor(policy) and critic (estimated value function) at the same time
    def __init__(self, obs_space=3, action_space=1):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = nn.Sequential(
            self.init_layer(nn.Linear(obs_space, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(obs_space, action_space), std=0.01)
        )
        self.critic = nn.Sequential(
            self.init_layer(nn.Linear(obs_space, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(obs_space, 1), std=1.)
        )

    def init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x):
        #h in this case is image features from a game, do we really need that for the pendulum?
        h = self.head(x)
        return self.actor(h), self.critic(h)

    def config_optimizer(self):
        #default settings for now
        return torch.optim.Adam(self.parameters(), lr=2.5e-4, eps=1e-5)

    #maybe make this into a storage class or move to agent(actor_critic init and store there)
    #num_steps = 128, num_envs = 1 for now, lets keep it simple

class RolloutBuffer(nn.Module):
    def __init__(self, obs_space=3, action_space=1, num_steps=128, num_envs=1):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.global_step = 0

        #init the rollout buffer
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.obs_space).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.action_space).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

    def push(self, ):
        pass

    def pop(self):
        pass

