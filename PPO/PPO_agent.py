import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

class RolloutBuffer():
    def __init__(self,max_steps,obs_shape, action_shape, device):
        self.size = max_steps
        self.obs = torch.zeros((max_steps, obs_shape)).to(device)
        self.actions = torch.zeros((max_steps, action_shape)).to(device)
        self.logprobs = torch.zeros((max_steps, action_shape)).to(device)
        self.rewards = torch.zeros((max_steps, 1)).to(device)
        self.dones = torch.zeros((max_steps, 1)).to(device)
        self.values = torch.zeros((max_steps, 1)).to(device)
        self.full = False
        self.idx = 0
    def push(self, obs, action, reward, done, logprob, value):
        if self.idx >= self.size:
            self.idx = 0
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.logprobs[self.idx] = logprob
        self.values[self.idx] = value
        self.idx += 1

    def sample(self):
        return None

class ActorCritic(nn.Module):
    #actor-critic module is a nice way to compute both actor(policy) and critic (estimated value function) at the same time
    def __init__(self, obs_length, actions):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            self.init_layer(nn.Linear(obs_length, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(obs_length, actions),std=0.01)
        )
        self.critic = nn.Sequential(
            self.init_layer(nn.Linear(obs_length, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(obs_length, 1),std=1.)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, actions))

    def forward(self, x):
        action = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action)

        #Calculate normal distribution
        action_std = torch.exp(action_logstd)
        probs = Normal(action, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(0), probs.entropy().sum(0), self.critic(x)
    def get_value(self,x):
        return self.critic(x)
    
    def load_agent(self,input_string):
        self.load_state_dict(torch.load(input_string,weights_only=True))

    def init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
      nn.init.orthogonal_(layer.weight, std)
      nn.init.constant_(layer.bias, bias_const)
      return layer