from ActorCritic import ActorCritic
from RolloutBuffer import RolloutBuffer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
class PPOAgent:
    def __init__(self, env:gym.Env, lr, n_steps, n_epoch,m_batch,clip_eps,vf_coef,e_coef, clip_coef,max_grad, device):
        """
        Inputs: env, lr, num_steps, num_epoch, mini_batch, clip eps, vf_coef, entropy_coef, device
        """
        self.n_steps = n_steps
        self.n_epoch = n_epoch
        self.m_batch = m_batch
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.e_coef = e_coef
        self.clip_coef = clip_coef
        self.max_grad = max_grad
        self.device = device
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.net = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.memory = RolloutBuffer(self.n_steps, obs_dim,act_dim)
        self.act_dim = act_dim

    def select_action(self, obs, multi_env=None):
        obs_v = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std, value = self.net(obs_v)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy()[0], log_prob.item(), value.item()

    def update(self, last_value, gamma, lam):
        advs, returns = self.memory.compute_returns_and_advantages(last_value, gamma, lam)

        obs = torch.FloatTensor(self.memory.obs).to(self.device)
        actions = torch.FloatTensor(self.memory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.memory.log_probs).to(self.device)
        old_values = torch.FloatTensor(self.memory.values).to(self.device)
        returns_v = torch.FloatTensor(returns).to(self.device)
        advantages_v = torch.FloatTensor(advs).to(self.device)
        advantages_v = (advantages_v - advantages_v.mean()) / (advantages_v.std() + 1e-8)

        dataset_size = self.n_steps
        for _ in range(self.n_epoch):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.m_batch):
                end = start + self.m_batch
                mb_idx = indices[start:end]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns_v[mb_idx]
                mb_advantages = advantages_v[mb_idx]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mean, std, value = self.net(mb_obs)
                dist = Normal(mean, std)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_clipped = old_values[mb_idx] + torch.clamp(
                    value - old_values[mb_idx],
                    -self.clip_eps,
                    self.clip_eps,
                )
                value_loss_unclipped = (value - mb_returns) ** 2
                value_loss_clipped = (value_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.e_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.optimizer.step()


        self.memory.reset()
    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device,weights_only=True))
    
