import numpy as np
class RolloutBuffer:
    def __init__(self, num_steps, obs_dim,act_dim):
        self.obs = np.zeros((num_steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, act_dim), dtype=np.float32)  
        self.log_probs = np.zeros(num_steps, dtype=np.float32)
        self.rewards = np.zeros(num_steps, dtype=np.float32)
        self.dones = np.zeros(num_steps, dtype=np.float32)
        self.values = np.zeros(num_steps, dtype=np.float32)
        self.idx = 0
        self.num_steps = num_steps

    def push(self, obs, action, log_prob, reward, done, value):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.log_probs[self.idx] = log_prob
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        self.values[self.idx] = value
        self.idx += 1

    def reset(self):
        self.idx = 0

    def compute_returns_and_advantages(self, last_value, gamma, lam):
        adv = np.zeros_like(self.rewards, dtype=np.float32)
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - self.dones[t]
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - self.dones[t+1]
                nextvalues = self.values[t+1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
            adv[t] = lastgaelam
        returns = adv + self.values
        return adv, returns
