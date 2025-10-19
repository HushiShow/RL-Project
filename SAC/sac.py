
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os
import gymnasium as gym


class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
    
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256,
                 name='critic', ckpt_dir='sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.ckpt_dir = ckpt_dir
        self.ckpt_file = os.path.join(self.ckpt_dir, name)

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q   = nn.Linear(self.fc2_dims, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(torch.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save(self):
        torch.save(self.state_dict(), self.ckpt_file)

    def load(self):
        self.load_state_dict(torch.load(self.ckpt_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims=256, fc2_dims=256,
                 name='value', ckpt_dir='sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.ckpt_dir = ckpt_dir
        self.ckpt_file = os.path.join(self.ckpt_dir, name)

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v   = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v
        
    def save(self):
        torch.save(self.state_dict(), self.ckpt_file)

    def load(self):
        self.load_state_dict(torch.load(self.ckpt_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256,
                 n_actions=2, name='actor', ckpt_dir='sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.ckpt_dir = ckpt_dir
        self.ckpt_file = os.path.join(ckpt_dir, self.name)
        self.max_action = max_action
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    
    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = F.softplus(sigma) + self.reparam_noise

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probs = Normal(mu, sigma)

        if reparameterize:
            actions = probs.rsample()
        else:
            actions = probs.sample()
        
        a = torch.tanh(actions)
        action = a * torch.as_tensor(self.max_action, device=self.device, dtype=actions.dtype)
        log_probs = probs.log_prob(actions) - torch.log(1 - a.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(dim=-1, keepdim=True)

        return action, log_probs


    def save(self):
            torch.save(self.state_dict(), self.ckpt_file)

    def load(self):
        self.load_state_dict(torch.load(self.ckpt_file))

class Agent():
    def __init__(self, alpha=1e-4, beta=1e-4, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1_000_000, tau=5e-3,
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        if env is None:
            raise ValueError("Agent requires an environment instance to infer action bounds.")

        max_action = np.asarray(env.action_space.high, dtype=np.float32)
        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                                  name='actor', max_action=max_action)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                     name='critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                                      name='critic_2')
        self.value = ValueNetwork(beta, input_dims, name='value')
        self.target_value = ValueNetwork(beta, input_dims, name='target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, obs):
        state = torch.as_tensor(obs, dtype=torch.float32, device=self.actor.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau) * target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)
    
    def save_models(self):
        self.actor.save()
        self.value.save()
        self.target_value.save()
        self.critic_1.save()
        self.critic_2.save()

    def load_models(self):
        self.actor.load()
        self.value.load()
        self.target_value.load()
        self.critic_1.load()
        self.critic_2.load()
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float, device=self.actor.device)
        done = torch.tensor(done, dtype=torch.bool, device=self.actor.device)
        state_ = torch.tensor(new_state, dtype=torch.float, device=self.actor.device)
        state = torch.tensor(state, dtype=torch.float, device=self.actor.device)
        action = torch.tensor(action, dtype=torch.float, device=self.actor.device)

        value = self.value(state).view(-1)
        with torch.no_grad():
            value_ = self.target_value(state_).view(-1)
        value_[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        self.value.optimizer.zero_grad()
        value_target = (critic_value - log_probs).detach()
        value_loss = 0.5 * F.mse_loss(value, value_target)
        value_loss.backward()
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state, actions)
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = torch.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        q_hat = self.scale*reward + self.gamma*value_.detach()
        q1_old_policy = self.critic_1.forward(state, action).view(-1)
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()


if __name__ == '__main__':
    env = gym.make('Pendulum-v1') # switch for humanoid
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_games = 250

    best_score = -np.inf
    score_history = []
    load_ckpt = False
    
    if load_ckpt:
        agent.load_models()
        env.render()
    
    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward
            agent.remember(obs, action, reward, obs_, done)
            if not load_ckpt:
                agent.learn()
            obs = obs_
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_ckpt:
                agent.save_models()
        
        print(f"episode: {i} score: {score}, avg_score: {avg_score}")

    if not load_ckpt:
        x = [i+1 for i in range(n_games)]
        #plotting