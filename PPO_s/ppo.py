import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions.categorical import Categorical
import gymnasium as gym


class ActorCritic(nn.Module):
    #actor-critic module is a nice way to compute both actor(policy) and critic (estimated value function) at the same time
    def __init__(self,
                 obs_space=3,
                 action_space=1,
                 num_steps=128,
                 num_envs=1,
                 annealing=False,
                 lr=2.5e-4,
                 gae=False):

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.action_space = action_space
        self.obs_space = obs_space
        self.annealing = annealing
        self.lr = lr
        self.global_step_counter = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.r_buffer = RolloutBuffer(obs_space=self.obs_space,
                            action_space=self.action_space,
                            num_steps=self.num_steps,
                            num_envs=self.num_envs)

        self.actor = nn.Sequential(
            self.init_layer(nn.Linear(self.obs_space, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(self.obs_space, self.action_space), std=0.01)
        )
        self.critic = nn.Sequential(
            self.init_layer(nn.Linear(self.obs_space, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(self.obs_space, 1), std=1.)
        )

    def init_layer(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer
    
    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

    #def forward(self, x):
        #h in this case is image features from a game, do we really need that for the pendulum?
    #    h = self.head(x)
    #    return self.actor(h), self.critic(h)

    def config_optimizer(self):
        #default settings for now
        return torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-5)

    def _run_steps(self, env=None):
        #run the amount of steps that we need to fill the buffer
        if not env:
            print("No environment, not running steps.")
            return
        
        obs, actions, logprobs, rewards, dones, values = None
        for step in range(0, self.num_steps):
            self.global_step_counter += 1*self.num_envs

            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            #exec game and log data
            next_obs, reward, done, info = env.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(done).to(self.device)

            #add to buffer
            self.r_buffer.push(obs=next_obs,
                        action=action,
                        logprob=logprob,
                        reward=reward,
                        done=done,
                        value=value,
                        step=step)

            #lets view this print
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={self.global_step_counter}, episodic_return={item['episode']['r']}, episodic_length:{item["episode"]["l"]}")
                    break
        
        # TODO: ADD bootstrapping here?
        # bootstrap value if not done, if  we are not done, we need to bootstrap, dont really understand this nor gae
        # should we bootstrap before push to the RolloutBuffer? I think no, maybe just get batch in training, then bootstrap?
        # but we bootstrap on timestep so probably need to do it in _run_steps method?


    #maybe make this into a storage class or move to agent(actor_critic init and store there)
    #num_steps = 128, num_envs = 1 for now, lets keep it simple
    def train(self, env, total_timesteps):
        num_updates = 0
        
        #start of game or environment or whatever when we start the training
        #reset env here? or how to pass first env step?? we can probably pass the whole env into this instead
        next_obs = torch.Tensor(env.reset()).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        batch_size = self.num_steps * self.num_envs
        num_updates = total_timesteps // batch_size

        for update in range(1, num_updates + 1):

            #lr cooldown
            if self.annealing:
                frac = 1.0 - (update - 1.0) / num_updates
                curr_lr = frac * self.lr

            # _run_steps or collect batch samples or whatever we should call this
            self._run_steps(env=env)
            #looking at guide, they do bootstrapping here, not sure what that is?

            #get batch and flatten it
            b_obs, b_actions, b_logprobs, b_rewards, b_dones, b_values = self.r_buffer.get_batch()
            # ADD FLATTENING OF THESE VARIABLES :^)

            

                

                


class RolloutBuffer(nn.Module):
    def __init__(self, obs_space=3, action_space=1, num_steps=128, num_envs=1):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_steps = num_steps
        self.num_envs = num_envs
        #self.global_step = 0

        #init the rollout buffer
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.obs_space).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.action_space).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

    def push(self, obs, action, logprob, reward, done, value, step):
        #we are basically just overwriting values here
        self.obs[step] = obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = done
        self.values[step] = value

    def get_batch(self):
        return self.obs, self.actions, self.logprobs, self.rewards, self.dones, self.values

