import time
import numpy as np
from collections import deque, namedtuple
from PPOAgent import PPOAgent
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

ENV_NAME = "HalfCheetah-v5"
SEED = 1
NUM_STEPS = 2048
NUM_EPOCHS = 10
MINI_BATCH_SIZE = 64
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_UPDATES = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_INTERVAL = 10
SAVE_PATH = "cheetah.pt"


def train():
    env = gym.make(ENV_NAME)
    env = gym.wrappers.ClipAction(env)
    agent = PPOAgent(env, LR, NUM_STEPS, NUM_EPOCHS, MINI_BATCH_SIZE, CLIP_EPS, VF_COEF, ENT_COEF, DEVICE)
    obs, _ = env.reset()
    ep_rewards = deque(maxlen=100)
    ep_reward = 0
    total_steps = 0
    ep_count = 0

    for update in range(1, TOTAL_UPDATES+1):
        for step in range(NUM_STEPS):
            action, logp, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = float(terminated or truncated)
            ep_reward += reward

            agent.memory.push(obs, action, logp, reward, done, value)
            obs = next_obs
            total_steps += 1

            if done:
                obs, _ = env.reset()
                ep_rewards.append(ep_reward)
                ep_reward = 0
                ep_count += 1

        # last value for GAE
        obs_v = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, _, last_value = agent.net(obs_v)
            last_value = last_value.item()

        agent.update(last_value,GAMMA, LAM)

        if update % PRINT_INTERVAL == 0:
            avg_reward = np.mean(ep_rewards) if ep_rewards else 0
            print(f"Update {update} | Steps {total_steps} | AvgReward {avg_reward:.2f} | Episodes {ep_count}")

    agent.save(SAVE_PATH)
    env.close()
    print("Training finished. Model saved to", SAVE_PATH)

if __name__ == "__main__":
    train()
