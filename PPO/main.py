import time
import numpy as np
from collections import deque, namedtuple
from PPOAgent import PPOAgent
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

ENV_NAME = "Walker2d-v5"
NUM_STEPS = 2048
NUM_EPOCHS = 10
MINI_BATCH_SIZE = 64
GAMMA = 0.99
LAM = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.01
VF_COEF = 0.5
CLIP_COEF = 0.2
MAX_GRAD_NORM = 0.5
TOTAL_UPDATES = 2000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_INTERVAL = 1
SAVE_PATH = "Walker2d_Norm.pt"
RENDER_MODE = None
RUN_NAME = "walker2d-restart-14"

def train(load):
    try:
        writer = SummaryWriter(f"runs/{RUN_NAME}")
        writer.add_text(
            "hyperparameters",
            f"Max steps: {NUM_STEPS},  Epochs: {NUM_EPOCHS}, mini batch size: {MINI_BATCH_SIZE}, Gamaa: {GAMMA}, LAM: {LAM}, Eplison Clipping: {CLIP_EPS}"+
            f"Learning rate: {LR}, Entrpoy Coef: {ENT_COEF}, Value Function Coef: {VF_COEF}, Max Grad Norm: {MAX_GRAD_NORM}, Environment: {ENV_NAME}",
        )
        env = gym.make(ENV_NAME,render_mode=None)
        env = gym.wrappers.ClipAction(env)
        #env = gym.wrappers.NormalizeObservation(env)
        #env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
        #env = gym.wrappers.NormalizeReward(env)
        #env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -100, 100))
        agent = PPOAgent(env, LR, NUM_STEPS, NUM_EPOCHS, MINI_BATCH_SIZE, CLIP_EPS, VF_COEF, ENT_COEF,CLIP_COEF,MAX_GRAD_NORM, DEVICE)
        if load == True:
            agent.load("Walker2d_restart2.pt")
        obs, _ = env.reset()
        ep_rewards = deque(maxlen=100)
        ep_reward = 0
        total_steps = 0
        ep_count = 0
        ep_length = 0
        for update in range(1, TOTAL_UPDATES+1):
            scale = 1.0 - (update - 1.0) / TOTAL_UPDATES
            newlr = scale * LR
            agent.optimizer.param_groups[0]["lr"] = newlr
            for step in range(NUM_STEPS):
                if RENDER_MODE == "human":
                    env.render()
                action, logp, value = agent.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = float(terminated or truncated)
                ep_reward += reward
                ep_length += 1
                agent.memory.push(obs, action, logp, reward, done, value)
                obs = next_obs
                total_steps += 1
                if done:
                    writer.add_scalar("charts/episodic_return", ep_reward, total_steps)
                    writer.add_scalar("charts/episodic_length", ep_length, total_steps)
                    obs, _ = env.reset()
                    ep_rewards.append(ep_reward)
                    ep_reward = 0
                    ep_length = 0
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

    except KeyboardInterrupt:
        agent.save(SAVE_PATH)
        writer.close()
        env.close()
        print("Training finished. Model saved to", SAVE_PATH)

    finally:
        writer.close()
        agent.save(SAVE_PATH)
        env.close()
        print("Training finished. Model saved to", SAVE_PATH)

if __name__ == "__main__":
    train(False)
