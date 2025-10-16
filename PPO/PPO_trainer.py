from PPO_agent import ActorCritic, RolloutBuffer
import gymnasium as gym
import torch.utils.tensorboard
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#-----GLOBAL CONSTANTS-----
MAX_STEPS = 5000
EPISODES = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(env: gym.Env,input_agent=None,lr=3e-4):
    n_actions = env.action_space.shape
    n_obs = env.observation_space.shape
    memory = RolloutBuffer(MAX_STEPS,n_obs,n_actions)
    agent = ActorCritic(n_obs,n_actions)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)


    if input_agent != None:
        agent.load_agent(input_agent)

    for episode in range(1, EPISODES+1, 1):
        done = False
        state,_ = env.reset()
        state = torch.from_numpy(state)
        for step in range(MAX_STEPS):

            with torch.no_grad:
                action, log_prob, entropy_prob, value = agent(state)
            next_state, reward, terminated, truncted,_ = env.step(action)
            done = terminated or truncted
            memory.push(state, action, reward, done, log_prob, value)
            if not done:
                state = torch.from_numpy(next_state)
            else:
                state,_ = env.reset()
                state = torch.from_numpy(state)
            
            print("hi")



def test(env: gym.Env, agent: nn.Module):
    done = False
    if agent == None:
        raise
    agent.eval()
    for _ in range(5):
        state,_ = env.reset()
        while not done:
            with torch.no_grad:
                action = agent(state)
            state, _, terminated, truncted,_ = env.step(action)
            done = terminated or truncted


if __name__ == "__main__":
    try:
        env = gym.make("Pendulum-v1",render=None)
        train(env)
    finally:
        env.close()
