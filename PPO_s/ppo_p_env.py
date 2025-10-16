import gymnasium as gym
env = gym.make("Pendulum-v1", render_mode="rgb_array", g=9.81)

env.reset(seed=42, options={"low": -0.7, "high": 0.5})

for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"action={action}, reward={reward}")
    print(f"observation={observation}")
    if terminated or truncated:
        observation, info = env.reset(options={"low": -0.7, "high": 0.5})

state, _ = env.reset()
print("Observation space:", env.observation_space.shape)
print("Action space:", env.action_space.shape)
print(state)
env.close()

env.close()