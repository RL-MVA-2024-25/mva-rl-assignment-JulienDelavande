import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env_hiv_new import FastHIVPatient

class ProjectAgent:
    def __init__(self):
        self.model = None

    def train(self, env, total_timesteps=100000):
        # Initialize the PPO model
        self.model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3, gamma=0.99, n_steps=2048,
                         batch_size=64, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5)
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, observation, use_random=False):
        if self.model is None:
            raise ValueError("Model is not trained. Call train() before using act().")
        action, _states = self.model.predict(observation, deterministic=not use_random)
        return action

    def save(self, path):
        if self.model is not None:
            self.model.save(path)
        else:
            raise ValueError("Model is not trained. Cannot save an untrained model.")

    def load(self, path):
        self.model = PPO.load(path)

if __name__ == "__main__":
    # Define a function to create new instances of the environment
    def make_env():
        return TimeLimit(env=FastHIVPatient(domain_randomization=False), max_episode_steps=200)

    # Create a vectorized environment with multiple instances
    vec_env = make_vec_env(make_env, n_envs=4)  # Vectorized environment for PPO

    agent = ProjectAgent()

    # Train the agent
    agent.train(vec_env, total_timesteps=200000)

    # Save the trained model
    agent.save("ppo_hiv_agent")

    # Evaluate the agent
    total_episodes = 10
    total_rewards = []

    # Use a single environment for evaluation
    eval_env = TimeLimit(env=FastHIVPatient(domain_randomization=False), max_episode_steps=200)
    for episode in range(total_episodes):
        state, _ = eval_env.reset()
        done = False
        truncated = False
        episode_reward = 0

        while not done and not truncated:
            action = agent.act(state)
            state, reward, done, truncated, _ = eval_env.step(action)
            episode_reward += reward

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    print(f"Average Reward over {total_episodes} episodes: {np.mean(total_rewards)}")
