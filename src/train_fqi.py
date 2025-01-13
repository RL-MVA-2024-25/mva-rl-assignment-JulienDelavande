import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import argparse
from typing import Protocol

class Agent(Protocol):
    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self) -> None:
        pass

class FQIReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self):
        return list(map(np.array, zip(*self.data)))

    def __len__(self):
        return len(self.data)

class FQI(Agent):
    def __init__(self, env, n_estimators=50, max_depth=5, gamma=0.99, iterations=50, buffer_size=int(1e5)):
        self.env = env
        self.gamma = gamma
        self.iterations = iterations

        # Initialize the regressor for Q-function approximation
        self.model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        )

        self.buffer = FQIReplayBuffer(capacity=buffer_size)

    def collect_data(self, n_episodes=100, max_steps=200):
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            for _ in range(max_steps):
                action = self.env.action_space.sample()
                next_state, reward, done, trunc, _ = self.env.step(action)
                self.buffer.append(state, action, reward, next_state, done)
                if done or trunc:
                    break
                state = next_state

    def fit(self):
        states, actions, rewards, next_states, dones = self.buffer.sample()

        # Initialize Q values
        Q_values = np.zeros((len(states), self.env.action_space.n))

        for it in range(self.iterations):
            print(f"Iteration {it + 1}/{self.iterations}")

            # Compute the target values
            if it > 0:
                next_Q_values = self.model.predict(next_states)
                max_next_Q = np.max(next_Q_values, axis=1)
            else:
                max_next_Q = np.zeros(len(states))

            targets = rewards + self.gamma * max_next_Q * (1 - dones)

            # Prepare training data for the model
            X_train = np.array(states)
            y_train = Q_values.copy()
            for i, action in enumerate(actions):
                y_train[i, action] = targets[i]

            # Train the model
            self.model.fit(X_train, y_train)

            # Update Q values for the next iteration
            Q_values = self.model.predict(X_train)

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        if use_random:
            return self.env.action_space.sample()
        Q_values = self.model.predict([observation])
        return np.argmax(Q_values)

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, path)

    def load(self) -> None:
        import joblib
        self.model = joblib.load("fqi_model.pkl")

if __name__ == "__main__":
    from gymnasium.wrappers import TimeLimit
    from env_hiv_new import FastHIVPatient

    parser = argparse.ArgumentParser(description="Fitted Q-Iteration with Random Forests")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the random forest")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum depth of each tree")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--iterations", type=int, default=50, help="Number of FQI iterations")
    parser.add_argument("--n_episodes", type=int, default=200, help="Number of episodes for data collection")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--buffer_size", type=int, default=int(1e5), help="Replay buffer size")
    parser.add_argument("--save_path", type=str, default="fqi_model.pkl", help="Path to save the model")
    args = parser.parse_args()

    env = TimeLimit(FastHIVPatient(domain_randomization=False), max_episode_steps=args.max_steps)

    fqi_agent = FQI(env, n_estimators=args.n_estimators, max_depth=args.max_depth, gamma=args.gamma, iterations=args.iterations, buffer_size=args.buffer_size)

    # Collect data by interacting with the environment
    fqi_agent.collect_data(n_episodes=args.n_episodes, max_steps=args.max_steps)

    # Train the FQI agent
    fqi_agent.fit()

    # Save the trained model
    fqi_agent.save(args.save_path)
    print(f"Model saved to {args.save_path}")

    # Evaluate the trained agent
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = fqi_agent.act(state)
        state, reward, done, trunc, _ = env.step(action)
        total_reward += reward

    print(f"Total reward after evaluation: {total_reward}")
