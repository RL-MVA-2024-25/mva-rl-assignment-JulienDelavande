import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import argparse
from typing import Protocol
from gymnasium.wrappers import TimeLimit
from env_hiv_new import FastHIVPatient
from evaluate import evaluate_agent
from tqdm import tqdm

GAMMA = 0.99
ITERATIONS = 50
NB_EPISODES_TEST = 5
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99
EPISODES_START = 100
EPISODES_BY_ITER = 10
BUFFER_SIZE = int(1e5)
MAX_STEPS = 200
N_ESTIMATORS = 100
MAX_DEPTH = 10
SEED = 42


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
    def __init__(self, args=None):
        self.gamma = args.gamma if args is not None else GAMMA
        self.iterations = args.iterations if args is not None else ITERATIONS
        self.nb_episodes_test = args.n_episodes_test if args is not None else NB_EPISODES_TEST
        self.epsilon = args.epsilon if args is not None else EPSILON 
        self.epsilon_min = args.epsilon_min if args is not None else EPSILON_MIN
        self.epsilon_decay = args.epsilon_decay if args is not None else EPSILON_DECAY
        self.episodes_start = args.episodes_start if args is not None else EPISODES_START
        self.episodes_by_iter = args.episodes_by_iter if args is not None else EPISODES_BY_ITER
        self.buffer_size = args.buffer_size if args is not None else BUFFER_SIZE
        self.max_episode_steps = args.max_steps if args is not None else MAX_STEPS
        

        # Initialize the regressor for Q-function approximation
        self.model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=args.n_estimators if args is not None else N_ESTIMATORS, 
                                  max_depth=args.max_depth if args is not None else MAX_DEPTH,
                                  random_state=args.seed if args is not None else SEED)
        )

        self.buffer = FQIReplayBuffer(capacity=self.buffer_size)

    def collect_data(self, env, n_episodes=100):
        for _ in tqdm(range(n_episodes)):
            state, _ = env.reset()
            for _ in range(self.max_episode_steps):
                if np.random.rand() < self.epsilon:
                    # Exploration : action aléatoire
                    action = env.action_space.sample()
                else:
                    # Exploitation : action selon la politique actuelle
                    Q_values = self.model.predict([state])
                    action = np.argmax(Q_values)

                next_state, reward, done, trunc, _ = env.step(action)
                self.buffer.append(state, action, reward, next_state, done)
                if done or trunc:
                    break
                state = next_state


    def fit(self, env):
        self.collect_data(env=env, n_episodes=self.episodes_start)
        
        states, actions, rewards, next_states, dones = self.buffer.sample()
        Q_values = np.zeros((len(states), env.action_space.n))

        for it in range(self.iterations):
            
            # Decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Sample the updated buffer
            states, actions, rewards, next_states, dones = self.buffer.sample()

            # Compute the target Q-values
            if it > 0:
                next_Q_values = self.model.predict(next_states)
                max_next_Q = np.max(next_Q_values, axis=1)
            else:
                max_next_Q = np.zeros(len(states))

            targets = rewards + self.gamma * max_next_Q * (1 - dones)

            # Train the model
            Q_values = np.zeros((len(states), env.action_space.n))
            y_train = Q_values.copy()
            X_train = np.array(states)
            for i, action in enumerate(actions):
                y_train[i, action] = targets[i]

            self.model.fit(X_train, y_train)
            Q_values = self.model.predict(X_train)
            
            # evaluate the model
            val_score = evaluate_agent(self, env=TimeLimit(FastHIVPatient(domain_randomization=False),
                                                                 max_episode_steps=self.max_episode_steps), nb_episode=self.nb_episodes_test)
            print("Iteration ", '{:3d}'.format(it),
                      ", epsilon ", '{:6.2f}'.format(self.epsilon),
                      ", memory size ", '{:5d}'.format(len(self.buffer)),
                      ", Evaluation score  ", '{:2e}'.format(val_score),
                      sep='')
            
            # Collect new data with epsilon-greedy policy
            self.collect_data(env=env, n_episodes=self.episodes_by_iter)


    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_episodes_test", type=int, default=5, help="Number of episodes for evaluation")
    parser.add_argument("--episodes_start", type=int, default=100, help="Number of episodes for initial data collection")
    parser.add_argument("--episodes_by_iter", type=int, default=10, help="Number of episodes per iteration")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon value")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Minimum epsilon value")
    parser.add_argument("--epsilon_decay", type=float, default=0.99, help="Epsilon decay factor")
    args = parser.parse_args()

    env = TimeLimit(FastHIVPatient(domain_randomization=False), max_episode_steps=args.max_steps)

    fqi_agent = FQI(args)

    # Train the FQI agent
    fqi_agent.fit(env)

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
