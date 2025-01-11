import numpy as np
import pickle
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from env_hiv_new import FastHIVPatient
from statistics import mean

class ProjectAgent:
    def __init__(self, state_bins=(10, 10, 10, 10, 10, 10), action_space=4, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_bins = state_bins
        self.action_space = action_space

        # Bornes des variables extraites de l'environnement
        self.lower = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Bornes inférieures
        self.upper = np.array([1e6, 5e4, 3200.0, 80.0, 2.5e5, 353200.0])  # Bornes supérieures

        # Préparer les seuils pour chaque dimension
        self.bins = [
            np.linspace(self.lower[i], self.upper[i], num=b + 1)[1:-1]
            for i, b in enumerate(self.state_bins)
        ]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table
        self.q_table = np.zeros((*self.state_bins, self.action_space))

    def discretize_state(self, state):
        """
        Discrétise un état continu en indices d'état discrets.
        """
        # Discrétiser chaque dimension en utilisant np.digitize
        discretized = tuple(np.digitize(state[i], self.bins[i]) for i in range(len(state)))
        #print(f"Discretized: {discretized}")
        return discretized


    def act(self, observation, use_random=False):
        discretized_state = self.discretize_state(observation)
        if use_random or np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        return np.argmax(self.q_table[discretized_state])  # Exploitation

    def update_q_table(self, state, action, reward, next_state, done):
        state_discrete = self.discretize_state(state)
        next_state_discrete = self.discretize_state(next_state)

        # Update Q-value using the Bellman equation
        target = reward
        if not done:
            target += self.discount_factor * np.max(self.q_table[next_state_discrete])
        self.q_table[state_discrete][action] += self.learning_rate * (
            target - self.q_table[state_discrete][action]
        )

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f)

    def load(self):
        with open("q_table.pkl", "rb") as f:
            self.q_table = pickle.load(f)

if __name__ == "__main__":
    env = TimeLimit(env=FastHIVPatient(domain_randomization=False), max_episode_steps=200)
    agent = ProjectAgent()

    episodes = 1000
    for episode in range(episodes):
        rewards = []
        print(f"Episode {episode + 1}/{episodes}")
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            #print(f"State: {state}")
            action = agent.act(state, use_random=True)
            #print(f"Action: {action}")
            next_state, reward, done, truncated, _ = env.step(action)
            #print(f"Reward: {reward}")
            #print(f"Next State: {next_state}")
            
            # Update Q-table
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            rewards.append(reward)

        print(f"total reward: {total_reward}")
        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Log progress
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Save the trained Q-table
    agent.save("q_table.pkl")
