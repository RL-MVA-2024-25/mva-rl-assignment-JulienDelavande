import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from env_hiv_new import FastHIVPatient
from main import seed_everything
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class ProjectAgent:
    def __init__(self, state_dim, action_space, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy_net = DQN(state_dim, action_space).to(self.device)
        self.target_net = DQN(state_dim, action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.memory = []
        self.batch_size = 64
        self.memory_size = 10000

    def act(self, observation, use_random=False):
        if use_random and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_space)  # Exploration
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.policy_net(observation)).item()  # Exploitation

    def store_transition(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self):
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32).to(self.device),
            torch.tensor(actions, dtype=torch.long).to(self.device),
            torch.tensor(rewards, dtype=torch.float32).to(self.device),
            torch.tensor(next_states, dtype=torch.float32).to(self.device),
            torch.tensor(dones, dtype=torch.float32).to(self.device),
        )

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.sample_batch()

        # Compute Q-values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

if __name__ == "__main__":
    env = TimeLimit(env=FastHIVPatient(domain_randomization=False), max_episode_steps=200)
    state_dim = env.observation_space.shape[0]
    action_space = env.action_space.n

    agent = ProjectAgent(state_dim=state_dim, action_space=action_space)

    episodes = 1000
    target_update = 10
    total_rewards = []
    verbose_t = 20

    for episode in range(episodes):
        print(f"Episode {episode + 1}/{episodes}")
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            action = agent.act(state, use_random=True)
            next_state, reward, done, truncated, _ = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
        
        total_rewards.append(total_reward)
        print(f"Total Reward = {total_reward}")
        # Decay epsilon
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Update target network
        if (episode + 1) % target_update == 0:
            agent.update_target_network()

        # Log progress
        if (episode + 1) % verbose_t == 0:
            print(f"Episode {episode + 1}: Total Reward = {total_reward}")
            print(f"last {verbose_t} Mean Reward = {np.mean(total_rewards[-verbose_t:])}")
            
    plt.plot(total_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Curve")
    plt.show()

    # Save the trained model
    agent.save("dqn_model.pth")


    from evaluate import evaluate_HIV, evaluate_HIV_population
    # Evaluate agent
    state_dim = 6
    action_space = 4

    agent = ProjectAgent(state_dim=state_dim, action_space=action_space)
    agent.load("dqn_model.pth")
    score_agent = evaluate_HIV(agent=agent, nb_episode=5)
    print(f"Score agent: {score_agent}")
    score_agent_dr = evaluate_HIV_population(agent=agent, nb_episode=20)
    print(f"Score agent with domain randomization: {score_agent_dr}")
