from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_agent

from functools import partial

import random
import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
import os

try:
    from env_hiv_new import FastHIVPatient
except:
    pass

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPISODES = 200
DOMAIN_RANDOMIZATION = False
GAMMA = 0.99
BATCH_SIZE = 512
BUFFER_SIZE = 1e5
EPSILON_MAX = 1.
EPSILON_MIN = 0.01
EPSILON_DECAY_PERIOD = 1e4
EPSILON_DELAY_DECAY = 100
LEARNING_RATE = 1e-3
GRADIENT_STEPS = 3
UPDATE_TARGET_FREQ = 200
NEURONS = 256
MODEL_PATH = "dqn2.pt"

SCALE_REWARD = int(1e9)
SCALE_OBSERVATION = True

DOUBLE_DQN = False
FAST_ENV = False

OBSERVATION_SPACE = 6
ACTION_SPACE = 4

MAX_EPISODES_STEPS = 200
NB_EPSIODES_TEST = 1


'''
First, we need to define buffer allowing us
to sample data form the environment with an iid assumption.
We will store all the samples in a training set with a distribution
that is close to the distribution of the policy we are trying to learn, 
and independently sample from this training set to train the model.

We will use the ReplayBuffer class introduced in the notebook RL4 (FIFO mechanism).
'''

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
        self.size = 0
        self.obs_means = np.zeros(OBSERVATION_SPACE)
        self.obs_stds = np.ones(OBSERVATION_SPACE)
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.size = min(self.size + 1, self.capacity)
        self.index = (self.index + 1) % self.capacity
        
        # Update the running mean and std
        observations = np.array([item[0] for item in self.data[:self.size]])
        self.obs_means = np.mean(observations, axis=0)
        self.obs_stds = np.std(observations, axis=0)

        
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
    

'''
Our model will build an approximate value iteration function using neural networks.
For this, we implemented a very simple Deep Q-Network (DQN) model described in 'Playing Atari with Deep Reinforcement Learning' by Mnih et al. (2013).
The model takes epsilon-greedy actions, with a decreasing epsilon over time, stores the samples in a replay buffer, 
and at each interaction compute the target values of a drawn mini-batch to take a gradient step.

The model was selected through a hyperparameter search, starting from the base model given in the RL4 notebook
We increased epsilon decay period to 1000 to increase exploration at the beginning of the training, (wait 1000 steps before decay)
and did the same with the epsilon delay to 100 (20 was too fast). 

The best tested results we got with a simple neural network with 5 hidden layers, 
starting at 256 neurons and increasing by a factor of 2 at each layer, the last layers being of size 1024
(the number of neurons are chosen as multiples of 2 to allow for better parallelization on GPUs).
The batch sizes were increased to 512 to allow for better generalization (as the problem is more complicated than cartpole), 
and the number of gradient steps was increased to 3 to allow for better convergence
The training was done on 200 episodes, but the best models were obtained after 100 episodes in general.
'''

class ProjectAgent:
    
    def __init__(self, args=None):
        self.episodes = args.episodes if args is not None else EPISODES
        self.nb_episodes_test = args.nb_episodes_test if args is not None else NB_EPSIODES_TEST
        self.gamma = args.gamma if args is not None else GAMMA                                      # 0.99 default value     ## discount factor
        self.batch_size = args.batch_size if args is not None else BATCH_SIZE                       # 512 default value      ## batch size
        buffer_size = args.buffer_size if args is not None else BUFFER_SIZE                       # 100_000 default value   ## buffer size
        self.epsilon_max = args.epsilon_max if args is not None else EPSILON_MAX                   # 1. default value       ## epsilon greedy
        self.epsilon_min = args.epsilon_min if args is not None else EPSILON_MIN                 # 0.01 default value     ## epsilon greedy
        self.epsilon_stop = args.epsilon_decay_period if args is not None else EPSILON_DECAY_PERIOD        # 10_000 default value    ## epsilon decay period
        self.epsilon_delay = args.epsilon_delay_decay if args is not None else EPSILON_DELAY_DECAY        # 100 default value      ## epsilon delay decay
        lr = args.learning_rate if args is not None else LEARNING_RATE                             # 0.001 default value    ## learning rate
        self.nb_gradient_steps = args.gradient_steps if args is not None else GRADIENT_STEPS           # 3 default value        ## gradient steps
        self.update_target_freq = args.update_target_freq if args is not None else UPDATE_TARGET_FREQ    # 200 default value      ## update target NET frequency
        self.neurons = args.neurons if args is not None else NEURONS                          # 256 default value      ## number of neurons
        self.criterion = torch.nn.SmoothL1Loss()               # Loss function
        
        self.path = os.path.join(os.path.dirname(__file__), args.model if args is not None else MODEL_PATH)
        self.nb_actions =  ACTION_SPACE # env.action_space.n,                           # 4 
        self.observation_space =OBSERVATION_SPACE # env.observation_space.shape[0],        # 6
        
        
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.memory = ReplayBuffer(buffer_size, DEVICE)
        self.model = self.get_DQN()
        self.target_model = deepcopy(self.model).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) # Standard optimizer
        
        self.scale_reward = args.scale_reward if args is not None else SCALE_REWARD
        self.scale_observation = args.scale_observation if args is not None else SCALE_OBSERVATION
        self.obs_means = np.zeros(self.observation_space)
        self.obs_stds = np.ones(self.observation_space)
        
            
        
        
    def act(self, observation, use_random=False):
        observation = torch.Tensor(observation).unsqueeze(0).to(DEVICE)
        if self.scale_observation:
            observation = self._normalize_state(observation)
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        cpu_device = torch.device("cpu")
        DQN = self.get_DQN()
        self.model = DQN.to(cpu_device)
        self.model.load_state_dict(torch.load(self.path, map_location=cpu_device, weights_only=True))
        self.model.eval()
        return


    def get_DQN(self):
        # Defining an instance of the DQN model tested
        state_dim = self.observation_space
        n_action = self.nb_actions
        nb_neurons = self.neurons
        DQN = torch.nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons*2),
            nn.ReLU(),
            nn.Linear(nb_neurons*2, nb_neurons*4),
            nn.ReLU(),
            nn.Linear(nb_neurons*4, nb_neurons*8),
            nn.ReLU(),
            nn.Linear(nb_neurons*8, n_action)
        ).to(DEVICE)
        return DQN


    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            if self.scale_observation:
                X = self._normalize_state(X)
                Y = self._normalize_state(Y)
            if self.scale_reward:
                R = R/self.scale_reward
            # run through the target model
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def train(self, env):
        episode_return = []
        val_scores = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0

        # Keep the score on the chosen patient
        # to get the model with the best result
        score = 0
        while episode < self.episodes:
            
            # Epsilon greedy policy
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
                
            # Choose an action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = self.act(state)

            # Take a step in the environment
            next_state, reward, done, trunc, _ = env.step(action)
            
            if self.scale_reward:
                reward = reward/self.scale_reward
                
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Update the model
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update the target model
            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
                
            # Update the state
            if done or trunc:
                episode += 1
                val_score = evaluate_agent(self, env=TimeLimit(FastHIVPatient(domain_randomization=False) if args.fast else HIVPatient(domain_randomization=False),
                                                                 max_episode_steps=MAX_EPISODES_STEPS), nb_episode=self.nb_episodes_test)
                val_scores.append(val_score)
                print("Episode ", '{:3d}'.format(episode),
                      ", epsilon ", '{:6.2f}'.format(epsilon),
                      ", memory size ", '{:5d}'.format(len(self.memory)),
                      ", episode return ", '{:2e}'.format(episode_cum_reward),
                      ", Evaluation score  ", '{:2e}'.format(val_score),
                      sep='')
                state, _ = env.reset()
                
                # Save the model if the score is better
                if val_score > score:
                  score = val_score
                  self.best_model = deepcopy(self.model).to(DEVICE)
                  self.save(self.path)
                  
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state
                
            step += 1
        return episode_return, val_scores
    
    def _normalize_state(self, state):
        # Standardisation avec moyenne et écart-type (ajuster ces valeurs)
        # running mean and std
        state_mean = self.memory.obs_means
        state_std = self.memory.obs_stds
        state_mean = torch.Tensor(state_mean).to(DEVICE)
        state_std = torch.Tensor(state_std).to(DEVICE)
        return (state - state_mean) / state_std


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to save the model")
    parser.add_argument("--episodes", type=int, default=EPISODES, help="Number of episodes to train the model")
    parser.add_argument("--domain_randomization", type=bool, default=DOMAIN_RANDOMIZATION, help="Domain randomization for training")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for the model")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor for the model")
    parser.add_argument("--buffer_size", type=int, default=BUFFER_SIZE, help="Size of the replay buffer")
    parser.add_argument("--epsilon_min", type=float, default=EPSILON_MIN, help="Minimum epsilon value for epsilon greedy")
    parser.add_argument("--epsilon_max", type=float, default=EPSILON_MAX, help="Maximum epsilon value for epsilon greedy")
    parser.add_argument("--epsilon_decay_period", type=int, default=EPSILON_DECAY_PERIOD, help="Epsilon decay period (wait X steps before decay)")
    parser.add_argument("--epsilon_delay_decay", type=int, default=EPSILON_DELAY_DECAY, help="Epsilon delay decay (wait X steps before decay)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for the model")
    parser.add_argument("--gradient_steps", type=int, default=GRADIENT_STEPS, help="Number of gradient steps for the model for each episode")
    parser.add_argument("--double", type=bool, default=DOUBLE_DQN, help="Use Double DQN model")
    parser.add_argument("--update_target_freq", type=int, default=UPDATE_TARGET_FREQ, help="Update target network frequency")
    parser.add_argument("--fast", type=bool, default=FAST_ENV, help="Use fast environment")
    parser.add_argument("--neurons", type=int, default=NEURONS, help="Number of input neurons for the model")
    parser.add_argument("--nb_episodes_test", type=int, default=NB_EPSIODES_TEST, help="Number of episodes to test the model")
    parser.add_argument("--scale_reward", type=float, default=SCALE_REWARD, help="Scale the reward")
    parser.add_argument("--scale_observation", type=bool, default=SCALE_OBSERVATION, help="Scale the observation")
    args = parser.parse_args()
    
    
    agent = ProjectAgent(args)
    
    if args.fast:
        env = TimeLimit(
        env=FastHIVPatient(domain_randomization=args.domain_randomization), max_episode_steps=200
        )
    else:
        env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200
        )
        
    episode_return, val_scores = agent.train(env)
    print(episode_return)
    agent.save(args.model)
    print(f"Model saved at {args.model}")
    print(f'Best model score: {max(val_scores)}')

## python src/train_dqn_new.py --model dqn.pt --episodes {EPISODES} --domain_randomization False --learning_rate {LEARNING_RATE} --gamma {GAMMA} --buffer_size {BUFFER_SIZE} --epsilon_min {EPSILON_MIN} --epsilon_max {EPSILON_MAX} --epsilon_decay_period {EPSILON_DECAY_PERIOD} --epsilon_delay_decay {EPSILON_DELAY_DECAY} --batch_size {BATCH_SIZE} --gradient_steps {GRADIENT_STEPS} --double False --update_target_freq {UPDATE_TARGET_FREQ} --fast True --neurons {NEURONS}