import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from functools import partial

import math
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV, evaluate_agent

try:
    from env_hiv_new import FastHIVPatient
except ImportError:
    # Si jamais vous avez un fallback
    FastHIVPatient = HIVPatient

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== CONSTANTES / DEFAULTS =====================
EPISODES              = 200
DOMAIN_RANDOMIZATION  = False
GAMMA                 = 0.99
BATCH_SIZE            = 512
BUFFER_SIZE           = 1e5
EPSILON_MAX           = 1.0
EPSILON_MIN           = 0.01
EPSILON_DECAY_PERIOD  = 1e4
EPSILON_DELAY_DECAY   = 100
LEARNING_RATE         = 1e-3
GRADIENT_STEPS        = 3
UPDATE_TARGET_FREQ    = 200
NEURONS               = 256
MODEL_PATH            = "dqn_rainbow.pt"
MAX_EPISODES_STEPS    = 200
NB_EPSIODES_TEST      = 1

OBSERVATION_SPACE     = 6
ACTION_SPACE          = 4

# ===================== RAINBOW IMPLEMENTATION =====================
DOUBLE = True
DUELING = True
NOISY = True
PRIORITIZED = True
N_STEP = 3
DISTRIBUTIONAL = True
ATOMS = 51
V_MIN = 1e6
V_MAX = 1e11

# ===================== NOISY LINEAR LAYER =====================
class NoisyLinear(nn.Module):
    """
    Implémentation d'une couche linéaire "noisy" 
    (Fortunato et al. 2017)
    """
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Paramètres trainables
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # Tensors non-trainables
        self.register_buffer('weight_epsilon', torch.zeros(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.zeros(out_features))

        # Initialisation des poids
        self.reset_parameters(sigma_init)

        # Pour l'échantillonnage
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(sigma_init / math.sqrt(self.in_features))

    def reset_noise(self):
        # Facteur d'échantillonnage factorisé (Fortunato et al.)
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon = eps_out.ger(eps_in)
        self.bias_epsilon = eps_out

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # À l'inférence, on utilise la valeur moyenne
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def _scale_noise(self, size):
        # Échantillon gaussien
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

# ===================== REPLAY BUFFER =====================
class ReplayBuffer:
    """
    Replay buffer classique (FIFO). 
    Ne gère PAS les priorités !
    """
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.buffer = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Transpose de la liste de tuples
        s, a, r, s_, d = map(np.array, zip(*batch))
        return (torch.tensor(s,  device=self.device, dtype=torch.float32),
                torch.tensor(a,  device=self.device, dtype=torch.long),
                torch.tensor(r,  device=self.device, dtype=torch.float32),
                torch.tensor(s_, device=self.device, dtype=torch.float32),
                torch.tensor(d,  device=self.device, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)

# ===================== PRIORITIZED REPLAY BUFFER =====================
class PrioritizedReplayBuffer:
    """
    Implémentation d'un Prioritized Replay (Schaul et al. 2016).
    On utilise un segment-tree (somme) ou un Fenwick Tree selon la littérature.
    Pour simplifier ici, on peut le coder de façon plus brute, 
    sachant que pour un usage intensif, on optimiserait davantage.
    """
    def __init__(self, capacity, device, alpha=0.6, beta=0.4, beta_increment=1e-5, eps=1e-2):
        self.capacity = int(capacity)
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.position = 0

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps
        self.device = device

    def append(self, s, a, r, s_, d):
        # On met la priorité max pour la nouvelle transition
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((s, a, r, s_, d))
        else:
            self.buffer[self.position] = (s, a, r, s_, d)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            probs = self.priorities ** self.alpha
        else:
            probs = self.priorities[:len(self.buffer)] ** self.alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        samples = [self.buffer[idx] for idx in indices]
        s, a, r, s_, d = map(np.array, zip(*samples))

        # Poids d'importance
        total = len(self.buffer)
        weights = (total * probs[indices])**(-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        return (torch.tensor(s,  device=self.device, dtype=torch.float32),
                torch.tensor(a,  device=self.device, dtype=torch.long),
                torch.tensor(r,  device=self.device, dtype=torch.float32),
                torch.tensor(s_, device=self.device, dtype=torch.float32),
                torch.tensor(d,  device=self.device, dtype=torch.float32),
                torch.tensor(indices, device=self.device, dtype=torch.long),
                torch.tensor(weights, device=self.device, dtype=torch.float32))

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + self.eps

    def __len__(self):
        return len(self.buffer)

# ===================== MODEL & DUELING / DISTRIBUTIONAL =====================
class DQNNet(nn.Module):
    """
    DQN classique : 
      - couches linéaires fully connected 
      - pas de distributional, ni dueling
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, noisy=False):
        super(DQNNet, self).__init__()
        # Selon le boolean noisy, on choisit entre Linear et NoisyLinear
        Linear = NoisyLinear if noisy else nn.Linear

        self.net = nn.Sequential(
            Linear(state_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            Linear(hidden_dim*2, hidden_dim*4),
            nn.ReLU(),
            Linear(hidden_dim*4, hidden_dim*8),
            nn.ReLU(),
            Linear(hidden_dim*8, action_dim)
        )

    def forward(self, x):
        return self.net(x)

    def reset_noise(self):
        """
        Pour les couches NoisyLinear, on a besoin de reset 
        l'échantillon de bruit à chaque forward/backward epoch. 
        Sinon, on ne fait rien.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DuelingDQNNet(nn.Module):
    """
    Dueling DQN : 
      - on sépare la branche "valeur" et "avantage"
      - on combine ensuite V + A - mean(A)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, noisy=False):
        super(DuelingDQNNet, self).__init__()
        Linear = NoisyLinear if noisy else nn.Linear

        # Feature extractor
        self.feature = nn.Sequential(
            Linear(state_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Branche Valeur
        self.value_stream = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, 1)
        )

        # Branche Avantage
        self.adv_stream = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        value = self.value_stream(x)
        advantage = self.adv_stream(x)
        # Combine
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class C51Network(nn.Module):
    """
    Implémentation d'un réseau Distributional C51 
    (avec ou sans Dueling, et/ou Noisy).
    On prédit la distribution Q par atomes. 
    => output shape = (batch_size, action_dim * atoms)
    On la reshape en (batch_size, action_dim, atoms)
    """
    def __init__(self, state_dim, action_dim, atoms=51, hidden_dim=256, 
                 dueling=False, noisy=False):
        super(C51Network, self).__init__()
        self.action_dim = action_dim
        self.atoms = atoms
        Linear = NoisyLinear if noisy else nn.Linear

        if dueling:
            # Dueling version
            self.feature = nn.Sequential(
                Linear(state_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            # Valeur
            self.value_stream = nn.Sequential(
                Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, atoms)
            )
            # Avantage
            self.adv_stream = nn.Sequential(
                Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, action_dim * atoms)
            )
        else:
            # Non dueling
            self.net = nn.Sequential(
                Linear(state_dim, hidden_dim),
                nn.ReLU(),
                Linear(hidden_dim, hidden_dim*2),
                nn.ReLU(),
                Linear(hidden_dim*2, hidden_dim*4),
                nn.ReLU(),
                Linear(hidden_dim*4, action_dim * atoms)
            )
            self.dueling = False

        self.dueling = dueling

    def forward(self, x):
        if self.dueling:
            x = self.feature(x)
            value = self.value_stream(x)                 # shape [B, atoms]
            advantage = self.adv_stream(x)               # shape [B, actions*atoms]
            advantage = advantage.view(-1, self.action_dim, self.atoms)
            # On ajoute la dimension atoms pour la value
            value = value.view(-1, 1, self.atoms)
            # broadcast sur l'action
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            logits = self.net(x)  # shape [B, action_dim * atoms]
            q_atoms = logits.view(-1, self.action_dim, self.atoms)

        # Appliquer softmax sur la dimension des atomes (distribution)
        probs = F.softmax(q_atoms, dim=2)
        return probs

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# ===================== PROJECT AGENT (Rainbow, modular) =====================
class ProjectAgent:
    def __init__(self, args):
        # ----------------------- HYPERPARAMS -----------------------
        self.episodes           = args.episodes
        self.nb_episodes_test   = args.nb_episodes_test
        self.gamma              = args.gamma
        self.batch_size         = args.batch_size
        self.epsilon_max        = args.epsilon_max
        self.epsilon_min        = args.epsilon_min
        self.epsilon_stop       = args.epsilon_decay_period
        self.epsilon_delay      = args.epsilon_delay_decay
        self.nb_gradient_steps  = args.gradient_steps
        self.update_target_freq = args.update_target_freq
        self.neurons            = args.neurons
        self.double_dqn         = args.double
        self.dueling            = args.dueling
        self.noisy              = args.noisy
        self.prioritized        = args.prioritized
        self.n_step             = args.n_step
        self.distributional     = args.distributional
        self.atoms              = args.atoms
        self.v_min              = args.v_min
        self.v_max              = args.v_max

        lr          = args.learning_rate
        buffer_size = args.buffer_size
        self.path   = os.path.join(os.path.dirname(__file__), args.model)

        # ----------------------- ENV SPACES -----------------------
        self.nb_actions = ACTION_SPACE
        self.observation_space = OBSERVATION_SPACE

        # -------------- SETUP REPLAY BUFFER --------------
        if self.prioritized:
            self.memory = PrioritizedReplayBuffer(buffer_size, DEVICE, 
                                                  alpha=0.6, beta=0.4, 
                                                  beta_increment=1e-5, eps=1e-2)
        else:
            self.memory = ReplayBuffer(buffer_size, DEVICE)

        # Multi-step small buffer pour accumuler avant ajout final
        self.n_step_buffer = []
        self.n_step_size   = self.n_step

        # -------------- SETUP MODEL --------------
        if self.distributional:
            # Distributional = C51
            self.model = C51Network(
                self.observation_space, 
                self.nb_actions, 
                atoms=self.atoms, 
                hidden_dim=self.neurons, 
                dueling=self.dueling, 
                noisy=self.noisy
            ).to(DEVICE)

            self.target_model = C51Network(
                self.observation_space, 
                self.nb_actions, 
                atoms=self.atoms, 
                hidden_dim=self.neurons, 
                dueling=self.dueling, 
                noisy=self.noisy
            ).to(DEVICE)
        else:
            # Q-values classiques
            if self.dueling:
                self.model = DuelingDQNNet(
                    self.observation_space, 
                    self.nb_actions, 
                    hidden_dim=self.neurons, 
                    noisy=self.noisy
                ).to(DEVICE)
                self.target_model = DuelingDQNNet(
                    self.observation_space, 
                    self.nb_actions, 
                    hidden_dim=self.neurons, 
                    noisy=self.noisy
                ).to(DEVICE)
            else:
                self.model = DQNNet(
                    self.observation_space, 
                    self.nb_actions, 
                    hidden_dim=self.neurons, 
                    noisy=self.noisy
                ).to(DEVICE)
                self.target_model = DQNNet(
                    self.observation_space, 
                    self.nb_actions, 
                    hidden_dim=self.neurons, 
                    noisy=self.noisy
                ).to(DEVICE)

        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # -------------- EPSILON INIT --------------
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/max(1, self.epsilon_stop)

        # -------------- C51 params --------------
        if self.distributional:
            # Crée la grille de soutien (support) z
            self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(DEVICE)
            self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)

        self.criterion = nn.SmoothL1Loss(reduction="none" if self.prioritized else "mean")

    def reset_noise(self):
        # Réinitialise le bruit pour les NoisyLayers
        if self.noisy:
            self.model.reset_noise()
            self.target_model.reset_noise()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self):
        cpu_device = torch.device("cpu")
        # Charger sur CPU pour éviter les soucis de compatibilité
        self.model.load_state_dict(torch.load(self.path, map_location=cpu_device))
        self.model.eval()

    def _append_transition(self, transition):
        """
        Ajoute la transition (s, a, r, s_, d) au buffer.
        Si n_step > 1, on accumule d'abord dans n_step_buffer.
        """
        if self.n_step_size == 1:
            # Pas de N-step
            if self.prioritized:
                self.memory.append(*transition)
            else:
                self.memory.append(*transition)
        else:
            # On stocke en buffer interne
            self.n_step_buffer.append(transition)
            if len(self.n_step_buffer) < self.n_step_size:
                return
            # On a accumulé n transitions
            R = 0
            for idx, (s, a, r, s_, d) in enumerate(self.n_step_buffer):
                R += (self.gamma**idx) * r
            s0, a0, _, _, _ = self.n_step_buffer[0]
            _, _, _, sN, dN = self.n_step_buffer[-1]
            # On place la transition agrégée
            self.memory.append(s0, a0, R, sN, dN)
            self.n_step_buffer.pop(0)

    def _projection_distribution(self, next_dist, rewards, dones):
        """
        Projection distributionnelle pour C51 :
        - next_dist: shape [B, actions, atoms]
        - rewards : shape [B]
        - dones   : shape [B]
        - On calcule la projection de la distribution Tz 
          (cf. eqn. 7 de l'article C51) 
        """
        batch_size = rewards.size(0)
        # next_dist est la distribution sur le support self.support
        # On va mapper sur Tz = r + gamma * support
        # support shape = [atoms]
        # next_z shape   = [B, atoms]
        next_z = self.support.unsqueeze(0).expand(batch_size, self.atoms)
        # Tz
        Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma * next_z)
        Tz = Tz.clamp(self.v_min, self.v_max)
        # Projection b
        b = (Tz - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Initialiser la distribution projetée
        proj_dist = torch.zeros_like(next_dist, device=DEVICE)
        for i in range(self.atoms):
            # On ajoute la masse de l'atome i
            # pour chaque batch item
            offset = torch.arange(0, batch_size, device=DEVICE) * self.atoms
            offset = offset.unsqueeze(1).expand(batch_size, self.atoms)
            l_index = (l + offset).view(-1)
            u_index = (u + offset).view(-1)

            proj_dist.view(-1).index_add_(
                0, l_index, (next_dist.view(-1)[:, i] * (u.float()-b).view(-1))
            )
            proj_dist.view(-1).index_add_(
                0, u_index, (next_dist.view(-1)[:, i] * (b-l.float()).view(-1))
            )
        return proj_dist

    def act(self, state, epsilon=0.):
        """
        Sélection d'action : epsilon-greedy ou argmax(Q).
        Si noisy, on ignore epsilon (souvent on peut le conserver, 
        mais la paper Rainbow le met à 0).
        """
        if self.noisy:
            # On sample l'action en se basant sur Q (pas besoin d’epsilon).
            epsilon = 0.0

        if random.random() < epsilon:
            return random.randint(0, self.nb_actions-1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                if self.distributional:
                    dist = self.model(state_t)  # shape [1, action_dim, atoms]
                    # On calcule E[Q] = somme a-> atoms( support_i * prob_i )
                    dist_mean = torch.sum(dist * self.support.view(1, 1, -1), dim=2)
                    return torch.argmax(dist_mean, dim=1).item()
                else:
                    q_values = self.model(state_t)
                    return torch.argmax(q_values, dim=1).item()

    def _compute_loss(self, batch):
        """
        Calcule la loss. Pour 
          - Distributional : KL entre distribution projetée et distribution courante 
          - Q-value        : Bellman / Huber
        """
        if self.prioritized:
            # Prioritized => batch = (s, a, r, s_, d, indices, weights)
            s, a, r, s_, d, indices, weights = batch
        else:
            s, a, r, s_, d = batch
            weights = None
            indices = None

        if self.distributional:
            # ---------- Distributional Loss ----------
            # dist(s,a) => on récupère la distribution du Q pour s
            dist_pred = self.model(s)  # [B, action_dim, atoms]
            dist_pred = dist_pred.gather(1, a.view(-1,1,1).expand(-1,1,self.atoms)).squeeze(1)
            # next_dist => distribution de la target
            with torch.no_grad():
                dist_next = self.target_model(s_)
                if self.double_dqn:
                    # Double => on choisit l'action via le model
                    dist_next_online = self.model(s_)
                    dist_mean_online = torch.sum(dist_next_online * self.support.view(1, 1, -1), dim=2)
                    best_actions = dist_mean_online.argmax(dim=1).view(-1,1,1)
                    dist_next = dist_next.gather(1, best_actions.expand(-1,1,self.atoms)).squeeze(1)
                else:
                    # max Q sur la target
                    dist_mean_target = torch.sum(dist_next * self.support.view(1, 1, -1), dim=2)
                    best_actions = dist_mean_target.argmax(dim=1).view(-1,1,1)
                    dist_next = dist_next.gather(1, best_actions.expand(-1,1,self.atoms)).squeeze(1)

            proj_dist = self._projection_distribution(dist_next, r, d)
            # On calcule la cross-entropy (KL)
            dist_pred = torch.clamp(dist_pred, 1e-5, 1.0)
            loss_elementwise = -(proj_dist * torch.log(dist_pred)).sum(dim=1)
        else:
            # ---------- Q-Learning Loss ----------
            Q = self.model(s)
            Q_acted = Q.gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if self.double_dqn:
                    # Double => action choisi par le model, Q évalué par la target
                    next_actions = self.model(s_).argmax(dim=1, keepdim=True)
                    Q_next_target = self.target_model(s_).gather(1, next_actions).squeeze(1)
                else:
                    Q_next_target = self.target_model(s_).max(dim=1)[0]
                target = r + (1 - d) * self.gamma * Q_next_target
            loss_elementwise = F.smooth_l1_loss(Q_acted, target, reduction='none')

        if self.prioritized:
            # On pèse par weights
            loss = (loss_elementwise * weights).mean()
            # On met à jour les priorités
            new_priorities = loss_elementwise.detach().cpu().numpy()
            self.memory.update_priorities(indices.cpu().numpy(), new_priorities)
        else:
            loss = loss_elementwise.mean()

        return loss

    def gradient_step(self):
        """
        Exécute une itération d'apprentissage (on sample un batch et on backprop).
        """
        if len(self.memory) < self.batch_size:
            return

        if self.prioritized:
            batch = self.memory.sample(self.batch_size)
        else:
            batch = self.memory.sample(self.batch_size)

        self.reset_noise()
        loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        epsilon = self.epsilon_max
        step = 0
        best_score = -1e9

        state, _ = env.reset()

        while episode < self.episodes:
            # Epsilon decay
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            action = self.act(state, epsilon)
            next_state, reward, done, trunc, _ = env.step(action)
            episode_cum_reward += reward

            # Stockage / replay buffer
            self._append_transition((state, action, reward, next_state, float(done)))

            # On exécute plusieurs gradient steps
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update target
            if step % self.update_target_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            if done or trunc:
                episode += 1
                # Reset n-step buffer
                self.n_step_buffer.clear()

                # Évaluation
                test_env = TimeLimit(FastHIVPatient(domain_randomization=False) 
                                     if args.fast else HIVPatient(domain_randomization=False), 
                                     max_episode_steps=MAX_EPISODES_STEPS)
                val_score = evaluate_agent(self, env=test_env, nb_episode=self.nb_episodes_test)

                print("Episode", episode, 
                      "Epsilon", round(epsilon,3), 
                      "Memory size", len(self.memory), 
                      "Episode return", f"{episode_cum_reward:.2e}",
                      "Evaluation score", f"{val_score:.2e}")

                # Sauvegarde du meilleur modèle
                if val_score > best_score:
                    best_score = val_score
                    self.save(self.path)

                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
                state, _ = env.reset()
            else:
                state = next_state

            step += 1

        return episode_return

# ===================== MAIN =====================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to save the model")
    # Expérimentation
    parser.add_argument("--episodes", type=int, default=EPISODES, help="Number of episodes to train")
    parser.add_argument("--nb_episodes_test", type=int, default=NB_EPSIODES_TEST, help="Number of episodes to test")
    parser.add_argument("--domain_randomization", action="store_true", default=DOMAIN_RANDOMIZATION)
    parser.add_argument("--fast", action="store_true", default=False)
    # Hyperparams
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--buffer_size", type=float, default=BUFFER_SIZE)
    parser.add_argument("--epsilon_min", type=float, default=EPSILON_MIN)
    parser.add_argument("--epsilon_max", type=float, default=EPSILON_MAX)
    parser.add_argument("--epsilon_decay_period", type=float, default=EPSILON_DECAY_PERIOD)
    parser.add_argument("--epsilon_delay_decay", type=float, default=EPSILON_DELAY_DECAY)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gradient_steps", type=int, default=GRADIENT_STEPS)
    parser.add_argument("--update_target_freq", type=int, default=UPDATE_TARGET_FREQ)
    parser.add_argument("--neurons", type=int, default=NEURONS)

    # Rainbow toggles
    parser.add_argument("--double", type=bool, default=DOUBLE, help="Use Double DQN")
    parser.add_argument("--dueling", type=bool, default=DUELING, help="Use Dueling DQN")
    parser.add_argument("--noisy", type=bool, default=NOISY, help="Use Noisy Networks")
    parser.add_argument("--prioritized", type=bool, default=PRIORITIZED, help="Use Prioritized Replay")
    parser.add_argument("--n_step", type=int, default=N_STEP, help="N-step returns")
    parser.add_argument("--distributional", type=bool, default=DISTRIBUTIONAL, help="Use Distributional RL (C51)")
    parser.add_argument("--atoms", type=int, default=ATOMS, help="Number of atoms for C51")
    parser.add_argument("--v_min", type=float, default=V_MIN, help="Min value for distributional support")
    parser.add_argument("--v_max", type=float, default=V_MAX, help="Max value for distributional support")

    args = parser.parse_args()

    agent = ProjectAgent(args)

    if args.fast:
        env = TimeLimit(
            env=FastHIVPatient(domain_randomization=args.domain_randomization), 
            max_episode_steps=MAX_EPISODES_STEPS
        )
    else:
        env = TimeLimit(
            env=HIVPatient(domain_randomization=args.domain_randomization), 
            max_episode_steps=MAX_EPISODES_STEPS
        )

    returns = agent.train(env)
    print(returns)
    agent.save(args.model)
    print(f"Model saved at {args.model}")

# ### !python src/train.py --model {MODEL_PATH} --episodes {EPISODES} --domain_randomization {DOMAIN_RANDOMIZATION} --learning_rate {LEARNING_RATE} --gamma {GAMMA} \
#     --buffer_size {BUFFER_SIZE} --epsilon_min {EPSILON_MIN} --epsilon_max {EPSILON_MAX} --epsilon_decay_period {EPSILON_DECAY_PERIOD} --epsilon_delay_decay {EPSILON_DELAY_DECAY} \
#     --batch_size {BATCH_SIZE} --gradient_steps {GRADIENT_STEPS} --double {DOUBLE_DQN} --update_target_freq {UPDATE_TARGET_FREQ} --fast {FAST_ENV} --neurons {NEURONS} --nb_episodes_test {NB_EPSIODES_TEST} \
#     --dueling {DUELING} --noisy {NOISY} --prioritized {PRIORITIZED} --n_step {N_STEP} --distributional {DISTRIBUTIONAL} --atoms {ATOMS} --v_min {V_MIN} --v_max {V_MAX}
