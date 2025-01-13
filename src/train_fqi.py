import argparse
import random
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Protocol
from gymnasium.wrappers import TimeLimit
from env_hiv_new import FastHIVPatient
from evaluate import evaluate_agent
from tqdm import tqdm

# --------------------
# Hyperparamètres par défaut
# --------------------
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

# --------------------
# Interfaces / Protocol
# --------------------
class Agent(Protocol):
    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        pass

    def save(self, path: str) -> None:
        pass

    def load(self) -> None:
        pass

# --------------------
# Replay Buffer FQI
# --------------------
class FQIReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0

    def append(self, s, a, r, s_, d):
        """ Stocke une transition dans le buffer. """
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self):
        """
        Retourne *toutes* les transitions stockées
        (pour FQI Offline Training, on réentraîne sur l'ensemble du buffer).
        """
        return list(map(np.array, zip(*self.data)))

    def __len__(self):
        return len(self.data)

# --------------------
# Agent FQI
# --------------------
class FQI(Agent):
    def __init__(self, args=None):
        # Récupère les arguments depuis l'ArgumentParser ou utilise des valeurs par défaut
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
        
        # Fixer la graine aléatoire pour reproductibilité
        seed = args.seed if args is not None else SEED
        random.seed(seed)
        np.random.seed(seed)

        # Initialize the regressor for Q-function approximation
        self.model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=args.n_estimators if args is not None else N_ESTIMATORS,
                max_depth=args.max_depth if args is not None else MAX_DEPTH,
                random_state=seed
            )
        )

        # Initialise le replay buffer
        self.buffer = FQIReplayBuffer(capacity=self.buffer_size)

    def collect_data(self, env, n_episodes=100):
        """
        Collecte des échantillons (state, action, reward, next_state, done) 
        en suivant une politique epsilon-greedy, puis stocke ces transitions 
        dans le buffer.
        """
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

                # Fusionner done avec trunc => TimeLimit = fin d'épisode
                done = done or trunc

                # Stocker la transition
                self.buffer.append(state, action, reward, next_state, done)

                if done:
                    break
                state = next_state

    def fit(self, env):
        # Collecte initiale de données
        self.collect_data(env=env, n_episodes=self.episodes_start)
        
        for it in range(self.iterations):
            # Décroissance de epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            # Récupère *toutes* les transitions
            states, actions, rewards, next_states, dones = self.buffer.sample()

            # Si ce n'est pas la première itération,
            # on prédit Q(s', a') pour pouvoir calculer max Q(s')
            if it > 0:
                next_Q_values = self.model.predict(next_states)
                max_next_Q = np.max(next_Q_values, axis=1)
            else:
                # Au tout premier appel, le modèle n'est pas encore entraîné
                max_next_Q = np.zeros(len(states))

            # Calcul de la cible
            targets = rewards + self.gamma * max_next_Q * (1 - dones)

            # Prépare X_train et y_train
            # Q_values aura la forme (N, env.action_space.n)
            nb_actions = env.action_space.n
            Q_values = np.zeros((len(states), nb_actions))
            y_train = Q_values.copy()
            X_train = np.array(states)

            # Pour chaque transition, on met à jour la Q-value d'action
            for i, action in enumerate(actions):
                y_train[i, action] = targets[i]

            # Entraînement du modèle de régression
            self.model.fit(X_train, y_train)

            # Évaluation sur nb_episodes_test
            val_score = evaluate_agent(
                self,
                env=TimeLimit(FastHIVPatient(domain_randomization=False),
                              max_episode_steps=self.max_episode_steps),
                nb_episode=self.nb_episodes_test
            )
            print(f"Iteration {it:3d}, "
                  f"epsilon {self.epsilon:6.2f}, "
                  f"memory size {len(self.buffer):5d}, "
                  f"Evaluation score  {val_score:2e}")

            # Récolte de nouvelles transitions
            self.collect_data(env=env, n_episodes=self.episodes_by_iter)

    def act(self, observation: np.ndarray, use_random: bool = False) -> int:
        """Renvoie l'action qui maximise Q(observation, a)."""
        if use_random and np.random.rand() < self.epsilon:
            return env.action_space.sample()
        Q_values = self.model.predict([observation])
        return np.argmax(Q_values)

    def save(self, path: str) -> None:
        """Sauvegarde du modèle sur disque."""
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """Chargement du modèle à partir du disque."""
        self.model = joblib.load(path)

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fitted Q-Iteration with Random Forests")
    parser.add_argument("--n_estimators", type=int, default=100, help="Nombre d'arbres du Random Forest")
    parser.add_argument("--max_depth", type=int, default=10, help="Profondeur maximale de chaque arbre")
    parser.add_argument("--gamma", type=float, default=0.99, help="Facteur de discount")
    parser.add_argument("--iterations", type=int, default=50, help="Nombre d'itérations FQI")
    parser.add_argument("--n_episodes", type=int, default=200, help="(Non utilisé ci-dessous) Nombre d'épisodes pour collecter des données")
    parser.add_argument("--max_steps", type=int, default=200, help="Nombre maximum de pas par épisode")
    parser.add_argument("--buffer_size", type=int, default=int(1e5), help="Taille du replay buffer")
    parser.add_argument("--save_path", type=str, default="fqi_model.pkl", help="Chemin pour sauvegarder le modèle")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    parser.add_argument("--n_episodes_test", type=int, default=5, help="Nombre d'épisodes pour l'évaluation")
    parser.add_argument("--episodes_start", type=int, default=100, help="Nombre d'épisodes initiaux pour la collecte de données")
    parser.add_argument("--episodes_by_iter", type=int, default=10, help="Nombre d'épisodes collectés à chaque itération")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Valeur initiale de l'epsilon pour epsilon-greedy")
    parser.add_argument("--epsilon_min", type=float, default=0.05, help="Valeur min de l'epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.99, help="Facteur de décroissance de l'epsilon")

    args = parser.parse_args()

    # Création de l'environnement
    env = TimeLimit(
        FastHIVPatient(domain_randomization=False),
        max_episode_steps=args.max_steps
    )
    
    # On peut fixer la graine sur l'environnement (si le wrapper/Env l'autorise)
    # env.action_space.seed(args.seed)
    # env.observation_space.seed(args.seed)

    # Instanciation de l'agent
    fqi_agent = FQI(args)

    # Entraînement
    fqi_agent.fit(env)

    # Sauvegarde du modèle
    fqi_agent.save(args.save_path)
    print(f"Modèle sauvegardé à l'emplacement : {args.save_path}")

    # Évaluation finale sur un épisode (à titre d'exemple)
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = fqi_agent.act(state)
        state, reward, done, trunc, _ = env.step(action)
        done = done or trunc
        total_reward += reward

    print(f"Récompense totale sur l'épisode final : {total_reward}")
