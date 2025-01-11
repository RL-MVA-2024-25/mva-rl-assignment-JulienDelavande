import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
import json

import torch as th
from imitation.algorithms import bc
from imitation.data import serialize
from imitation.data.rollout import (
    TrajectoryAccumulator,
    types,
    GenTrajTerminationFn,
    make_sample_until,
    spaces,
    rollout_stats,
    unwrap_traj,
    dataclasses,
)
from imitation.util.logger import configure as configure_logger
from imitation.util.util import save_policy
from imitation.data import rollout
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
from gymnasium.wrappers import TimeLimit, TransformObservation
from coolname import generate_slug
from tqdm.rich import trange, tqdm

DISABLE_WANDB = False
try:
    import wandb
except ImportError:
    DISABLE_WANDB = True

from env_hiv_new import FastHIVPatient


@dataclass
class DoOnce:
    do: bool = True

    def __call__(self):
        if self.do:
            self.do = False
            return True
        return False


def build_env(domain_randomization: bool):
    env = FastHIVPatient(domain_randomization=domain_randomization)
    env = TransformObservation(
        env,
        lambda obs: np.log(np.maximum(obs, 1e-8)),
        env.observation_space,
    )
    env = TimeLimit(env, max_episode_steps=200)
    return env


def generate_heuristic_rollouts(
    add_fixed_env: bool,
    n_envs: int,
    num_rollouts: int,
    rng: np.random.Generator,
) -> types.TrajectoryWithRew:
    all_trajectories = []

    for _ in trange(0, num_rollouts, n_envs, desc="Generating rollouts"):
        venv = make_vec_env(
            lambda *, _do_once: build_env(_do_once()),
            n_envs=n_envs,
            vec_env_cls=SubprocVecEnv,
            env_kwargs=dict(_do_once=DoOnce(add_fixed_env)),
        )
        sample_until = make_sample_until(min_episodes=n_envs)
        # Collect rollout tuples.
        # accumulator for incomplete trajectories
        trajectories_accum = TrajectoryAccumulator()
        trajectories = []
        obs = venv.reset()
        assert isinstance(
            obs,
            (np.ndarray, dict),
        ), "Tuple observations are not supported."
        wrapped_obs = types.maybe_wrap_in_dictobs(obs)

        # we use dictobs to iterate over the envs in a vecenv
        for env_idx, ob in enumerate(wrapped_obs):
            # Seed with first obs only. Inside loop, we'll only add second obs from
            # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
            # get all observations, but they're not duplicated into "next obs" and
            # "previous obs" (this matters for, e.g., Atari, where observations are
            # really big).
            trajectories_accum.add_step(dict(obs=ob), env_idx)

        # Now, we sample until `sample_until(trajectories)` is true.
        # If we just stopped then this would introduce a bias towards shorter episodes,
        # since longer episodes are more likely to still be active, i.e. in the process
        # of being sampled from. To avoid this, we continue sampling until all epsiodes
        # are complete.
        #
        # To start with, all environments are active.
        active = np.ones(venv.num_envs, dtype=bool)
        dones = np.zeros(venv.num_envs, dtype=bool)
        while np.any(active):
            # policy gets unwrapped observations (eg as dict, not dictobs)
            acts = np.array(
                venv.env_method(
                    "greedy_action", num_watch_steps=5, consecutive_actions=1
                )
            )
            obs, rews, dones, infos = venv.step(acts)
            assert isinstance(
                obs,
                (np.ndarray, dict),
            ), "Tuple observations are not supported."
            wrapped_obs = types.maybe_wrap_in_dictobs(obs)

            # If an environment is inactive, i.e. the episode completed for that
            # environment after `sample_until(trajectories)` was true, then we do
            # *not* want to add any subsequent trajectories from it. We avoid this
            # by just making it never done.
            dones &= active

            new_trajs = trajectories_accum.add_steps_and_auto_finish(
                acts,
                wrapped_obs,
                rews,
                dones,
                infos,
            )
            trajectories.extend(new_trajs)

            if sample_until(trajectories):
                # Termination condition has been reached. Mark as inactive any
                # environments where a trajectory was completed this timestep.
                active &= ~dones

        all_trajectories.extend(trajectories)
    trajectories = all_trajectories

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)  # type: ignore[arg-type]

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        if isinstance(venv.observation_space, spaces.Dict):
            exp_obs = {}
            for k, v in venv.observation_space.items():
                assert v.shape is not None
                exp_obs[k] = (n_steps + 1,) + v.shape
        else:
            obs_space_shape = venv.observation_space.shape
            assert obs_space_shape is not None
            exp_obs = (n_steps + 1,) + obs_space_shape  # type: ignore[assignment]
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        assert venv.action_space.shape is not None
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    # trajectories = [unwrap_traj(traj) for traj in trajectories]
    trajectories = [dataclasses.replace(traj, infos=None) for traj in trajectories]
    stats = rollout_stats(trajectories)
    print(f"Rollout stats: {stats}")
    return trajectories


def validation_step_on_epoch(
    bc_trainer: bc.BC,
    num_envs: int,
    exp_name: str,
    repeat_idx: int,
    best_score: float,
    best_random_reward: float,
    best_deterministic_reward: float,
):
    def callback():
        env = make_vec_env(
            lambda: build_env(domain_randomization=True),
            n_envs=num_envs,
        )
        det_env = make_vec_env(
            lambda: build_env(domain_randomization=False),
            n_envs=num_envs,
        )
        mean_reward, std_reward = evaluate_policy(
            bc_trainer.policy, env, n_eval_episodes=20
        )
        det_mean_reward, det_std_reward = evaluate_policy(
            bc_trainer.policy, det_env, n_eval_episodes=10
        )
        print("-" * 5 + f"Epoch {callback.epoch} - Validation step" + "-" * 5)
        print(f"Random env reward: {mean_reward:.2e} ± {std_reward:.2e}")
        print(f"Deterministic env reward: {det_mean_reward:.2e} ± {det_std_reward:.2e}")
        mean_reward_sample, std_reward_sample = evaluate_policy(
            bc_trainer.policy, env, n_eval_episodes=20, deterministic=False
        )
        print(
            f"Random env reward (sample): {mean_reward_sample:.2e} ± {std_reward_sample:.2e}"
        )
        det_mean_reward_sample, det_std_reward_sample = evaluate_policy(
            bc_trainer.policy, det_env, n_eval_episodes=20, deterministic=False
        )
        print(
            f"Deterministic env reward (sample): {det_mean_reward_sample:.2e} ± {det_std_reward_sample:.2e}"
        )
        print("-" * 20)
        if not DISABLE_WANDB:
            wandb.log(
                {
                    "validation/rnd_env_reward": mean_reward,
                    "validation/det_env_reward": det_mean_reward,
                    "validation/rnd_env_reward_sample": mean_reward_sample,
                    "validation/det_env_reward_sample": det_mean_reward_sample,
                },
                commit=False,
            )
        score = calculate_score(mean_reward, det_mean_reward)
        score_sample = calculate_score(mean_reward_sample, det_mean_reward_sample)
        if max(score, score_sample) > callback.best_score or (
            mean_reward > callback.best_random_reward
            and det_mean_reward > callback.best_deterministic_reward
        ):
            save_path = Path("models/bc") / exp_name
            save_path.mkdir(parents=True, exist_ok=True)
            improves_score = max(score, score_sample) > callback.best_score
            callback.best_score = max(score, score_sample)
            callback.best_random_reward = mean_reward
            callback.best_deterministic_reward = det_mean_reward
            save_policy(
                bc_trainer.policy,
                save_path / f"repeat_{repeat_idx}_best.pkl",
            )
            with open(
                save_path / f"repeat_{repeat_idx}_best_info.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "score": score,
                        "score_sample": score_sample,
                        "epoch": callback.epoch,
                        "improved score": improves_score,
                    },
                    f,
                )
            if not DISABLE_WANDB:
                wandb.log(
                    {
                        "validation/score": score,
                        "validation/score_sample": score_sample,
                        "validation/best_score": callback.best_score,
                        "validation/best_random_reward": callback.best_random_reward,
                        "validation/best_deterministic_reward": callback.best_deterministic_reward,
                    }
                )
        callback.epoch += 1

    callback.epoch = 0
    callback.best_score = best_score
    callback.best_random_reward = best_random_reward
    callback.best_deterministic_reward = best_deterministic_reward
    return callback


def calculate_score(random_env_reward: float, deterministic_env_reward: float):
    score = 0
    if deterministic_env_reward >= 3432807.680391572:
        score += 1
    if deterministic_env_reward >= 1e8:
        score += 1
    if deterministic_env_reward >= 1e9:
        score += 1
    if deterministic_env_reward >= 1e10:
        score += 1
    if deterministic_env_reward >= 2e10:
        score += 1
    if deterministic_env_reward >= 5e10:
        score += 1
    if random_env_reward >= 1e10:
        score += 1
    if random_env_reward >= 2e10:
        score += 1
    if random_env_reward >= 5e10:
        score += 1
    return score


def main(
    num_rollouts: int,
    num_envs: int,
    exp_name: str,
    add_fixed_env: bool = True,
    device: str = "auto",
    rollout_path: Path | None = None,
    n_epochs: int = 5,
    n_repeats: int = 1,
):
    rng = np.random.default_rng()
    if rollout_path is None:
        rollouts = generate_heuristic_rollouts(
            add_fixed_env, num_envs, num_rollouts, rng=rng
        )
        save_path = Path("data/rollouts") / (
            exp_name + "_" + str(num_rollouts) + ".traj"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        serialize.save(save_path, rollouts)
        print(f"Saved rollouts to {save_path}")
    else:
        rollouts = serialize.load(rollout_path)
        print(f"Loaded rollouts from {rollout_path}")
    transitions = rollout.flatten_trajectories(rollouts)
    env = make_vec_env(
        lambda: build_env(domain_randomization=True),
        n_envs=num_envs,
    )
    det_env = make_vec_env(
        lambda: build_env(domain_randomization=False),
        n_envs=num_envs,
    )
    dummy_env = FastHIVPatient(domain_randomization=False)
    best_score = 0
    best_random_reward = 0
    best_deterministic_reward = 0
    for i in range(n_repeats):
        if not DISABLE_WANDB:
            wandb.log({"num_repeat": i}, commit=False)
        bc_trainer = bc.BC(
            observation_space=dummy_env.observation_space,
            action_space=dummy_env.action_space,
            demonstrations=transitions,
            rng=rng,
            device=device,
            custom_logger=(
                configure_logger(
                    folder=Path("logs") / exp_name, format_strs=["wandb", "log"]
                )
                if not DISABLE_WANDB
                else None
            ),
        )
        callback = validation_step_on_epoch(
            bc_trainer,
            num_envs,
            exp_name,
            i,
            best_score,
            best_random_reward,
            best_deterministic_reward,
        )
        bc_trainer.train(
            n_epochs=n_epochs,
            on_epoch_end=callback,
        )
        mean_reward, std_reward = evaluate_policy(
            bc_trainer.policy, env, n_eval_episodes=10
        )
        det_mean_reward, det_std_reward = evaluate_policy(
            bc_trainer.policy, det_env, n_eval_episodes=10
        )
        print(f"Reward: {mean_reward:.2e} ± {std_reward:.2e}")
        print(f"Det env reward: {det_mean_reward:.2e} ± {det_std_reward:.2e}")
        best_score = callback.best_score
        best_random_reward = callback.best_random_reward
        best_deterministic_reward = callback.best_deterministic_reward
        # save policy
        save_path = Path("models/bc") / exp_name / f"final_{i}_{num_rollouts}.pkl"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_policy(bc_trainer.policy, save_path)
    # bc_trainer.save_policy(save_path)
    print(f"Saved policy to {save_path}")


def train():
    parser = ArgumentParser()
    parser.add_argument("--num-rollouts", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=10)
    parser.add_argument("--no-fixed-env", action="store_false", dest="add_fixed_env")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--rollout-path", "-p", type=Path, default=None)
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--name", default=generate_slug(2))
    args = parser.parse_args()
    exp_name = str(int(time.time())) + "_" + args.name
    print(f"Experiment name: {exp_name}")
    if not DISABLE_WANDB:
        wandb.init(project="hiv-imitation", name=exp_name, sync_tensorboard=True)
    main(
        num_rollouts=args.num_rollouts,
        num_envs=args.num_envs,
        exp_name=exp_name,
        add_fixed_env=args.add_fixed_env,
        device=args.device,
        rollout_path=args.rollout_path,
        n_epochs=args.n_epochs,
        n_repeats=args.n_repeats,
    )


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

SAVE_PATH = Path(__file__).parent.parent / "models"
MODEL_NAME = "bc/repeat_0_best.pkl"


class ProjectAgent:
    def act(self, observation, use_random=False):
        observation = np.log(np.maximum(observation, 1e-8))
        return self.policy.predict(observation, deterministic=True)[0]

    def save(self, path):
        th.save(self.policy, path)

    def load(self):
        self.policy = th.load(SAVE_PATH / MODEL_NAME, weights_only=False)
        
        
        
if __name__ == "__main__":
    train()