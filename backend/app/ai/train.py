import os
import random
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from ..env.gymnasium_wrapper import RPSGymEnv


def make_env(**kwargs):
    def _init():
        return RPSGymEnv(**kwargs)
    return _init


def evaluate(model, env_kwargs, n_episodes=20):
    rewards = []
    wins = []
    for _ in range(n_episodes):
        env = RPSGymEnv(**env_kwargs)
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action[0]))
            total += reward
            done = terminated or truncated
        rewards.append(total)
        wins.append(1 if info.get("winner") else 0)
    return float(np.mean(rewards)), float(np.mean(wins)) * 100


def train(
    total_episodes=1000,
    eval_every=20,
    save_every=50,
    board_size=64,
    agents_per_type=20,
    episode_length=1000,
    log_dir="runs",
    checkpoint_dir="checkpoints",
    seed=42,
):
    env_kwargs = dict(
        board_size=board_size,
        agents_per_type=agents_per_type,
        episode_length=episode_length,
        seed=seed,
    )

    log_path = Path(log_dir)
    ckpt_path = Path(checkpoint_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env(**env_kwargs)])

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log=str(log_path),
    )

    best_reward = -float("inf")
    best_path = ckpt_path / "best.pth"
    total_steps = 0
    steps_per_episode = episode_length

    for ep in range(1, total_episodes + 1):
        model.learn(total_timesteps=steps_per_episode, reset_num_timesteps=False)
        total_steps += steps_per_episode

        if ep % eval_every == 0:
            mean_rew, win_rate = evaluate(model, env_kwargs, n_episodes=20)
            tag = f"eval/mean_reward"
            model.logger.record(tag, mean_rew)
            model.logger.record("eval/win_rate_%", win_rate)
            model.logger.dump(total_steps)
            improved = mean_rew > best_reward
            if improved:
                best_reward = mean_rew
                model.save(str(best_path / "model"))
                print(f"  * BEST ep {ep}: reward={mean_rew:.3f} win={win_rate:.0f}%")

        if ep % save_every == 0:
            model.save(str(ckpt_path / f"ep_{ep:05d}"))

    model.save(str(ckpt_path / "final"))
    print(f"\nTrening zakonczony. Best reward: {best_reward:.3f}")
    print(f"Checkpointy: {ckpt_path.resolve()}")
    print(f"TensorBoard: tensorboard --logdir {log_path.resolve()}")
    return model