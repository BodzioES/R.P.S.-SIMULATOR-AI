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


def train(
    total_episodes=1000,
    eval_every=10,
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

    total_timesteps = total_episodes * episode_length
    eval_freq = eval_every * episode_length

    env = DummyVecEnv([make_env(**env_kwargs)])
    eval_env = DummyVecEnv([make_env(**env_kwargs)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_path / "best"),
        log_path=str(log_path),
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(log_path),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
    )

    model.save(str(ckpt_path / "final"))
    print(f"\nTrening zakonczony.")
    print(f"Checkpointy: {ckpt_path.resolve()}")
    print(f"TensorBoard: tensorboard --logdir {log_path.resolve()}")
    return model