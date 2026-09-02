from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    episode_length=300,
    log_dir="runs",
    checkpoint_dir="checkpoints",
    seed=42,
    resume=False,
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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env = DummyVecEnv([make_env(**env_kwargs)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    vecnorm_path = ckpt_path / "vecnorm_stats.pkl"
    model_path = ckpt_path / "model.zip"

    if resume and model_path.exists() and vecnorm_path.exists():
        print(f"Resuming from {model_path}")
        env = VecNormalize.load(str(vecnorm_path), env)
        env.norm_reward = True
        env.clip_reward = 10.0
        model = PPO.load(str(model_path), env=env, device="cpu")
        eval_env = VecNormalize.load(str(vecnorm_path), eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        if resume:
            print(f"No checkpoint found at {model_path}, starting fresh.")
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            target_kl=0.1,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=str(log_path),
        )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(ckpt_path / "best"),
        log_path=str(log_path),
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        reset_num_timesteps=not resume,
    )

    model.save(str(ckpt_path / "final"))
    # alias dla kompatybilnosci z /api/sim/start (szuka model.zip) i view_trained.py
    final_path = ckpt_path / "final.zip"
    if final_path.exists():
        import shutil
        shutil.copy(str(final_path), str(ckpt_path / "model.zip"))
    best_model = ckpt_path / "best" / "best_model.zip"
    if best_model.exists():
        import shutil
        shutil.copy(str(best_model), str(ckpt_path / "best" / "model.zip"))
    env.save(str(ckpt_path / "vecnorm_stats.pkl"))
    print(f"\nTraining finished.")
    print(f"Checkpoints: {ckpt_path.resolve()}")
    print(f"TensorBoard: tensorboard --logdir {log_path.resolve()}")
    return model