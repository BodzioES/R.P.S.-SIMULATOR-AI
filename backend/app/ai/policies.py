import json
import random

from ..env.entities import Type
from ..env.rps_env import RPSEnv


class RandomPolicy:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def actions(self, env):
        actions = {}
        for a in env.agents:
            dx = self.rng.uniform(-1.0, 1.0)
            dy = self.rng.uniform(-1.0, 1.0)
            mag = (dx * dx + dy * dy) ** 0.5
            if mag > 1.0:
                dx /= mag
                dy /= mag
            actions[a.id] = (dx, dy)
        return actions


def _read_training_config(model_dir):
    config_path = model_dir / "training_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class LearnedPolicy:
    def __init__(self, model_path, board_size=None, agents_per_type=None, vecnorm_path=None):
        from pathlib import Path
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        from ..env.gymnasium_wrapper import RPSGymEnv

        model_path = Path(model_path)
        if not model_path.exists():
            alt = model_path.parent / "best_model.zip"
            if alt.exists():
                model_path = alt

        # czytaj training_config z katalogu modelu
        cfg = _read_training_config(model_path.parent)
        vision_mode = cfg.get("vision_mode", "grid")
        vision_radius = cfg.get("vision_radius", 4.0)
        vision_k = cfg.get("vision_k", 10)
        obs_window = cfg.get("obs_window", 9)
        vision_k_messaging = cfg.get("vision_k_messaging", 10)

        self.board_size = board_size
        self.agents_per_type = agents_per_type
        self.vecnorm_path = Path(vecnorm_path) if vecnorm_path else (model_path.parent / "vecnorm_stats.pkl")

        if self.vecnorm_path.exists() and board_size is not None and agents_per_type is not None:
            def make_env():
                return RPSGymEnv(
                    board_size=board_size, agents_per_type=agents_per_type,
                    vision_mode=vision_mode, vision_radius=vision_radius,
                    vision_k=vision_k, obs_window=obs_window,
                    vision_k_messaging=vision_k_messaging,
                )
            dummy = DummyVecEnv([make_env])
            self.vec_env = VecNormalize.load(str(self.vecnorm_path), dummy)
            self.vec_env.training = False
            self.vec_env.norm_reward = False
        else:
            self.vec_env = None

        self.model = PPO.load(str(model_path))

    def _build_flat_obs(self, env):
        import numpy as np

        obs_dict = env.observations()
        parts = []
        for i in range(len(env.agents)):
            window, own, wall, pop, msgs = obs_dict[i]
            parts.append(np.array(window, dtype=np.float32).reshape(-1))
            parts.append(np.array(own, dtype=np.float32))
            parts.append(np.array(wall, dtype=np.float32))
            parts.append(np.array(pop, dtype=np.float32))
            if msgs is not None:
                parts.append(np.array(msgs, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def actions(self, env):
        import numpy as np

        flat = self._build_flat_obs(env)
        if self.vec_env is not None:
            obs = self.vec_env.normalize_obs(flat.reshape(1, -1))
        else:
            obs = flat.reshape(1, -1)

        action, _ = self.model.predict(obs, deterministic=True)
        raw = np.array(action, dtype=np.float32).reshape(-1, 2)
        actions = {}
        for idx, a in enumerate(env.agents):
            dx, dy = float(raw[idx, 0]), float(raw[idx, 1])
            mag = (dx * dx + dy * dy) ** 0.5
            if mag > 1.0:
                dx /= mag
                dy /= mag
            actions[a.id] = (dx, dy)
        return actions
