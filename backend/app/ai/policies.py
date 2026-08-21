import random

from ..env.entities import Type
from ..env.grid import MOVES
from ..env.rps_env import RPSEnv

NUM_ACTIONS = len(MOVES)


class RandomPolicy:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def actions(self, env):
        return {a.id: self.rng.randrange(NUM_ACTIONS) for a in env.agents}


class LearnedPolicy:
    def __init__(self, model_path):
        from stable_baselines3 import PPO
        self.model = PPO.load(model_path)

    def actions(self, env):
        obs_dict = env.observations()
        actions = {}
        for agent in env.agents:
            window, own = obs_dict[agent.id]
            import numpy as np
            flat = np.concatenate([np.array(window).reshape(-1), np.array(own)])
            flat = flat.reshape(1, -1).astype(np.float32)
            action, _ = self.model.predict(flat, deterministic=True)
            actions[agent.id] = int(action[0])
        return actions