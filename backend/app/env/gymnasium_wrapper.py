import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import AGENTS_PER_TYPE, BOARD_SIZE, EPISODE_LENGTH, OBS_WINDOW
from .entities import Type
from .rps_env import RPSEnv

OBS_PER_AGENT = OBS_WINDOW * OBS_WINDOW * len(Type) + len(Type)
NUM_TYPES = len(Type)


class RPSGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=BOARD_SIZE, agents_per_type=AGENTS_PER_TYPE,
                 episode_length=EPISODE_LENGTH, seed=None):
        super().__init__()
        self.env = RPSEnv(board_size=board_size, agents_per_type=agents_per_type,
                          episode_length=episode_length, seed=seed)
        self.num_agents = agents_per_type * NUM_TYPES
        self.obs_size = self.num_agents * OBS_PER_AGENT
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.obs_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(9)

    def _build_obs(self):
        obs_dict = self.env.observations()
        parts = []
        for i in range(self.num_agents):
            window, own = obs_dict[i]
            parts.append(np.array(window, dtype=np.float32).reshape(-1))
            parts.append(np.array(own, dtype=np.float32))
        return np.concatenate(parts)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        return self._build_obs(), {}

    def step(self, action):
        actions = {i: int(action) for i in range(self.num_agents)}
        _, rewards, done, info = self.env.step(actions)
        total = sum(rewards.values())
        mean_reward = total / len(rewards) if rewards else 0.0
        conversions = info.get("conversions", 0)
        shaped = mean_reward + (0.5 if conversions > 0 else 0.0)
        return self._build_obs(), shaped, done, False, info