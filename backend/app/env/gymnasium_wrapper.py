import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..config import (
    AGENTS_PER_TYPE,
    BOARD_SIZE,
    EPISODE_LENGTH,
    OBS_WINDOW,
    VISION_K,
    VISION_MODE,
    VISION_RADIUS,
)
from .entities import Type
from .rps_env import RPSEnv

NUM_TYPES = len(Type)
if VISION_MODE == "radius":
    OBS_PER_AGENT = VISION_K * (2 + NUM_TYPES) + NUM_TYPES + 4 + 3  # K*(dx,dy+onehot) + own + wall + pop
else:
    OBS_PER_AGENT = OBS_WINDOW * OBS_WINDOW * NUM_TYPES + NUM_TYPES + 4 + 3  # +4 wall +3 pop


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
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_agents * 2,),
            dtype=np.float32,
        )

    def _build_obs(self):
        obs_dict = self.env.observations()
        parts = []
        for i in range(self.num_agents):
            window, own, wall, pop = obs_dict[i]
            parts.append(np.array(window, dtype=np.float32).reshape(-1))
            parts.append(np.array(own, dtype=np.float32))
            parts.append(np.array(wall, dtype=np.float32))
            parts.append(np.array(pop, dtype=np.float32))
        return np.concatenate(parts)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        return self._build_obs(), {}

    def step(self, action):
        raw = np.array(action, dtype=np.float32).reshape(self.num_agents, 2)
        actions = {}
        for i in range(self.num_agents):
            dx, dy = float(raw[i, 0]), float(raw[i, 1])
            mag = (dx * dx + dy * dy) ** 0.5
            if mag > 1.0:
                dx /= mag
                dy /= mag
            actions[i] = (dx, dy)

        _, rewards, done, info = self.env.step(actions)
        total = sum(rewards.values())
        mean_reward = total / len(rewards) if rewards else 0.0
        conversions = info.get("conversions", 0)

        shaped = mean_reward
        shaped += conversions * 2.0
        shaped -= 0.01
        # wall/corner penalty: zniechęć do chowania się przy ścianach
        wall_hits = 0
        corner_hits = 0
        for a in self.env.agents:
            near_wall = min(a.x, a.y, self.env.board_size - a.x, self.env.board_size - a.y)
            if near_wall < 0.7:
                wall_hits += 1
                if min(a.x, self.env.board_size - a.x) < 0.7 and min(a.y, self.env.board_size - a.y) < 0.7:
                    corner_hits += 1
        shaped -= wall_hits * 0.05
        shaped -= corner_hits * 0.08

        populations = info.get("populations", {})
        total_pop = sum(populations.values()) or 1
        max_pop = max(populations.values())
        max_share = max_pop / total_pop
        shaped += (max_share - 1 / 3) * 1.0

        threshold_60 = int(total_pop * 0.6 + 0.999)
        done_bonus_60 = 15.0 if max_pop >= threshold_60 else 0.0

        if done and info.get("winning_type") is not None:
            shaped += 200.0
        elif done:
            shaped -= 20.0
            shaped += done_bonus_60

        return self._build_obs(), shaped, done, False, info
