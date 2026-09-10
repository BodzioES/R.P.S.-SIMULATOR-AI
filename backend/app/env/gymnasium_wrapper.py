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
from .grid import euclidean_dist
from .rps_env import RPSEnv

NUM_TYPES = len(Type)


def calc_obs_per_agent(vision_mode=VISION_MODE, vision_k=VISION_K, obs_window=OBS_WINDOW):
    if vision_mode == "radius":
        return vision_k * (2 + NUM_TYPES) + NUM_TYPES + 4 + 3
    return obs_window * obs_window * NUM_TYPES + NUM_TYPES + 4 + 3


class RPSGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=BOARD_SIZE, agents_per_type=AGENTS_PER_TYPE,
                 episode_length=EPISODE_LENGTH, speed=1.0,
                 vision_mode=VISION_MODE, vision_radius=VISION_RADIUS,
                 vision_k=VISION_K, obs_window=OBS_WINDOW,
                 seed=None):
        super().__init__()
        self.env = RPSEnv(
            board_size=board_size, agents_per_type=agents_per_type,
            episode_length=episode_length, speed=speed,
            vision_mode=vision_mode, vision_radius=vision_radius,
            vision_k=vision_k, obs_window=obs_window,
            seed=seed,
        )
        self.num_agents = agents_per_type * NUM_TYPES
        obs_per_agent = calc_obs_per_agent(vision_mode, vision_k, obs_window)
        self.obs_size = self.num_agents * obs_per_agent
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
        shaped += conversions * 5.0
        shaped -= 0.01
        # wall/corner penalty
        wall_hits = 0
        corner_hits = 0
        for a in self.env.agents:
            near_wall = min(a.x, a.y, self.env.board_size - a.x, self.env.board_size - a.y)
            if near_wall < 0.7:
                wall_hits += 1
                if min(a.x, self.env.board_size - a.x) < 0.7 and min(a.y, self.env.board_size - a.y) < 0.7:
                    corner_hits += 1
        shaped -= wall_hits * 0.2
        shaped -= corner_hits * 0.25

        # bonus za grupe
        for a in self.env.agents:
            allies_near = 0
            for b in self.env.agents:
                if b.type == a.type and b.id != a.id:
                    if euclidean_dist(a, b) < 3.0:
                        allies_near += 1
            shaped += allies_near * 0.3

        # bonus za otoczenie
        for enemy in self.env.agents:
            enemies_in_range = 0
            allies_in_range = 0
            for a in self.env.agents:
                if a.id == enemy.id:
                    continue
                d = euclidean_dist(a, enemy)
                if d < 2.0:
                    if a.type == enemy.type:
                        enemies_in_range += 1
                    else:
                        allies_in_range += 1
            if allies_in_range > enemies_in_range:
                shaped += (allies_in_range - enemies_in_range) * 0.5

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
