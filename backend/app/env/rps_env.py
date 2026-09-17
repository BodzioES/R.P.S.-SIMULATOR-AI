import random

from ..config import (
    AGENTS_PER_TYPE, BOARD_SIZE, EPISODE_LENGTH,
    OBS_WINDOW, VISION_K, VISION_K_MESSAGING, VISION_MODE, VISION_RADIUS,
)
from .entities import Type
from .grid import create_agents, move_agent, population_counts, resolve_collisions
from .observations import encode_observation, compute_messages
from .reward import compute_rewards


class RPSEnv:
    def __init__(self, board_size=BOARD_SIZE, agents_per_type=AGENTS_PER_TYPE,
                 episode_length=EPISODE_LENGTH, speed=1.0,
                 vision_mode=VISION_MODE, vision_radius=VISION_RADIUS,
                 vision_k=VISION_K, obs_window=OBS_WINDOW,
                 vision_k_messaging=VISION_K_MESSAGING,
                 seed=None):
        self.board_size = board_size
        self.agents_per_type = agents_per_type
        self.episode_length = episode_length
        self.speed = speed
        self.vision_mode = vision_mode
        self.vision_radius = vision_radius
        self.vision_k = vision_k
        self.obs_window = obs_window
        self.vision_k_messaging = vision_k_messaging
        self.rng = random.Random(seed)
        self.agents = []
        self.steps = 0
        self.done = False
        self.messages = {}

    def reset(self):
        self.agents = create_agents(self.rng, self.board_size, self.agents_per_type)
        self.steps = 0
        self.done = False
        self._compute_messages()
        return self.state(), {"populations": self.populations}

    def _compute_messages(self):
        own_msgs = {}
        for agent in self.agents:
            own_msgs[agent.id] = compute_messages(agent, self.agents, self.board_size, self.vision_radius)

        self.messages = {}
        for agent in self.agents:
            best_msgs = []
            for other in self.agents:
                if other.id == agent.id:
                    continue
                if other.type == agent.type:
                    dx = other.x - agent.x
                    dy = other.y - agent.y
                    dist = (dx * dx + dy * dy) ** 0.5
                    if dist <= self.vision_radius:
                        best_msgs.append(own_msgs[other.id])
            best_msgs.sort(key=lambda m: m[0], reverse=True)
            k = self.vision_k_messaging
            slots = []
            for i in range(k):
                if i < len(best_msgs):
                    slots.extend(best_msgs[i])
                else:
                    slots.extend([0.0, 0.0])
            self.messages[agent.id] = slots

    def state(self):
        return [
            {"id": a.id, "type": a.type.value, "x": round(a.x, 3), "y": round(a.y, 3)}
            for a in self.agents
        ]

    @property
    def populations(self):
        return population_counts(self.agents)

    @property
    def winning_type(self):
        counts = self.populations
        total = len(self.agents)
        for t, count in counts.items():
            if count == total:
                return t
        return None

    def observations(self):
        pops = self.populations
        return {
            a.id: encode_observation(
                a, self.agents, self.board_size, pops,
                vision_mode=self.vision_mode,
                vision_radius=self.vision_radius,
                vision_k=self.vision_k,
                obs_window=self.obs_window,
                messages=self.messages.get(a.id, [0.0, 0.0]),
                vision_k_messaging=self.vision_k_messaging,
            )
            for a in self.agents
        }

    def step(self, actions):
        if self.done:
            raise RuntimeError("Environment is done; call reset() first.")

        prev_pop = population_counts(self.agents)
        prev_types = {a.id: a.type for a in self.agents}

        for agent in self.agents:
            dx, dy = actions[agent.id]
            agent.x, agent.y = move_agent(agent, dx, dy, self.board_size, self.speed)

        resolve_collisions(self.agents)
        self._compute_messages()

        new_pop = population_counts(self.agents)
        rewards = compute_rewards(prev_pop, prev_types, new_pop, self.agents)
        conversions = sum(1 for a in self.agents if prev_types[a.id] != a.type)

        self.steps += 1
        if self.winning_type is not None or (self.episode_length > 0 and self.steps >= self.episode_length):
            self.done = True

        info = {
            "populations": new_pop,
            "winning_type": self.winning_type,
            "conversions": conversions,
        }
        return self.state(), rewards, self.done, info

    def render(self):
        grid = [["." for _ in range(self.board_size)] for _ in range(self.board_size)]
        for a in self.agents:
            grid[int(a.y) % self.board_size][int(a.x) % self.board_size] = a.type.name[0]
        return "\n".join("".join(row) for row in grid)
