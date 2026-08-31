import random

from ..config import AGENTS_PER_TYPE, BOARD_SIZE, EPISODE_LENGTH
from .entities import Type
from .grid import create_agents, move_agent, population_counts, resolve_collisions
from .observations import encode_observation
from .reward import compute_rewards


class RPSEnv:
    def __init__(self, board_size=BOARD_SIZE, agents_per_type=AGENTS_PER_TYPE,
                 episode_length=EPISODE_LENGTH, seed=None):
        self.board_size = board_size
        self.agents_per_type = agents_per_type
        self.episode_length = episode_length
        self.rng = random.Random(seed)
        self.agents = []
        self.steps = 0
        self.done = False

    def reset(self):
        self.agents = create_agents(self.rng, self.board_size, self.agents_per_type)
        self.steps = 0
        self.done = False
        return self.state(), {"populations": self.populations}

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
        return {
            a.id: encode_observation(a, self.agents, self.board_size)
            for a in self.agents
        }

    def step(self, actions):
        if self.done:
            raise RuntimeError("Environment is done; call reset() first.")

        prev_pop = population_counts(self.agents)
        prev_types = {a.id: a.type for a in self.agents}

        for agent in self.agents:
            dx, dy = actions[agent.id]
            agent.x, agent.y = move_agent(agent, dx, dy, self.board_size)

        resolve_collisions(self.agents, self.board_size)

        new_pop = population_counts(self.agents)
        rewards = compute_rewards(prev_pop, prev_types, new_pop, self.agents)
        conversions = sum(1 for a in self.agents if prev_types[a.id] != a.type)

        self.steps += 1
        if self.winning_type is not None or self.steps >= self.episode_length:
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
