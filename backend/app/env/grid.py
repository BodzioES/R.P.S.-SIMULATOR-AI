from collections import defaultdict

from .entities import Agent, Type
from .rules import beats

MOVES = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]


def create_agents(rng, board_size, agents_per_type):
    agents = []
    used = set()
    agent_id = 0
    for t in Type:
        for _ in range(agents_per_type):
            while True:
                x = rng.randrange(board_size)
                y = rng.randrange(board_size)
                if (x, y) not in used:
                    used.add((x, y))
                    break
            agents.append(Agent(id=agent_id, type=t, x=x, y=y))
            agent_id += 1
    return agents


def move_agent(agent, action, board_size):
    dx, dy = MOVES[action]
    return (agent.x + dx) % board_size, (agent.y + dy) % board_size


def group_by_cell(agents):
    by_cell = defaultdict(list)
    for a in agents:
        by_cell[(a.x, a.y)].append(a)
    return by_cell


def resolve_collisions(agents):
    for cell_agents in group_by_cell(agents).values():
        if len(cell_agents) < 2:
            continue
        ordered = sorted(cell_agents, key=lambda a: a.id)
        original = {a.id: a.type for a in ordered}
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                a, b = ordered[i], ordered[j]
                if beats(original[a.id], original[b.id]):
                    b.type = original[a.id]


def population_counts(agents):
    counts = {t: 0 for t in Type}
    for a in agents:
        counts[a.type] += 1
    return counts