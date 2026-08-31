import math

from .entities import Agent, Type
from .rules import beats

from ..config import SPEED, COLLISION_DIAMETER


def create_agents(rng, board_size, agents_per_type):
    agents = []
    agent_id = 0
    for t in Type:
        for _ in range(agents_per_type):
            x = rng.random() * board_size
            y = rng.random() * board_size
            agents.append(Agent(id=agent_id, type=t, x=x, y=y))
            agent_id += 1
    return agents


def move_agent(agent, dx, dy, board_size):
    return (agent.x + dx * SPEED) % board_size, (agent.y + dy * SPEED) % board_size


def toroidal_dist(a, b, board_size):
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    if dx > board_size / 2:
        dx = board_size - dx
    if dy > board_size / 2:
        dy = board_size - dy
    return math.sqrt(dx * dx + dy * dy)


def _find_clusters(agents, board_size):
    n = len(agents)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if toroidal_dist(agents[i], agents[j], board_size) <= COLLISION_DIAMETER:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(agents[i])
    return list(clusters.values())


def resolve_collisions(agents, board_size):
    snapshot = {a.id: a.type for a in agents}
    clusters = _find_clusters(agents, board_size)
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        types_here = {snapshot[a.id] for a in cluster}
        if len(types_here) != 2:
            continue
        t_list = list(types_here)
        if beats(t_list[0], t_list[1]):
            winner, loser = t_list[0], t_list[1]
        else:
            winner, loser = t_list[1], t_list[0]
        for a in cluster:
            if snapshot[a.id] == loser:
                a.type = winner


def population_counts(agents):
    counts = {t: 0 for t in Type}
    for a in agents:
        counts[a.type] += 1
    return counts
