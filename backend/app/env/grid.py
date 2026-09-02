import math

from .entities import Agent, Type
from .rules import beats

from ..config import SPEED, COLLISION_DIAMETER, AGENT_RADIUS


def create_agents(rng, board_size, agents_per_type):
    agents = []
    agent_id = 0
    margin = AGENT_RADIUS + 0.01
    for t in Type:
        for _ in range(agents_per_type):
            x = rng.uniform(margin, board_size - margin)
            y = rng.uniform(margin, board_size - margin)
            agents.append(Agent(id=agent_id, type=t, x=x, y=y))
            agent_id += 1
    return agents


def move_agent(agent, dx, dy, board_size):
    new_x = agent.x + dx * SPEED
    new_y = agent.y + dy * SPEED

    if new_x < 0:
        new_x = -new_x
    elif new_x >= board_size:
        new_x = 2 * board_size - new_x

    if new_y < 0:
        new_y = -new_y
    elif new_y >= board_size:
        new_y = 2 * board_size - new_y

    new_x = max(0.0, min(new_x, board_size - 0.001))
    new_y = max(0.0, min(new_y, board_size - 0.001))

    return new_x, new_y


def euclidean_dist(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def _find_clusters(agents):
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
            if euclidean_dist(agents[i], agents[j]) <= COLLISION_DIAMETER:
                union(i, j)

    clusters = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(agents[i])
    return list(clusters.values())


def resolve_collisions(agents):
    snapshot = {a.id: a.type for a in agents}
    clusters = _find_clusters(agents)
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
