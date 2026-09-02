from ..config import OBS_WINDOW, VISION_K, VISION_MODE, VISION_RADIUS
from .entities import Type

NUM_TYPES = len(Type)


def relative_delta(agent, other, board_size):
    dx = other.x - agent.x
    dy = other.y - agent.y
    return dx, dy


def _encode_grid(agent, agents, board_size):
    r = OBS_WINDOW // 2
    window = [[[0.0] * NUM_TYPES for _ in range(OBS_WINDOW)] for _ in range(OBS_WINDOW)]
    for other in agents:
        if other.id == agent.id:
            continue
        dx, dy = relative_delta(agent, other, board_size)
        bx = round(dx)
        by = round(dy)
        if -r <= bx <= r and -r <= by <= r:
            window[by + r][bx + r][other.type.value] = 1.0
    return window


def _encode_radius(agent, agents, board_size):
    cands = []
    for other in agents:
        if other.id == agent.id:
            continue
        dx, dy = relative_delta(agent, other, board_size)
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= VISION_RADIUS:
            cands.append((dist, dx, dy, other.type))
    cands.sort(key=lambda x: x[0])
    slots = []
    for i in range(VISION_K):
        if i < len(cands):
            _, dx, dy, t = cands[i]
            onehot = [0.0] * NUM_TYPES
            onehot[t.value] = 1.0
            slots.append([dx / VISION_RADIUS, dy / VISION_RADIUS] + onehot)
        else:
            slots.append([0.0, 0.0] + [0.0] * NUM_TYPES)
    return slots


def encode_observation(agent, agents, board_size, populations=None):
    own = [0.0] * NUM_TYPES
    own[agent.type.value] = 1.0
    wall = [
        agent.x / board_size,
        agent.y / board_size,
        (board_size - agent.x) / board_size,
        (board_size - agent.y) / board_size,
    ]
    # global population normalized 0..1 (agent wie ile kogo zostalo)
    if populations is not None:
        total = sum(populations.values()) or 1
        pop = [populations[t] / total for t in Type]
    else:
        pop = [1 / 3, 1 / 3, 1 / 3]
    if VISION_MODE == "radius":
        slots = _encode_radius(agent, agents, board_size)
        return slots, own, wall, pop
    window = _encode_grid(agent, agents, board_size)
    return window, own, wall, pop
