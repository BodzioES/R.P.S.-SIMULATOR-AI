from ..config import OBS_WINDOW, BOARD_SIZE
from .entities import Type

NUM_TYPES = len(Type)


def toroidal_delta(a, b, board_size):
    dx = b.x - a.x
    dy = b.y - a.y
    if dx > board_size / 2:
        dx -= board_size
    elif dx < -board_size / 2:
        dx += board_size
    if dy > board_size / 2:
        dy -= board_size
    elif dy < -board_size / 2:
        dy += board_size
    return dx, dy


def encode_observation(agent, agents, board_size):
    r = OBS_WINDOW // 2
    window = [[[0.0] * NUM_TYPES for _ in range(OBS_WINDOW)] for _ in range(OBS_WINDOW)]

    for other in agents:
        if other.id == agent.id:
            continue
        dx, dy = toroidal_delta(agent, other, board_size)
        bx = round(dx)
        by = round(dy)
        if -r <= bx <= r and -r <= by <= r:
            window[by + r][bx + r][other.type.value] = 1.0

    own = [0.0] * NUM_TYPES
    own[agent.type.value] = 1.0
    return window, own
