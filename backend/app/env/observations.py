from ..config import OBS_WINDOW
from .entities import Type

NUM_TYPES = len(Type)


def encode_observation(agent, cell_map, board_size):
    r = OBS_WINDOW // 2
    window = [[[0.0] * NUM_TYPES for _ in range(OBS_WINDOW)] for _ in range(OBS_WINDOW)]

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            cell = cell_map.get(((agent.x + dx) % board_size, (agent.y + dy) % board_size))
            if cell is None:
                continue
            for other in cell:
                window[dy + r][dx + r][other.type.value] = 1.0

    own = [0.0] * NUM_TYPES
    own[agent.type.value] = 1.0
    return window, own