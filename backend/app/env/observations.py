from .entities import Type

NUM_TYPES = len(Type)


def relative_delta(agent, other, board_size):
    dx = other.x - agent.x
    dy = other.y - agent.y
    return dx, dy


def _encode_grid(agent, agents, board_size, obs_window=9):
    r = obs_window // 2
    window = [[[0.0] * NUM_TYPES for _ in range(obs_window)] for _ in range(obs_window)]
    for other in agents:
        if other.id == agent.id:
            continue
        dx, dy = relative_delta(agent, other, board_size)
        bx = round(dx)
        by = round(dy)
        if -r <= bx <= r and -r <= by <= r:
            window[by + r][bx + r][other.type.value] = 1.0
    return window


def _encode_radius(agent, agents, board_size, vision_radius=4.0, vision_k=10):
    cands = []
    for other in agents:
        if other.id == agent.id:
            continue
        dx, dy = relative_delta(agent, other, board_size)
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= vision_radius:
            cands.append((dist, dx, dy, other.type))
    cands.sort(key=lambda x: x[0])
    slots = []
    for i in range(vision_k):
        if i < len(cands):
            _, dx, dy, t = cands[i]
            onehot = [0.0] * NUM_TYPES
            onehot[t.value] = 1.0
            slots.append([dx / vision_radius, dy / vision_radius] + onehot)
        else:
            slots.append([0.0, 0.0] + [0.0] * NUM_TYPES)
    return slots


def compute_messages(agent, agents, board_size, vision_radius=7.0):
    """Oblicz sygnaly komunikacji: [opportunity_level, danger_level]"""
    # Types that this agent can convert (beats)
    from .rules import beats
    can_convert = set()
    vulnerable_to = set()
    for t in Type:
        if t != agent.type:
            if beats(agent.type, t):
                can_convert.add(t)
            elif beats(t, agent.type):
                vulnerable_to.add(t)

    # Count nearby enemies
    convertibles_near = 0
    threats_near = 0
    for other in agents:
        if other.id == agent.id:
            continue
        if other.type == agent.type:
            continue
        dx, dy = relative_delta(agent, other, board_size)
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= vision_radius:
            if other.type in can_convert:
                convertibles_near += 1
            elif other.type in vulnerable_to:
                threats_near += 1

    opportunity = min(convertibles_near / 3.0, 1.0)
    danger = min(threats_near / 3.0, 1.0)
    return [opportunity, danger]


def encode_observation(agent, agents, board_size, populations=None,
                       vision_mode="grid", vision_radius=4.0, vision_k=10, obs_window=9,
                       messages=None, vision_k_messaging=10):
    own = [0.0] * NUM_TYPES
    own[agent.type.value] = 1.0
    wall = [
        agent.x / board_size,
        agent.y / board_size,
        (board_size - agent.x) / board_size,
        (board_size - agent.y) / board_size,
    ]
    if populations is not None:
        total = sum(populations.values()) or 1
        pop = [populations[t] / total for t in Type]
    else:
        pop = [1 / 3, 1 / 3, 1 / 3]
    if vision_mode == "radius":
        slots = _encode_radius(agent, agents, board_size, vision_radius, vision_k)
        return slots, own, wall, pop, messages
    window = _encode_grid(agent, agents, board_size, obs_window)
    return window, own, wall, pop, messages
