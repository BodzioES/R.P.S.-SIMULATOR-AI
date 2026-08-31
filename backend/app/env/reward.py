from ..config import STEP_COST


def compute_rewards(prev_pop, prev_types, new_pop, agents):
    rewards = {}
    for a in agents:
        team = prev_types[a.id]
        diff = new_pop[team] - prev_pop[team]
        scaled = diff * 2 if diff < 0 else diff
        rewards[a.id] = scaled + STEP_COST
    return rewards