from ..config import STEP_COST


def compute_rewards(prev_pop, prev_types, new_pop, agents):
    rewards = {}
    for a in agents:
        team = prev_types[a.id]
        rewards[a.id] = (new_pop[team] - prev_pop[team]) + STEP_COST
    return rewards