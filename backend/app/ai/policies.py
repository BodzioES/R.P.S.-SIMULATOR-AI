import random

from ..env.grid import MOVES

NUM_ACTIONS = len(MOVES)


class RandomPolicy:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def actions(self, env):
        return {a.id: self.rng.randrange(NUM_ACTIONS) for a in env.agents}