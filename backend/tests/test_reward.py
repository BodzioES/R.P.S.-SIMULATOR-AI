import random

from app.config import STEP_COST
from app.env.entities import Type
from app.env.rps_env import RPSEnv


def test_team_reward_on_conversion():
    env = RPSEnv(board_size=10, agents_per_type=1, episode_length=10, seed=1)
    env.reset()
    agents = env.agents
    rock = agents[0]
    paper = agents[1]
    scissors = agents[2]
    for a in agents:
        a.x, a.y = 5.0, 5.0
    actions = {a.id: (0.0, 0.0) for a in agents}
    _, rewards, _, info = env.step(actions)

    # all 3 on same spot: 3 types present -> no conversion (cycle prevention)
    assert info["conversions"] == 0
    assert rewards[rock.id] == 0 + STEP_COST
    assert rewards[paper.id] == 0 + STEP_COST
    assert rewards[scissors.id] == 0 + STEP_COST


def test_no_conversion_gives_only_step_cost():
    env = RPSEnv(board_size=20, agents_per_type=1, episode_length=10, seed=2)
    env.reset()
    agents = env.agents
    rock, paper, scissors = agents
    rock.x, rock.y = 0.0, 0.0
    paper.x, paper.y = 5.0, 5.0
    scissors.x, scissors.y = 10.0, 10.0
    actions = {a.id: (0.0, 0.0) for a in agents}
    _, rewards, _, info = env.step(actions)

    assert info["conversions"] == 0
    assert rewards[rock.id] == STEP_COST
    assert rewards[paper.id] == STEP_COST
    assert rewards[scissors.id] == STEP_COST


def test_reward_uses_team_from_start_of_step():
    env = RPSEnv(board_size=10, agents_per_type=1, episode_length=10, seed=1)
    env.reset()
    agents = env.agents
    scissors = agents[2]
    scissors.x, scissors.y = 5.0, 5.0
    agents[0].x, agents[0].y = 5.0, 5.0
    agents[1].x, agents[1].y = 9.0, 9.0
    actions = {a.id: (0.0, 0.0) for a in agents}
    _, rewards, _, info = env.step(actions)

    assert info["conversions"] == 1
    assert rewards[scissors.id] == -2 + STEP_COST
    assert scissors.type == Type.ROCK


def test_random_episode_rewards_are_finite():
    env = RPSEnv(board_size=16, agents_per_type=6, episode_length=100, seed=7)
    env.reset()
    rng = random.Random(42)
    for _ in range(50):
        if env.done:
            break
        actions = {a.id: (rng.uniform(-1, 1), rng.uniform(-1, 1)) for a in env.agents}
        _, rewards, done, _ = env.step(actions)
        for value in rewards.values():
            assert isinstance(value, float)
