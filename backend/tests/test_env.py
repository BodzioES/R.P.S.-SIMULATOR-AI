import random

from app.env.entities import Agent, Type
from app.env.grid import create_agents, move_agent, resolve_collisions
from app.env.rps_env import RPSEnv


def test_reset_creates_expected_agents():
    env = RPSEnv(board_size=10, agents_per_type=5, seed=1)
    env.reset()
    assert len(env.agents) == 15
    assert env.populations == {
        Type.ROCK: 5,
        Type.PAPER: 5,
        Type.SCISSORS: 5,
    }


def test_reset_agents_have_float_coords():
    env = RPSEnv(board_size=10, agents_per_type=5, seed=1)
    env.reset()
    for a in env.agents:
        assert isinstance(a.x, float)
        assert isinstance(a.y, float)
        assert 0 <= a.x < 10
        assert 0 <= a.y < 10


def test_move_bounces_off_wall():
    agent = Agent(id=0, type=Type.ROCK, x=0.1, y=0.1)
    x, y = move_agent(agent, -1.0, -1.0, board_size=10)
    assert x > 0.0
    assert y > 0.0
    assert x < 0.6
    assert y < 0.6


def test_stay_action_keeps_position():
    agent = Agent(id=0, type=Type.ROCK, x=5.0, y=5.0)
    x, y = move_agent(agent, 0.0, 0.0, board_size=10)
    assert abs(x - 5.0) < 0.001
    assert abs(y - 5.0) < 0.001


def test_collision_converts_scissors_to_rock():
    rock = Agent(id=0, type=Type.ROCK, x=3.0, y=3.0)
    scissors = Agent(id=1, type=Type.SCISSORS, x=3.0, y=3.0)
    resolve_collisions([rock, scissors])
    assert scissors.type == Type.ROCK
    assert rock.type == Type.ROCK


def test_weaker_does_not_convert_stronger():
    scissors = Agent(id=0, type=Type.SCISSORS, x=3.0, y=3.0)
    rock = Agent(id=1, type=Type.ROCK, x=3.0, y=3.0)
    resolve_collisions([scissors, rock])
    assert scissors.type == Type.ROCK
    assert rock.type == Type.ROCK


def test_same_type_untouched():
    a = Agent(id=0, type=Type.PAPER, x=3.0, y=3.0)
    b = Agent(id=1, type=Type.PAPER, x=3.0, y=3.0)
    resolve_collisions([a, b])
    assert a.type == Type.PAPER
    assert b.type == Type.PAPER


def test_winning_type_detected():
    env = RPSEnv(board_size=10, agents_per_type=2, seed=1)
    env.reset()
    for agent in env.agents:
        agent.type = Type.ROCK
    assert env.winning_type == Type.ROCK


def test_episode_ends_on_full_dominance():
    env = RPSEnv(board_size=10, agents_per_type=2, seed=1)
    env.reset()
    for agent in env.agents:
        agent.type = Type.ROCK
    actions = {a.id: (0.0, 0.0) for a in env.agents}
    _, _, done, info = env.step(actions)
    assert done is True
    assert info["winning_type"] == Type.ROCK


def test_episode_ends_after_length_limit():
    env = RPSEnv(board_size=8, agents_per_type=2, episode_length=5, seed=3)
    env.reset()
    rng = random.Random(99)
    for _ in range(6):
        if env.done:
            break
        actions = {a.id: (rng.uniform(-1, 1), rng.uniform(-1, 1)) for a in env.agents}
        env.step(actions)
    assert env.done is True


def test_random_episode_never_raises():
    env = RPSEnv(board_size=16, agents_per_type=6, seed=7)
    env.reset()
    rng = random.Random(42)
    steps = 0
    while not env.done:
        actions = {a.id: (rng.uniform(-1, 1), rng.uniform(-1, 1)) for a in env.agents}
        _, _, done, info = env.step(actions)
        steps += 1
        assert steps <= env.episode_length
        total = sum(info["populations"].values())
        assert total == len(env.agents)
    assert env.done is True


def test_create_agents_uses_all_types():
    rng = random.Random(1)
    agents = create_agents(rng, board_size=10, agents_per_type=4)
    types = {a.type for a in agents}
    assert types == {Type.ROCK, Type.PAPER, Type.SCISSORS}
