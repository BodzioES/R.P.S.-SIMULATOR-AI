from app.config import OBS_WINDOW
from app.env.entities import Type
from app.env.rps_env import RPSEnv


def test_window_and_own_shapes():
    env = RPSEnv(board_size=10, agents_per_type=1, seed=1)
    env.reset()
    obs = env.observations()
    assert len(obs) == 3
    window, own, wall, pop = next(iter(obs.values()))
    assert len(window) == OBS_WINDOW
    assert len(window[0]) == OBS_WINDOW
    assert len(window[0][0]) == 3
    assert len(own) == 3
    assert len(wall) == 4
    assert len(pop) == 3


def test_own_onehot_matches_type():
    env = RPSEnv(board_size=10, agents_per_type=1, seed=1)
    env.reset()
    obs = env.observations()
    for a in env.agents:
        _, own, _, _ = obs[a.id]
        assert own[a.type.value] == 1.0
        assert sum(own) == 1.0


def test_wall_distances():
    env = RPSEnv(board_size=10, agents_per_type=1, seed=1)
    env.reset()
    agents = env.agents
    agents[0].x, agents[0].y = 2.0, 8.0
    _, _, wall, _ = env.observations()[agents[0].id]
    assert abs(wall[0] - 0.2) < 0.01  # x/board
    assert abs(wall[1] - 0.8) < 0.01  # y/board


def test_center_excludes_self():
    env = RPSEnv(board_size=10, agents_per_type=1, seed=1)
    env.reset()
    agents = env.agents
    a0 = agents[0]
    a0.x, a0.y = 5.0, 5.0
    agents[1].x, agents[1].y = 1.0, 1.0
    agents[2].x, agents[2].y = 2.0, 2.0
    obs = env.observations()
    r = OBS_WINDOW // 2
    window, own, _, _ = obs[a0.id]
    assert window[r][r][a0.type.value] == 0.0
    assert own[a0.type.value] == 1.0


def test_enemy_appears_at_relative_position():
    env = RPSEnv(board_size=10, agents_per_type=1, seed=1)
    env.reset()
    agents = env.agents
    rock = agents[0]
    paper = agents[1]
    rock.x, rock.y = 5.0, 5.0
    paper.x, paper.y = 6.0, 5.0
    agents[2].x, agents[2].y = 9.0, 9.0
    obs = env.observations()
    r = OBS_WINDOW // 2
    window, own, _, _ = obs[rock.id]
    assert own[Type.ROCK.value] == 1.0
    assert window[r][r + 1][Type.PAPER.value] == 1.0


def test_far_agent_not_in_window():
    env = RPSEnv(board_size=10, agents_per_type=1, seed=1)
    env.reset()
    agents = env.agents
    rock = agents[0]
    paper = agents[1]
    rock.x, rock.y = 5.0, 5.0
    paper.x, paper.y = 0.0, 0.0
    agents[2].x, agents[2].y = 2.0, 2.0
    obs = env.observations()
    r = OBS_WINDOW // 2
    window, _, _, _ = obs[rock.id]
    for row in window:
        for cell in row:
            assert cell[Type.PAPER.value] == 0.0
