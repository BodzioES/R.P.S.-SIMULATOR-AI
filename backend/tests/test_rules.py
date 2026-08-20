from app.env.entities import Type
from app.env.rules import beats


def test_rock_beats_scissors():
    assert beats(Type.ROCK, Type.SCISSORS) is True


def test_scissors_beats_paper():
    assert beats(Type.SCISSORS, Type.PAPER) is True


def test_paper_beats_rock():
    assert beats(Type.PAPER, Type.ROCK) is True


def test_loser_does_not_beat_winner():
    assert beats(Type.SCISSORS, Type.ROCK) is False
    assert beats(Type.PAPER, Type.SCISSORS) is False
    assert beats(Type.ROCK, Type.PAPER) is False


def test_same_type_never_converts():
    for t in Type:
        assert beats(t, t) is False