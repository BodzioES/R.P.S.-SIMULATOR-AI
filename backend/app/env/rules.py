from .entities import Type


def beats(a: Type, b: Type) -> bool:
    return (
        (a == Type.ROCK and b == Type.SCISSORS)
        or (a == Type.SCISSORS and b == Type.PAPER)
        or (a == Type.PAPER and b == Type.ROCK)
    )