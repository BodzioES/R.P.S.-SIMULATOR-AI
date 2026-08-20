from dataclasses import dataclass
from enum import Enum


class Type(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


@dataclass
class Agent:
    id: int
    type: Type
    x: int
    y: int