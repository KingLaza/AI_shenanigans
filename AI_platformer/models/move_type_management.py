import random
from enum import auto, Flag


class MoveType(Flag):
    NONE = 0
    UP = auto()
    LEFT = auto()
    RIGHT = auto()

VALID_MOVES = [
    MoveType.UP,
    MoveType.LEFT,
    MoveType.RIGHT,
    MoveType.UP | MoveType.LEFT,
    MoveType.UP | MoveType.RIGHT,
]

def get_random_move():
    return random.choice(VALID_MOVES)