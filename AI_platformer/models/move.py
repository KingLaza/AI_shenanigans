import random

from .configs import Configs
from .move_type_management import get_random_move


class Move:
    def __init__(self,move_type = 0):
        self.move_type = get_random_move()
        self.strength = random.randint(0, Configs.MAX_CHARGE)
        self.initial_strength = self.strength

    def randomize(self):
        self.move_type = get_random_move()
        self.strength = random.randint(0, Configs.MAX_CHARGE)

