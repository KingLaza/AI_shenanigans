import random

from game import Game

class Move:
    def __init__(self,move_type = 0):
        self.move_type = random.randint(0, 4)
        self.strength = random.randint(0, Game.MAX_CHARGE)
        self.initial_strength = self.strength

    def randomize(self):
        self.move_type = random.randint(0,4)
        self.strength = random.randint(0, Game.MAX_CHARGE)

