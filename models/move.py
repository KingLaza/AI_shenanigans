
class Move:
    def __init__(self, charge_strength=0, left = False, right = False, move_time=0):
        self.charge_strength = charge_strength
        self.left = left
        self.right = right
        self.move_time = move_time