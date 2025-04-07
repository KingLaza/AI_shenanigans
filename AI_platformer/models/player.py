from pygame.math import Vector2
import uuid

class Player:
    def __init__(self, position=Vector2(0,0), width=30, height=60, type="CPU"):
        self.position = position
        self.width = width
        self.height = height
        self.id = uuid.uuid4()
        self.velocity = Vector2(0, 0)
        self.current_charge = 0
        self.jumping = False
        self.on_ground = True
        self.charging = False
        self.type = type
        self.moves = []


    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Player) and self.id == other.id

    def __repr__(self):
        return f"Player(id={self.id})"

