from pygame.math import Vector2
import uuid

from game import Game
from move import Move

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
        self.curr_move_count = 0
        self.moves = [Move() for _ in range(Game.MOVE_COUNT)]
        self.curr_move = self.moves[0]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Player) and self.id == other.id

    def __repr__(self):
        return f"Player(id={self.id})"

    def play_move(self):
        if self.curr_move_count > 19:
            return

        match self.curr_move.move_type:
            case 0:
                n = 3
                # skok samo gore
            case 1:
                n = 3
                # skok na levo
            case 2:
                n = 3
                # skok na desno
            case 3:
                n = 3
                # hodanje na levo
            case 4:
                n = 3
                # hodanje ne desno


        if self.curr_move.strength <= 0:
            self.curr_move_count += 1
            self.curr_move = None \
                if self.curr_move_count>=Game.MOVE_COUNT \
                else self.moves[self.curr_move_count]

