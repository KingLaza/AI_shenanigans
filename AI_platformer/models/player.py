from pygame.math import Vector2
import uuid

from .configs import Configs

from .move import Move

class Player:
    def __init__(self, position=Vector2(0,0), width=30, height=60, type="CPU"):
        self.position = position
        self.width = width
        self.height = height
        self.id = uuid.uuid4()
        self.velocity = Vector2(0, 0)
        self.jumping = False
        self.on_ground = True
        self.charging = False
        self.type = type
        self.curr_move_count = 0
        self.moves = [Move() for _ in range(Configs.MOVE_COUNT)]
        self.curr_move = self.moves[0]
        self.current_charge = self.curr_move.initial_strength  #change if non robot plays
        self.move_over = False

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Player) and self.id == other.id

    def __repr__(self):
        return f"Player(id={self.id})"

    def play_move(self):
        if self.move_over:
            self.move_over = False
            self.velocity = Vector2(0, 0)       #better to be here I guess (should be useless now)
            self.curr_move_count += 1
            if self.curr_move_count >= Configs.MOVE_COUNT:
                return
            else:
                self.curr_move = self.moves[self.curr_move_count]
                self.current_charge = self.curr_move.initial_strength

        if self.on_ground:
            match self.curr_move.move_type:
                case 0:
                    if self.current_charge == 0:
                        self.on_ground = False
                        self.jumping = True
                        self.velocity = Vector2(0, -self.curr_move.initial_strength)
                    if self.current_charge == -1:
                        self.move_over = True
                    self.current_charge -= 1
                    # skok samo gore
                case 1:
                    if self.current_charge == 0:
                        self.on_ground = False
                        self.jumping = True
                        self.velocity = Vector2(-Configs.MAX_CHARGE/2, -self.curr_move.initial_strength)
                    if self.current_charge == -1:
                        self.move_over = True
                    self.current_charge -= 1
                    # skok na levo
                case 2:
                    if self.current_charge == 0:
                        self.on_ground = False
                        self.jumping = True
                        self.velocity = Vector2(Configs.MAX_CHARGE / 2, -self.curr_move.initial_strength)
                    if self.current_charge == -1:
                        self.move_over = True
                        self.velocity = Vector2(0, 0)
                    self.current_charge -= 1
                    # skok na desno
                case 3:
                    if self.current_charge == 0:
                        self.move_over = True
                        self.velocity = Vector2(0, 0)
                    else:
                        self.velocity = Vector2(-Configs.WALK_SPEED, 0)
                        self.current_charge -= 1
                    # hodanje na levo
                case 4:
                    if self.current_charge == 0:
                        self.move_over = True
                        self.velocity = Vector2(0, 0)
                    else:
                        self.velocity = Vector2(Configs.WALK_SPEED, 0)
                        self.current_charge -= 1
                    # hodanje ne desno
        return

    def apply_gravity(self):
        if self.jumping:
            self.velocity.y += Configs.GRAVITY     #It's plus.. I didn't make a mistake..

    def update_position(self):
        if self.velocity != Vector2(0, 0):
            self.position.x += self.velocity.x
            self.position.y += self.velocity.y

    def set_position(self, x, y):
        self.position = Vector2(x, y)

