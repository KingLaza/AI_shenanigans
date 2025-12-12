from copy import deepcopy

from pygame.math import Vector2
import uuid

from .configs import Configs

from .move import Move

class Player:
    def __init__(self, position=Vector2(0,0), width=30, height=50, type="CPU"):
        self.position = Vector2(0,0)
        self.position.x = position.x
        self.position.y = Configs.VIRTUAL_HEIGHT - position.y
        self.relative_position = deepcopy(position)
        self.width = width
        self.height = height            #maybe move to configs later
        self.id = uuid.uuid4()
        self.velocity = Vector2(0, 0)
        self.current_level = 0      #change this later
        self.jumping = False
        self.on_ground = True
        self.charging = False
        self.type = type
        self.curr_move_count = 0
        self.moves = [Move() for _ in range(Configs.MOVE_COUNT)]
        self.curr_move = self.moves[0]
        self.current_charge = self.curr_move.initial_strength  #change if non robot plays
        self.move_over = False
        self.jump_direction = "up"      #I added this just for the test player

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
                        self.velocity = Vector2(-Configs.WALK_SPEED, -self.curr_move.initial_strength)
                    if self.current_charge == -1:
                        self.move_over = True
                    self.current_charge -= 1
                    # skok na levo
                case 2:
                    if self.current_charge == 0:
                        self.on_ground = False
                        self.jumping = True
                        self.velocity = Vector2(Configs.WALK_SPEED, -self.curr_move.initial_strength)
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
        if not self.on_ground:                      #removed jumping because of TestPlyaer.. might bring it back later
            self.velocity.y += Configs.GRAVITY     #It's plus.. I didn't make a mistake..

    def update_position(self):
        if self.velocity != Vector2(0, 0):
            self.position.x += self.velocity.x
            self.position.y -= self.velocity.y      #position.y will now act like height from the bottom of the map (0 to map_max_height)
            self.relative_position.x = (self.relative_position.x + self.velocity.x) % Configs.VIRTUAL_WIDTH
            self.relative_position.y = (self.relative_position.y + self.velocity.y) % Configs.VIRTUAL_HEIGHT
            self.current_level = max(0, int(self.position.y // Configs.VIRTUAL_HEIGHT))

    def set_position(self, x, y):
        self.position = Vector2(x, Configs.VIRTUAL_HEIGHT - y)
        self.relative_position = Vector2(x, y)

    def reset_player(self, x, y):
        self.position = Vector2(x, Configs.VIRTUAL_HEIGHT - y)
        self.relative_position = Vector2(x, y)
        self.velocity = Vector2(0, 0)
        self.on_ground = True
        self.jumping = False
        self.move_over = False
        self.charging = False
        self.curr_move_count = 0
        self.moves = [Move() for _ in range(Configs.MOVE_COUNT)]
        self.curr_move = self.moves[0]
        self.current_charge = self.curr_move.initial_strength

    def intersects_line(self, line):
        player_lines = self.get_rectangle_edges(self.relative_position.x, self.relative_position.y, self.width, self.height)
        line_vector = Vector2(line.x2-line.x1, line.y2-line.y1)

        for player_line in player_lines:
            if self.line_intersect(player_line[0], player_line[1], player_line[2], line):
                return True
        return False

    def line_intersect(self, p1, p2, line_type, line):

        # if (line_type != line.type):          #this didn't really do anything..
        #     return False
        # Convert player line segment to vector
        vect1 = p2 - p1
        # Convert the other line segment to a vector
        line_vector = Vector2(line.x2 - line.x1, line.y2 - line.y1)

        bDotDPerp = vect1.x * line_vector.y - vect1.y * line_vector.x

        # If the determinant is zero, the lines are parallel and do not intersect
        if bDotDPerp == 0:
            return False

        # Compute the vector from p1 to the line's start (line.x1, line.y1)
        vect2 = line.start - p1

        # Parametric intersection calculation
        t = (vect2.x * line_vector.y - vect2.y * line_vector.x) / bDotDPerp
        u = (vect2.x * vect1.y - vect2.y * vect1.x) / bDotDPerp

        # If t and u are between 0 and 1, the lines intersect
        if 0 <= t <= 1 and 0 <= u <= 1:
            return True

        return False

    def get_rectangle_edges(self, rect_x, rect_y, width, height):
        # Define the 4 corners of the rectangle
        top_left = Vector2(rect_x, rect_y)  # Top-left corner at (rect_x, rect_y)
        top_right = Vector2(rect_x + width, rect_y)  # Top-right corner
        bottom_left = Vector2(rect_x, rect_y + height)  # Bottom-left corner
        bottom_right = Vector2(rect_x + width, rect_y + height)  # Bottom-right corner

        # Return pairs of points representing the edges (lines) of the rectangle
        edges = [
            (top_left, top_right, "horizontal"),  # Top edge
            (top_right, bottom_right, "vertical"),  # Right edge
            (bottom_right, bottom_left ,"horizontal"),  # Bottom edge
            (bottom_left, top_left, "vertical")  # Left edge
        ]

        return edges

    #INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8

    # def compute_out_code(self, x, y, px1, py1, px2, py2):
    #     code = Player.INSIDE
    #
    #     if x < px1:
    #         code |= Player.LEFT
    #     elif x > px2:
    #         code |= Player.RIGHT
    #     if y < py1:
    #         code |= Player.BOTTOM
    #     elif y > py2:
    #         code |= Player.TOP
    #     return code
    #
    # def intersects_line(self, line):

