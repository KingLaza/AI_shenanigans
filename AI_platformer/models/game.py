import copy
import math

import pygame
from pygame.examples.moveit import WIDTH, HEIGHT

from .configs import Configs
from .data import LINES
from .player import Player
from .line import Line

class Game:
    MIN_CHARGE = Configs.MIN_CHARGE
    MAX_CHARGE = Configs.MAX_CHARGE
    MAX_JUMP_HEIGHT = Configs.MAX_JUMP_HEIGHT
    WALK_SPEED = Configs.WALK_SPEED      #could be more
    MOVE_COUNT = Configs.MOVE_COUNT
    FPS = Configs.FPS
    GRAVITY = Configs.GRAVITY
    BLUE = Configs.BLUE
    WHITE = Configs.WHITE
    BLACK = Configs.BLACK

    GROUND_Y = HEIGHT+60
    print(WIDTH, HEIGHT, "za game")

    def __init__(self, map="classic"):
        self.map = map
        self.players = set()
        self.collision_lines = []
        self.running = True
        #self.collision_lines = [((0, HEIGHT+60), (WIDTH+200, HEIGHT+60))]           #don't know why it has to be +200 on width but ok
        self.collision_lines = [Line(x1, y1, x2, y2) for (x1, y1, x2, y2) in LINES]

        pygame.init()

        # Set up the screen (you can adjust the dimensions)
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Game Title")

        # Set the clock for FPS control
        self.clock = pygame.time.Clock()

        #adding player and all that:
        #nah, not for now. bust later we need to add a lot of players, maybe not here but..

    def add_player(self, player):
        self.players.add(player)

    def process_players(self):
        for player in self.players:
            print("Very important work")

    def collision_handler(self):
        # for player in self.players:
        #     player.on_ground = False  # Reset before checking
        #
        #     for line in self.collision_lines:
        #         x1, y1, x2, y2 = line.x1, line.y1, line.x2, line.y2
        #         margin_of_error = 10
        #
        #         if line.type == "Horizontal":
        #             # Check if player is falling and is above the line, but close enough to "land" on it
        #             if (
        #                     player.velocity.y >= 0 and
        #                     player.position.y <= y1 and
        #                     abs(player.position.y + player.velocity.y - y1) < margin_of_error and  # Tweak threshold as needed
        #                     x1 <= player.position.x <= x2
        #             ):
        #                 player.position.y = y1
        #                 player.velocity.y = 0
        #                 player.on_ground = True
        #                 player.jumping = False
        #
        #
        #         elif line.type == "Vertical":
        #             # Normalize y1 and y2 to make sure y1 <= y2
        #             y_top = min(y1, y2)
        #             y_bottom = max(y1, y2)
        #             player_half_width = player.width / 2 if hasattr(player, 'width') else 0  # Optional
        #             if (
        #                 y_top <= player.position.y <= y_bottom and
        #                 abs(player.position.x - x1) <= margin_of_error + player_half_width
        #             ):
        #                 player.velocity.x = -player.velocity.x  # Bounce off
                        # Optional: nudge player slightly away to avoid sticking
                        # if player.position.x < x1:
                        #     player.position.x = x1 - (margin_of_error + 1)
                        # else:
                        #     player.position.x = x1 + (margin_of_error + 1)
            # Ground collision (if still in air and hits bottom)
        # for player in self.players:
        #     ground_y = self.GROUND_Y
        #     if player.position.y >= ground_y:
        #         player.position.y = ground_y
        #         player.velocity.y = 0
        #         player.velocity.x = 0
        #         player.on_ground = True
        #         player.jumping = False

        for player in self.players:
            # if player.on_ground:
            #     continue

            for line in self.collision_lines:
                if not player.intersects_line(line):
                    continue

                type = line.type
                final_position = player.position
                print("line hit: ", type, line.start, line.end, "player pos", player.position)
                match type:
                    case "horizontal":

                        if player.velocity.y > 0:
                            player.position.y = line.y1 - player.height
                            print(player.position, player.velocity)
                            player.velocity.y = 0
                            player.velocity.x = 0
                            player.on_ground = True
                            player.jumping = False
                        elif player.velocity.y < 0:
                            player.position.y = line.y1
                            player.velocity.y = 0

                    case "vertical":
                        distance_to_line = player.position.x - line.x1

                        move_distance = player.width + 12 if player.velocity.x > 0 else -12  # This is the minimum distance to move beyond the line

                        if distance_to_line > 0:
                            player.position.x = line.x1 + move_distance
                        else:  # Player is to the left of the line
                            player.position.x = line.x1 - move_distance


                        player.velocity.x *= -1







    def render(self):
        # Render game elements (players, background, etc.)
        self.screen.fill(Game.WHITE)  # Clear the screen with black
        # You can render players here using Pygame surfaces
        self.render_lines()
        # Draw ground
        self.render_players()


        pygame.display.flip()  # Update the screen

    def add_cpu_players(self, count, start_position):
        for _ in range(count):
            self.add_player(Player(position=copy.deepcopy(start_position)))

    def render_players(self):
        for player in self.players:
            pos = player.position
            pygame.draw.rect(self.screen, Game.BLUE, (pos.x, pos.y, player.width, player.height))

    def render_lines(self):
        for line in self.collision_lines:
            pygame.draw.line(self.screen, self.BLACK, line.start, line.end, 3)

    def handle_events(self):
        # Handle game events (key presses, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False


    def update(self):
        # Update game state (move players, check collisions, etc.)
        self.process_players()

    def update_players(self):
        for player in self.players:

            player.play_move()
            player.apply_gravity()
            player.update_position()

    def update_test_player(self):    #for you to test it yourself
        player = self.players[0]

        #logic for move making here

        player.apply_gravity()
        player.update_position()

    def set_players_positions(self):
        for player in self.players:
            player.set_position(WIDTH//2, HEIGHT-60)        #I guess

    def run(self):
        while self.running:
            self.handle_events()  # <- process inputs / quit events
            #self.update()  # <- update game state
            self.update_players()
            self.collision_handler()
            self.render()  # <- draw stuff
            #self.render_players()
            self.clock.tick(Game.FPS)  # <- limit to 60 FPS

    def runTest(self):
        while self.running:
            self.update_test_player()
            self.collision_handler()
            self.render()
            self.clock.tick(Game.FPS)

    def close_game(self):
        pygame.quit()
        quit()
