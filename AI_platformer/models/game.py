import pygame
from pygame.examples.moveit import WIDTH, HEIGHT

from .configs import Configs
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

    GROUND_Y = HEIGHT

    def __init__(self, map="classic"):
        self.map = map
        self.players = set()
        self.collision_lines = []
        self.running = True
        self.collision_lines = [((0, HEIGHT+60), (WIDTH+200, HEIGHT+60))]           #don't know why it has to be +200 on width but ok
        #self.collision_lines.append(Line(0, HEIGHT - 30, WIDTH, HEIGHT - 30))

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
        ground_y = self.GROUND_Y
        for player in self.players:
            if player.position.y >= ground_y:
                player.position.y = ground_y
                player.velocity.y = 0
                player.velocity.x = 0
                player.on_ground = True  # Player lands
                player.jumping = False

    def render(self):
        # Render game elements (players, background, etc.)
        self.screen.fill(Game.WHITE)  # Clear the screen with black
        # You can render players here using Pygame surfaces
        self.render_lines()
        # Draw ground
        self.render_players()


        pygame.display.flip()  # Update the screen

    def render_players(self):
        for player in self.players:
            pos = player.position
            pygame.draw.rect(self.screen, Game.BLUE, (pos.x, pos.y, player.width, player.height))

    def render_lines(self):
        for line in self.collision_lines:
            pygame.draw.line(self.screen, self.BLACK, line[0], line[1], 3)

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

    def set_players_positions(self):
        for player in self.players:
            player.set_position(WIDTH//2, HEIGHT-60)        #I guess

    def run(self):
        while self.running:
            #self.handle_events()  # <- process inputs / quit events
            #self.update()  # <- update game state
            self.update_players()
            self.collision_handler()
            self.render()  # <- draw stuff
            #self.render_players()
            self.clock.tick(Game.FPS)  # <- limit to 60 FPS

    def close_game(self):
        pygame.quit()
        quit()
