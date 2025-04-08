import pygame
from pygame.examples.moveit import WIDTH, HEIGHT

from player import Player

class Game:
    MIN_CHARGE = 1
    MAX_CHARGE = 60
    MAX_JUMP_HEIGHT = 100
    WALK_SPEED = 3      #could be more
    MOVE_COUNT = 20
    FPS = 60
    GRAVITY = (9.81 / FPS) * 3
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)

    def __init__(self, map="classic"):
        self.map = map
        self.players = set()
        self.collision_lines = []
        self.running = True

        pygame.init()

        # Set up the screen (you can adjust the dimensions)
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Game Title")

        # Set the clock for FPS control
        self.clock = pygame.time.Clock()

        #adding player and all that:
        self.add_player(Player())       #right?
        self.set_players_positions()

    def add_player(self, player):
        self.players.add(player)

    def process_players(self):
        for player in self.players:
            print("Very important work")

    def render(self):
        # Render game elements (players, background, etc.)
        self.screen.fill(Game.WHITE)  # Clear the screen with black
        # You can render players here using Pygame surfaces

        self.render_players()


        pygame.display.flip()  # Update the screen

    def render_players(self):
        for player in self.players:
            pos = player.position
            pygame.draw.rect(self.screen, Game.BLUE, (pos.x, pos.y, player.width, player.height))

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
            self.handle_events()  # <- process inputs / quit events
            self.update()  # <- update game state
            self.update_players()
            self.render()  # <- draw stuff
            self.render_players()
            self.clock.tick(Game.FPS)  # <- limit to 60 FPS

    def close_game(self):
        pygame.quit()
        quit()
