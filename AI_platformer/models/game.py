import copy
import json
from copy import deepcopy

import pygame
#from pygame.examples.moveit import WIDTH, HEIGHT
from pygame.math import Vector2

from .configs import Configs
from .data import LINES, TYPE_PRIORITY
from .line import Line
from .player import Player
from .level import Level


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

    GROUND_Y = Configs.VIRTUAL_HEIGHT + 60
    print(Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT, "za game")

    def __init__(self, map="classic"):
        self.map = map
        self.players = set()
        self.collision_lines = []
        self.levels = []
        for i in range (10):
            b = i
            if i<10:
                b = '0' + str(i)
            self.levels.append(Level(str(b)))
        # self.levels.append(Level("00"))
        # self.levels.append(Level("01"))
        self.current_level = 0
        self.fullscreen = True  # or False if you start windowed
        self.x_offset = 0
        self.y_offset = 0
        self.star_position = Vector2(0, 0) #I wish to use this to reset the game when needed
        self.running = True
        self.show_lines = False
        #self.collision_lines = [((0, HEIGHT+60), (WIDTH+200, HEIGHT+60))]           #don't know why it has to be +200 on width but ok
        # self.collision_lines = sorted(
        #     [Line(x1, y1, x2, y2) for (x1, y1, x2, y2) in LINES],
        #     key=lambda line: TYPE_PRIORITY[line.type]
        # )
        #self.load_lines()       #not needed anymore
        self.testPlayer = Player(Vector2(Configs.VIRTUAL_WIDTH//2, Configs.VIRTUAL_HEIGHT-120))     #added for testing
        self.prev_click_pos = Vector2(-1, -1)
        self.game_paused = False
        pygame.init()


        # Set up the screen (you can adjust the dimensions)
        #self.screen = pygame.display.set_mode((800, 600))
        self.real_screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.RESIZABLE)  # or specific (1920, 1080)
        #added this line below so that the game is staretd in a smaller screen
        #self.window = pygame.display.set_mode((Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT))
        self.virtual_screen = pygame.Surface((Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT))
        #self.screen_rect = self.real_screen.get_rect()

        # background stuff
        self.bg_image_original = pygame.image.load("pictures/jk_00_00.png").convert()  # replace with your image
        self.bg_visible = True

        self.scaled_bg = self.get_scaled_bg((Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT))

        pygame.display.set_caption("Jump Emperor")

        #load levels with information
        for level in self.levels:
            level.load_assets()

        # Set the clock for FPS control
        self.clock = pygame.time.Clock()

        #adding player and all that:
        #nah, not for now. bust later we need to add a lot of players, maybe not here but..

    def get_scaled_bg(self, screen_size):
        return pygame.transform.scale(self.bg_image_original, screen_size)
    def add_player(self, player):
        self.players.add(player)

    def process_players(self):
        for player in self.players:
            print("Very important work")

    import json

    def save_lines(self, filename="lines/lines_00_01.json"):
        lines_data = []
        for line in self.collision_lines:
            lines_data.append({
                "x1": line.start.x,
                "y1": line.start.y,
                "x2": line.end.x,
                "y2": line.end.y
            })
        with open("lines/" + filename + ".json", "w") as f:
            json.dump(lines_data, f, indent=4)
            print("lines should be saved now")

    def load_lines(self, filename="lines_00_00.json"):
        try:
            print("load lines started")
            with open(filename, "r") as f:
                lines_data = json.load(f)
                for data in lines_data:
                    line = Line(data["x1"], data["y1"], data["x2"], data["y2"])
                    self.collision_lines.append(line)
            print("lines loaded successfully I guess", len(self.collision_lines))
        except FileNotFoundError:
            print("No saved lines found.")

    def collision_handler(self):

        for player in self.players:
            # if player.on_ground:
            #     continue
            was_on_ground = False
            for line in self.levels[player.current_level].lines:
                if not player.intersects_line(line):
                    continue

                type = line.type
                final_position = player.position

                #print("line hit: ", type, line.start, line.end, "player pos", player.position)
                match type:
                    case "horizontal":

                        if player.velocity.y > 0:
                            player.relative_position.y = line.y1 - player.height
                            player.position.y = Configs.VIRTUAL_HEIGHT - (line.y1 - player.height) + player.current_level * Configs.VIRTUAL_HEIGHT
                            player.velocity.y = 0
                            player.velocity.x = 0
                            player.on_ground = True
                            player.jumping = False
                            was_on_ground = True
                        if player.velocity.y == 0:
                            was_on_ground = True
                        elif player.velocity.y < 0:
                            player.relative_position.y = line.y1
                            player.position.y = Configs.VIRTUAL_HEIGHT - line.y1 + player.current_level * Configs.VIRTUAL_HEIGHT
                            player.velocity.y = 0

                    case "vertical":
                        if player.velocity.x > 0:
                            player.relative_position.x = line.x1 - player.width - 3
                            player.position.x = line.x1 - player.width - 3 + 0 #no sideways levels for now
                        elif player.velocity.x < 0:
                            player.relative_position.x = line.x1  + 2
                            player.position.x = line.x1 + 2 + 0 #for now

                        player.velocity.x *= -1

            if not was_on_ground and player.on_ground:
                print("Player walked off an edge!")
                player.on_ground = False

    def render(self):
        self.virtual_screen.fill(Game.WHITE)
        if self.bg_visible:
            #self.virtual_screen.blit(self.scaled_bg, (0, 0))
            self.virtual_screen.blit(self.levels[self.current_level].bg_picture_scaled, (0, 0))
        self.render_lines(self.virtual_screen)
        self.render_players(self.virtual_screen)

        screen_width, screen_height = self.real_screen.get_size()
        scale = min(screen_width / Configs.VIRTUAL_WIDTH, screen_height / Configs.VIRTUAL_HEIGHT)
        new_width = int(Configs.VIRTUAL_WIDTH * scale)
        new_height = int(Configs.VIRTUAL_HEIGHT * scale)
        scaled_surface = pygame.transform.scale(self.virtual_screen, (new_width, new_height))

        self.x_offset = (screen_width - new_width) // 2
        self.y_offset = (screen_height - new_height) // 2

        self.real_screen.fill((0, 0, 0))  # black bars
        self.real_screen.blit(scaled_surface, (self.x_offset, self.y_offset))
        pygame.display.flip()

    def add_cpu_players(self, count, start_position):
        self.start_position = start_position
        for _ in range(count):
            self.add_player(Player(position=copy.deepcopy(start_position)))

    def render_players(self, surface):
        max_level = 0
        for player in self.players:
            if player.current_level > max_level:
                max_level = player.current_level
        self.current_level = max_level
        for player in self.players:
            if player.current_level < self.current_level:
                continue
            pos = player.relative_position
            pygame.draw.rect(surface, Game.BLUE, (pos.x, pos.y, player.width, player.height))

    def render_lines(self, surface):
        if self.show_lines:
            for line in self.levels[self.current_level].lines:
                pygame.draw.line(surface, self.BLACK, line.start, line.end, 3)
            for line in self.collision_lines:  #only for now, remove later
                pygame.draw.line(surface, self.BLACK, line.start, line.end, 3)

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.window = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.RESIZABLE)
        else:
            self.window = pygame.display.set_mode((Configs.VIRTUAL_WIDTH, Configs.VIRTUAL_HEIGHT), pygame.RESIZABLE)

    def handle_events(self):            #added a few things
        # Handle game events (key presses, etc.)
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.MOUSEBUTTONDOWN and self.game_paused == True:
                if event.button == 1:  # Left mouse button
                    mouse_x, mouse_y = event.pos
                    screen_w, screen_h = pygame.display.get_surface().get_size()
                    if (mouse_x < self.x_offset or mouse_x > screen_w-self.x_offset):
                        continue
                    if (mouse_y < self.y_offset or mouse_y > screen_h-self.y_offset):
                        continue

                    # Calculate scale ratios
                    scale_x = Configs.VIRTUAL_WIDTH / (screen_w-2*self.x_offset)
                    scale_y = Configs.VIRTUAL_HEIGHT / (screen_h-2*self.y_offset)

                    mouse_x = mouse_x - self.x_offset
                    mouse_y = mouse_y - self.y_offset
                    print("clicked on pos: ", mouse_x, mouse_y)
                    # Scale to virtual coordinates
                    virtual_mouse_x = mouse_x * scale_x
                    virtual_mouse_y = mouse_y * scale_y
                    pos = (virtual_mouse_x, virtual_mouse_y)

                    if self.prev_click_pos.x != -1 and self.prev_click_pos.y != -1:
                        dx = abs(pos[0] - self.prev_click_pos.x)
                        dy = abs(pos[1] - self.prev_click_pos.y)

                        if dx < dy:
                            # Make it vertical: keep x same as previous
                            pos = (self.prev_click_pos.x, pos[1])
                        else:
                            # Make it horizontal: keep y same as previous
                            pos = (pos[0], self.prev_click_pos.y)

                        line = Line(self.prev_click_pos.x, self.prev_click_pos.y, pos[0], pos[1])
                        self.collision_lines.append(line)
                        self.prev_click_pos = Vector2(-1, -1)
                    else:
                        self.prev_click_pos = Vector2(pos[0], pos[1])
                    print("Left click at:", pos)
            if event.type == pygame.KEYDOWN:
                if event.unicode.isdigit():
                    print("Teleporting to map: " + str(int(event.unicode)))
                    self.teleport_to_map_number(int(event.unicode))
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_p:
                    self.game_paused = not self.game_paused
                if event.key == pygame.K_TAB:  # or K_s
                    self.toggle_fullscreen()
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                    self.bg_visible = not self.bg_visible
                if event.key == pygame.K_z:
                    print("clicked z")
                    self.collision_lines.pop()
                if event.key == pygame.K_s:
                    temp_name = self.current_level
                    if self.current_level < 10:
                        temp_name = '0' + str(self.current_level)
                    print("clicked s")
                    self.save_lines("lines_00_" + temp_name)       #da se sacuva sta se izcrta
                if event.key == pygame.K_i:
                    self.show_lines = not self.show_lines
                if event.key == pygame.K_o:
                    for p in self.players:
                        pl = p
                    print("player 0 pos: ", pl.position, " relative pos: ", pl.relative_position, " curr_lvl: ", pl.current_level, " game lvl: ", self.current_level)
                if event.key == pygame.K_r:
                    for p in self.players:
                        p.reset_player(self.start_position.x, self.start_position.y)

            # if event.type == pygame.VIDEORESIZE:
            #     screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            #     self.scaled_bg = self.get_scaled_bg(event.size)

    def update(self):
        # Update game state (move players, check collisions, etc.)
        self.process_players()

    def update_players(self):
        for player in self.players:

            player.play_move()
            player.apply_gravity()
            player.update_position()

    def update_test_player(self):    #for you to test it yourself

        for p in self.players:
            player = p

        #logic for move making here
        self.test_player_handler(player)

        if not self.game_paused:
            player.apply_gravity()
            player.update_position()

    def set_players_positions(self):
        for player in self.players:
            player.set_position(Configs.VIRTUAL_WIDTH//2, Configs.VIRTUAL_HEIGHT-60)        #I guess

    def run(self, paused=False):
        self.game_paused = paused
        while self.running:
            self.handle_events()  # <- process inputs / quit events
            #self.update()  # <- update game state
            if not self.game_paused:        #added for paused game (for drawing lines)
                self.update_players()
                self.collision_handler()
            self.render()  # <- draw stuff
            #self.render_players()
            self.clock.tick(Game.FPS)  # <- limit to 60 FPS

    def runTest(self, paused=False):
        self.game_paused = paused
        self.start_position = Vector2(Configs.VIRTUAL_WIDTH//2, Configs.VIRTUAL_HEIGHT - 120)
        player = Player(deepcopy(self.start_position))
        self.add_player(player)
        while self.running:
            #self.handle_events()  # <- process inputs / quit events

            self.update_test_player()
            self.collision_handler()
            self.render()
            self.clock.tick(Game.FPS)

    def close_game(self):
        pygame.quit()
        quit()

    def test_player_handler(self, player):      #suboptimal
        keys = pygame.key.get_pressed()
        #player = self.players[0]



        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

                # Start charging jump
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_p:
                    self.game_paused = not self.game_paused
                    return
                if event.key == pygame.K_SPACE and player.on_ground:
                    player.velocity.x = 0
                    player.current_charge += 0.5  # Start charging
                    player.charging = True
                    player.jump_direction = "up"  # Reset to straight jump
                if event.key == pygame.K_TAB:  # or K_s if you want S
                    self.toggle_fullscreen()
                if event.key == pygame.K_LSHIFT or event.key == pygame.K_RSHIFT:
                        self.bg_visible = not self.bg_visible
                if event.key == pygame.K_z:
                    print("clicked z")
                    self.collision_lines.pop()
                if event.key == pygame.K_s:
                    temp_name = self.current_level
                    if self.current_level < 10:
                        temp_name = '0' + str(self.current_level)
                    print("clicked s")
                    self.save_lines("lines_00_" + temp_name)  #
                if event.key == pygame.K_i:
                    self.show_lines = not self.show_lines
                if event.key == pygame.K_o:
                    for p in self.players:
                        pl = p
                    print("player 0 pos: ", pl.position, " relative pos: ", pl.relative_position, " curr_lvl: ", pl.current_level, " game lvl: ", self.current_level)
                if event.key == pygame.K_r:
                    for p in self.players:
                        p.reset_player(self.start_position.x, self.start_position.y)

            # if event.type == pygame.VIDEORESIZE:
            #     screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            #     self.scaled_bg = self.get_scaled_bg(event.size)

            if event.type == pygame.MOUSEBUTTONDOWN and self.game_paused==True:
                if event.button == 1:  # Left mouse button
                    mouse_x, mouse_y = event.pos
                    screen_w, screen_h = pygame.display.get_surface().get_size()
                    if (mouse_x < self.x_offset or mouse_x > screen_w - self.x_offset):
                        continue
                    if (mouse_y < self.y_offset or mouse_y > screen_h - self.y_offset):
                        continue

                    # Calculate scale ratios
                    scale_x = Configs.VIRTUAL_WIDTH / (screen_w - 2 * self.x_offset)
                    scale_y = Configs.VIRTUAL_HEIGHT / (screen_h - 2 * self.y_offset)

                    mouse_x = mouse_x - self.x_offset
                    mouse_y = mouse_y - self.y_offset
                    print("clicked on pos: ", mouse_x, mouse_y)
                    # Scale to virtual coordinates
                    virtual_mouse_x = mouse_x * scale_x
                    virtual_mouse_y = mouse_y * scale_y
                    pos = (virtual_mouse_x, virtual_mouse_y)

                    if self.prev_click_pos.x != -1 and self.prev_click_pos.y != -1:
                        dx = abs(pos[0] - self.prev_click_pos.x)
                        dy = abs(pos[1] - self.prev_click_pos.y)

                        if dx < dy:
                            # Make it vertical: keep x same as previous
                            pos = (self.prev_click_pos.x, pos[1])
                        else:
                            # Make it horizontal: keep y same as previous
                            pos = (pos[0], self.prev_click_pos.y)

                        line = Line(self.prev_click_pos.x, self.prev_click_pos.y, pos[0], pos[1])
                        self.collision_lines.append(line)
                        self.prev_click_pos = Vector2(-1, -1)
                    else:
                        self.prev_click_pos = Vector2(pos[0], pos[1])
                    print("Left click at:", pos)

            # Release jump
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE and player.charging:
                    # Determine jump direction only at the moment of release
                    player.jumping = True
                    if keys[pygame.K_LEFT]:
                        player.jump_direction = "left"
                    elif keys[pygame.K_RIGHT]:
                        player.jump_direction = "right"

                    # Calculate jump force
                    jump_force = player.current_charge #min_jump_strength + (jump_charge / abs(max_jump_strength)) * (max_jump_strength - min_jump_strength)
                    player.current_charge = 0
                    player.velocity.y = -jump_force

                    if player.jump_direction == "left":
                        player.velocity.x = -self.WALK_SPEED
                    elif player.jump_direction == "right":
                        player.velocity.x = self.WALK_SPEED
                    else:
                        player.velocity.x = 0  # Jump straight up

                    player.charging = False
                    player.on_ground = False  # Player leaves the ground

        if not self.game_paused:
            # Walking (only allowed if NOT charging a jump)
            if not player.charging and player.on_ground:
                player.current_charge = 0
                if keys[pygame.K_LEFT]:
                    player.velocity.x = -self.WALK_SPEED
                elif keys[pygame.K_RIGHT]:
                    player.velocity.x = self.WALK_SPEED
                else:
                    player.velocity.x = 0

            # Charge jump if space is held
            if player.charging:
                player.current_charge += 1      #yea, why not
                if player.current_charge >= abs(self.MAX_CHARGE):  # Auto-jump when fully charged
                    if keys[pygame.K_LEFT]:
                        player.jump_direction = "left"
                    elif keys[pygame.K_RIGHT]:
                        player.jump_direction = "right"

                    player.velocity.y = -self.MAX_CHARGE
                    if player.jump_direction == "left":
                        player.velocity.x = -self.WALK_SPEED
                    elif player.jump_direction == "right":
                        player.velocity.x = self.WALK_SPEED
                    else:
                        player.velocity.x = 0

                    player.charging = False
                    player.on_ground = False

    def insert_sorted(self, line):
        priority = TYPE_PRIORITY[line.type]
        for i, existing in enumerate(self.collision_lines):
            if TYPE_PRIORITY[existing.type] > priority:
                self.collision_lines.insert(i, line)
                return
        self.collision_lines.append(line)

    def teleport_to_map_number(self, map_number):
        pass
        #not finised yet, need to gather the data first